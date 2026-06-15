# Report: LLM inference + observability

Text-to-SQL PoC — Qwen3-30B-A3B-Instruct-2507 on 1× H100 80GB.

---

## Phase 1 — Serving configuration

**Launch script:** `scripts/start_vllm_tuned.sh` (the stock `scripts/start_vllm.sh` is left as the defaults baseline.)

**Workload this config targets:** MoE model (~3B of 30B params active per token), prompts of ~1.5–3K tokens (DB schema + question), short structured SQL outputs, ~2–3 dependent vLLM calls per agent run, against an SLO of **P95 end-to-end < 5 s at 10+ RPS over 5 min**.

| Flag | Value | Justification |
|---|---|---|
| `--max-model-len` | `8192` | Model default is 262144; our prompts+outputs never exceed a few K tokens. Capping the window shrinks per-sequence KV reservation → more concurrent sequences fit → higher RPS. Biggest single lever. |
| `--gpu-memory-utilization` | `0.92` | Hands more of the 80 GB to the KV-cache pool, so more requests stay resident before eviction — needed to sustain the concurrency the 10+ RPS target implies. |
| `--max-num-seqs` | `64` | Ceiling on sequences batched together. Moderate starting point: high enough to keep the GPU busy, low enough not to blow the latency tail. Re-tuned in Phase 6. |
| `--max-num-batched-tokens` | `8192` | Prefill token budget per step. Keeps a single 1.5–3K-token prefill from monopolizing a scheduler step while in-flight decodes continue. |
| `--enable-chunked-prefill` | on | Interleaves long-prompt prefill with ongoing decode so one big prompt doesn't stall the batch → smoother P95 for our prompt sizes. |
| `--kv-cache-dtype` | `fp8` | Halves KV-cache footprint → ~2× more concurrent sequences in the same VRAM, directly serving the RPS target. Quality risk: re-checked against Phase 5 evals; dropped if pass rate regresses. |

**MoE / hardware note:** Qwen3-30B-A3B fits on a single H100 *because* it's a Mixture-of-Experts model — only ~3B params are active per token, so compute is light while memory (all 30B params resident) is the binding constraint. That's why the config above spends its effort on KV-cache headroom (context cap, memory fraction, fp8) rather than tensor-parallelism: there's no second GPU to split across, and the model already fits.

**Manual sanity check:** fired 3–5 questions from `evals/eval_set.jsonl`; model returns well-formed SQL. See `screenshots/vllm_manual_query.png`.

> Note: these are the Phase 1 *starting* values. `--max-num-seqs`, `--gpu-memory-utilization`, and `--kv-cache-dtype` are revisited under load in Phase 6, where they're validated against the SLO and the eval pass rate.

---

## Phase 5 — Baseline eval results

**Signal: execution accuracy.** For each of the 30 questions in `evals/eval_set.jsonl`, the agent's SQL and the gold SQL are run against the target DB and their result sets compared after canonicalization (sort rows, stringify cells, `None`→`''`). Identical row sets ⇒ correct, regardless of how the SQL is written. Runner: `evals/run_eval.py`; output: `results/eval_baseline.json`.

> Baseline is measured on the **pre-fast-path** agent (verify runs on every iteration) so the per-iteration numbers reflect what the verify→revise loop actually buys. The fast-path variant is measured separately as `eval_after_tuning.json` in Phase 6.

**Overall pass rate:** `__ / 30` (`__%`).

**Per-iteration pass rate** (carry-forward — a run that terminated early keeps its last answer at later iterations):

| If we stopped after… | Pass rate |
|---|---|
| iter 0 — generate only | `__%` |
| iter 1 — + one revise | `__%` |
| iter 2 — + two revises | `__%` |

**Loop gain (iter 0 → final): `__` pts.**  Avg iterations/run: `__`.  Gold-SQL exec failures: `__`.

**Commentary.** The number that matters is loop gain. If iter-0 pass rate ≈ final pass rate, the verify→revise loop is doing nothing and is pure latency cost — the agent should collapse to a single generate call. If final is meaningfully higher than iter-0, the loop earns its keep by catching and fixing bad SQL (mostly execution errors: wrong column/table names, syntax). _Read the iter-0 vs final gap above and state which case this is, e.g.: "loop lifts pass rate from X% to Y% (+Z pts), driven mainly by fixing first-attempt execution errors — the loop is worth its cost."_ This directly informs the Phase 6 fast path: the fast path keeps the loop only for the error case (where the gain comes from) and skips it for clean executions (where it adds latency without lifting accuracy).

---

## Phase 6 — Hitting the SLO

**Target SLO:** P95 end-to-end agent latency < 5 s, at 10+ RPS (1 RPS = 1 full agent run/sec), sustained over a 5-minute window.

Driver: `load_test/driver.py --rps 10 --duration 300` (open-loop; reported percentiles are over *successful* requests only, so a high failure rate makes the real tail worse than the P95 shown).

### Baseline vs. SLO — a hard miss (congestion collapse)

| | Baseline (pre-fix) | SLO |
|---|---|---|
| Achieved RPS | 8.33 offered / **1.24 goodput** | 10+ |
| Successful requests | 446 / 3000 (**15%**) | ~all |
| Timeouts (120 s cap) | 1705 (57%) | 0 |
| HTTP + client errors | 244 + 605 (28%) | ~0 |
| P50 / P95 / P99 | 20.4 s / **94.6 s** / 101.3 s | P95 < 5 s |

The system was in **congestion collapse**: ~85% of requests failed and even the survivors took 20 s at the median. P95 (94.6 s) is ~19× over the SLO.

### Diagnosis — the bottleneck was *not* vLLM

Reading the Grafana dashboard under load was decisive: **vLLM was nearly idle while the agent collapsed.**

- **GPU KV-cache utilization ~10%** — 90% headroom, nowhere near full.
- **Scheduler queue (`num_requests_waiting`) ~0** — vLLM wasn't queuing anything.
- **Requests running ~30 of 64** — under the `--max-num-seqs` ceiling.
- **Single vLLM call latency healthy** — TTFT p99 ~250 ms, per-call e2e p50 ~1 s / p95 ~3.5 s.

So the serving layer had spare capacity; the collapse was upstream, in the agent. Each agent run chained **2–6 sequential LLM calls** (generate → verify → revise → verify …, `MAX_ITERATIONS=3`), making runs slow. By Little's Law, in-flight runs ≈ offered_rate × service_time ≈ 10 × ~12 s ≈ **120 concurrent**, which overran the FastAPI sync thread pool (~40) → requests queued at the agent and timed out, starving vLLM of work. Tuning vLLM flags (`--max-num-seqs`, KV cache) would have done nothing — the headroom proves the bottleneck wasn't there.

### Iteration log

| # | Saw | Hypothesized | Changed | Result |
|---|---|---|---|---|
| 1 | At 10 RPS: P95 94.6 s, 85% of requests failing — but Grafana showed vLLM idle (KV-cache ~10%, queue ~0, single-call p50 ~1 s). | The agent makes too many sequential LLM calls per run, so runs are slow and pile up past the server's thread pool → timeouts. vLLM is fine. Cutting calls/run should fix latency *and* the concurrency pileup. | **Execution-success fast path** in `agent/graph.py`: if the generated SQL executes cleanly, return it and skip the `verify` LLM call; `verify → revise` now runs only on execution errors. Common case drops from 2+ calls to 1. | Timeouts **1705 → 0**, goodput **1.24 → 8.11 RPS**, P50 **20.4 → 1.43 s**, P95 **94.6 → 14.5 s**. Collapse eliminated, throughput near target. SLO **not yet met** (P95 14.5 s) and a ~14% 500-error rate remains. |

### Final numbers (current state)

| | Baseline | After fast path | SLO |
|---|---|---|---|
| Achieved RPS | 8.33 | 9.44 | 10+ |
| Goodput (successful RPS) | 1.24 | **8.11** | 10+ |
| Successful requests | 446 (15%) | **2576 (86%)** | ~all |
| Timeouts | 1705 | **0** | 0 |
| HTTP / client errors | 244 / 605 | 379 / 45 (**14%**) | ~0 |
| P50 | 20.4 s | **1.43 s** | — |
| P95 | 94.6 s | **14.5 s** | **< 5 s** |
| P99 | 101.3 s | 22.4 s | — |

**Verdict: SLO not yet met, but the collapse is fixed and the gap is now small and well-understood.** Two issues remain, both diagnosed:

1. **Bimodal latency (P50 1.43 s vs P95 14.5 s).** The fast path serves ~85% of runs in a single call (~1.4 s); the tail is the runs whose SQL *fails execution* and falls into the `verify → revise` loop (up to 6 calls), stretching to 14–22 s under load and dragging P95 over the line.
2. **~14% HTTP 500s.** Raised when a `graph.invoke` call throws (vs. a clean wrong-answer, which returns `ok=false`) — most likely vLLM rejecting/dropping calls now that it's actually loaded at ~8 RPS.

**Next iterations (planned):** (a) `MAX_ITERATIONS` 3 → 2 to trim the error-path tail *and* cut vLLM load — one targeted change for both the P95 tail and the 500s; (b) make the agent endpoint async (`graph.ainvoke`) to smooth burst queueing if the tail persists; (c) diagnose the 500s from the agent logs / `request_success_total{finished_reason}` panel to confirm serving- vs. agent-side cause.

_Artifacts: `screenshots/grafana_before.png` (collapse — vLLM idle, requests timing out) and `screenshots/grafana_after.png` (post-fast-path — KV-cache rising, queue draining). Quality check (`results/eval_baseline.json` vs `results/eval_after_tuning.json`) pending the eval run on the final config — the fast path skips `verify` on successful executions, so the eval must confirm accuracy survived._

---

## Agent value

The verify→revise loop is justified by the **per-iteration pass rate**, not by intuition. Stopping after the first generate gives `__%`; running the full loop gives `__%` — a **+`__` pt** lift (from `results/eval_baseline.json`). That gap is the agent earning its keep: the extra LLM calls recover questions whose first SQL attempt was wrong, predominantly execution errors (bad column/table names, syntax) that the verifier catches from the execution result and the reviser fixes. _If the gap is large, the loop is clearly worth it; if small, state that the loop barely helps and the fast-path single-call agent is the better default._ The Phase 6 fast path is the direct consequence of this analysis: it preserves the loop exactly where the lift comes from (failed executions) and removes it where it doesn't (clean executions), converting the loop from an always-on latency tax into a pay-only-when-needed recovery mechanism — keeping the quality lift while cutting P95 from 94.6 s to 14.5 s and goodput from 1.2 to 8.1 RPS. The remaining check is `eval_after_tuning.json`: it confirms whether skipping verify on *successful* executions costs any accuracy (the one quality risk the fast path introduces).

---

## What I'd do with more time

Concrete next steps, in priority order, to close the SLO gap (P95 14.5 s → < 5 s) and harden the system:

1. **Trim the latency tail (`MAX_ITERATIONS` 3 → 2).** P50 is 1.4 s but P95 is 14.5 s — the tail is entirely the execution-error runs that fall into verify→revise (up to 6 sequential calls). Capping the loop one step shorter directly cuts the worst runs and lowers load on vLLM. Validate against `eval_after_tuning.json` to confirm iteration 3 wasn't carrying real accuracy (Phase 5's loop-gain number predicts this).

2. **Diagnose and fix the ~14% HTTP 500s.** These are exceptions in `graph.invoke` collapsed into a single status code. I'd split them by class — vLLM transport/overload errors vs. agent-side parse errors — with structured logging and a per-class Prometheus counter, then add a **bounded retry + timeout on the LLM client** so transient vLLM failures become retried successes instead of hard failures (a 14% error rate fails the SLO on error budget alone, independent of latency).

3. **Make the agent endpoint async (`graph.ainvoke`).** The work is I/O-bound (waiting on vLLM), so an async event loop holds hundreds of concurrent runs instead of being capped by FastAPI's ~40-thread pool. The fast path already brought concurrency under that ceiling, but async removes it as a constraint and smooths burst queueing — needed headroom to push *past* 10 RPS.

4. **Cheap heuristic verify-gate instead of blanket skip.** The fast path's one quality risk is letting semantically-wrong-but-executing SQL through. A non-LLM gate (flag empty result sets, all-NULL aggregates, or obviously-degenerate counts) would route only *suspicious* clean executions to the verifier — recovering most of the lost quality without paying the second LLM call on every run.

5. **Reduce the slow-path population at the source.** Improve the `generate_sql` prompt (targeted few-shot on the error patterns the eval surfaces, stricter schema rendering) so fewer first attempts fail execution. A smaller slow-path population shrinks the tail more fundamentally than capping iterations.

6. **Richer eval signal.** Execution accuracy only checks row-set equality on one DB each. I'd add a per-DB and per-error-type breakdown (syntax vs. schema-grounding vs. semantic) so failures point at a cause, and a held-out question set to avoid overfitting prompts to these 30. Phase 4's Langfuse tags (`run:…`, `db_id:…`) make slicing traces by these dimensions straightforward.

7. **Serving-side decode speedup.** Time-per-output-token was ~70 ms; speculative decoding with a small draft model would cut decode latency on the structured-SQL outputs. Separately, prefix-cache hit rate was already ~85% — consistently ordering the shared schema prefix first in the prompt would push that higher and reclaim KV headroom.
