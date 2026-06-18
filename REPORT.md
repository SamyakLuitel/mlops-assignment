# Report: LLM inference + observability

Text-to-SQL PoC — `Qwen3-30B-A3B-Instruct-2507` on 1× H100 80GB. Analysts ask
questions in English; a LangGraph agent generates SQLite, runs it against a BIRD
database, and returns rows. The platform SLO under test:

> **P95 end-to-end agent latency < 5 s, at 10+ RPS (1 RPS = one full agent run/sec), sustained over a 5-minute window.**

---

## Phase 1 — Serving configuration

**Launch script:** `scripts/start_vllm_tuned.sh` (stock `scripts/start_vllm.sh` left as the defaults baseline).

**Workload it targets:** an MoE model (~3B of 30B params active per token), prompts of ~1.5–3K tokens (schema + question), short structured SQL outputs, and ~1–3 dependent vLLM calls per agent run.

| Flag | Value | Justification |
|---|---|---|
| `--max-model-len` | `8192` | Model default is 262144; our prompts+outputs never approach it. A smaller window shrinks per-sequence KV reservation → more concurrent sequences fit → higher RPS. Biggest single lever. |
| `--gpu-memory-utilization` | `0.92` | Hands more of the 80 GB to the KV-cache pool, so more requests stay resident before eviction — needed to sustain the concurrency 10+ RPS implies. |
| `--max-num-seqs` | `64` | Ceiling on sequences batched together. High enough to keep the GPU busy, low enough to bound the latency tail. |
| `--max-num-batched-tokens` | `8192` | Prefill token budget per step; keeps one 1.5–3K-token prefill from monopolizing a scheduler step while decodes continue. |
| `--enable-chunked-prefill` | on | Interleaves long-prompt prefill with ongoing decode so one big prompt doesn't stall the batch → smoother P95. |
| `--kv-cache-dtype` | `fp8` | Halves KV-cache footprint → ~2× more concurrent sequences in the same VRAM, directly serving the RPS target. Quality risk re-checked against Phase 5 evals. |

**MoE / hardware note.** Qwen3-30B-A3B fits on one H100 *because* it is Mixture-of-Experts — only ~3B params are active per token, so compute is light while memory (all 30B params resident) is the binding constraint. That is why the config spends its effort on KV-cache headroom (context cap, memory fraction, fp8) rather than tensor parallelism: there is no second GPU to split across, and the model already fits.

**Sanity check:** fired questions from `evals/eval_set.jsonl` manually; the model returns well-formed SQL (`screenshots/vllm_manual_query.png`).

---

## Phase 2 — Observability dashboard

Dashboard JSON: `infra/grafana/provisioning/dashboards/serving.json`. Built from vLLM's `/metrics` to answer "is it slow, and *where* in the request lifecycle?":

- **Latency** — TTFT and end-to-end request latency percentiles (p50/p95/p99) from the vLLM histograms; time-per-output-token for decode cost.
- **Throughput** — prompt + generation tokens/sec, requests running vs. waiting, successful-request rate.
- **KV cache** — GPU KV-cache utilization and `num_requests_waiting`, the headroom/eviction signals.

Every panel reacts under load (`screenshots/grafana_serving.png`). The dashboard is what made the Phase 6 diagnosis possible: it showed the serving layer was *idle* while the agent collapsed (below).

---

## Phase 3 — Agent design

LangGraph graph (`agent/graph.py`), final shape:

```
START → attach_schema → generate_sql → execute → verify
                                                    │
                                        ok=true ────┤────► END
                                                    │
                                       ok=false ────┴────► revise → execute → verify (loop)
```

`verify` always runs (so every Langfuse trace shows the full waterfall), but it carries a **cheap gate**: if the SQL executed cleanly it returns `ok=true` with **no LLM call**, so the common case is a single LLM call (generate only). The verify LLM is spent only when execution **errored** — the one case a revise can actually fix. The loop is capped at `MAX_ITERATIONS = 2` (one revise). Prompts live in `agent/prompts.py`. The server (`agent/server.py`) is async (`graph.ainvoke`) and exposes `POST /answer`.

---

## Phase 4 — Agent tracing

Langfuse captures the LangGraph spans via the callback handler (`langfuse.langchain.CallbackHandler`, picked up from `.env`). Each `/answer` run forwards request tags (`run:…`, `db_id:…`) as Langfuse trace metadata, so traces are filterable. A trace shows the `generate_sql → verify → (revise)` waterfall with per-span prompt, response, latency, and token count (`screenshots/langfuse_trace.png`); the tagged trace list is in `screenshots/langfuse_tags.png`.

---

## Phase 5 — Baseline eval results

**Signal: execution accuracy.** For each of the 30 questions in `evals/eval_set.jsonl`, the agent's SQL and the gold SQL are run against the target DB and their result sets compared after canonicalization (sort rows, stringify cells, `None`→`''`). Identical row sets ⇒ correct, regardless of how the SQL is written. Runner: `evals/run_eval.py`; output: `results/eval_baseline.json`.

**Overall pass rate: 9 / 30 (30%).**

| If we stopped after… | Pass rate |
|---|---|
| iter 0 — generate only | 30% (9/30) |
| iter 1 — + one revise | 30% (9/30) |
| iter 2 — + two revises | 30% (9/30) |

**Loop gain (iter 0 → final): 0 pts.** Avg iterations/run: 1.6. Gold-SQL exec failures: 0.

**Commentary.** The number that matters is loop gain, and here it is **zero**: iter-0 pass rate equals final pass rate. Every question the agent gets right, it gets right on the first generate; no wrong first answer was converted to a correct one by a revise. The loop *did* fire (avg 1.6 iterations/run — verify flagged answers and revise re-ran), it just never turned a miss into a hit. The misses are not the kind the loop is built for: with `gold_exec_failures = 0`, the 21 misses are genuine agent errors, mostly *semantic* mistakes that still execute cleanly (a generic "is this plausible?" verifier doesn't catch them and doesn't steer the reviser to the fix), rather than the broken-SQL cases (bad column/table, syntax) the loop handles well. So on this set the always-on loop is pure latency cost with no accuracy return — which is exactly what motivates the Phase 6 fast path.

The post-tuning eval (`results/eval_after_tuning.json`, fast-path agent) measured **10/30 (33.3%)** with avg 1.03 iterations/run — accuracy did not regress (within noise), confirming that skipping the verify LLM on clean executions costs nothing the eval can see.

---

## Phase 6 — Hitting the SLO

Driver: `load_test/driver.py --rps 10 --duration 300` (open-loop; percentiles are over *successful* requests only, so a high failure rate makes the real tail worse than shown).

### Baseline — a hard miss (congestion collapse)

| | Baseline | SLO |
|---|---|---|
| Achieved RPS | 8.33 offered / **1.67 goodput** | 10+ |
| Successful requests | 600 / 3000 (**20%**) | ~all |
| Timeouts (120 s cap) | 1557 (52%) | 0 |
| HTTP + client errors | 239 + 604 (28%) | ~0 |
| P50 / P95 / P99 | 46.1 s / **112.4 s** / 117.0 s | P95 < 5 s |

The system was in **congestion collapse**: ~80% of requests failed and survivors took 46 s at the median. P95 was ~22× over the SLO.

### Diagnosis — the bottleneck was *not* vLLM

The Grafana dashboard was decisive: **vLLM was nearly idle while the agent collapsed.** KV-cache utilization ~10%, `num_requests_waiting` ~0, single-call latency healthy (TTFT p99 ~250 ms, per-call p50 ~1 s). The serving layer had spare capacity; the collapse was upstream. Each agent run chained multiple sequential LLM calls, and the **synchronous** FastAPI endpoint capped concurrency at its ~40-thread pool. By Little's Law, 10 RPS × multi-second runs ⇒ 100+ in-flight, overrunning the pool → requests queued at the agent and timed out, starving vLLM of work. Tuning vLLM flags would have done nothing — the headroom proves the bottleneck wasn't there.

### Iteration log

| # | Saw | Hypothesized | Changed | Result |
|---|---|---|---|---|
| 1 | P95 112 s, ~80% failing, but Grafana showed vLLM idle (KV ~10%, queue ~0). | Too many sequential LLM calls per run + a sync thread-pool ceiling → pileup → timeouts. vLLM is fine. | **Fast path** (skip the verify LLM on clean executions) + **async endpoint** (`graph.ainvoke`, async LLM nodes) + timeout/retry on the client. | Collapse eliminated: timeouts **1557 → 0**, P50 **46 → 2.5 s**, P95 **112 → ~19 s**, goodput **1.67 → ~8 RPS**. |
| 2 | After moving the gate inside `verify` and gating the LLM verify on `row_count > 0` to catch empty results: P50 jumped to **6.9 s**, P95 to **38.9 s** — a regression. | Most *wrong* SQL returns 0 rows, so the "suspicious" path fired on the median request, adding verify+revise calls. Phase 5 showed the loop recovers nothing anyway (loop gain 0), so this was pure latency. | Reverted: gate skips the verify LLM on **any** clean execution; `MAX_ITERATIONS` **3 → 2**. | P50 **6.9 → 2.5 s**, P95 **38.9 → 19.4 s**. Regression undone. |
| 3 | P99 28 s, **max 118 s** tail; HTTP-error count unchanged with vs. without retries (~375). | The ~12% errors are **non-transient** (retries don't reduce them), so an aggressive retry budget (`timeout=20, max_retries=2`) just inflated the tail. | `timeout` **20 → 10**, `max_retries` **2 → 1**, `max_tokens=512`. | P95 **19.4 → 18.0 s**, P50 → 2.25 s. Marginal — and the 117 s max *persisted*, confirming the tail is the error-path runs and the errors themselves, not retry storms. |

### Final numbers vs. SLO

| | Baseline | **Final** | SLO |
|---|---|---|---|
| Goodput (successful RPS) | 1.67 | **7.84** | 10+ |
| Successful requests | 600 (20%) | **2560 (85%)** | ~all |
| Timeouts | 1557 | **0** | 0 |
| HTTP / client errors | 239 / 604 | 372 / 68 (**~15%**) | ~0 |
| P50 | 46.1 s | **2.25 s** | — |
| P95 | 112.4 s | **18.0 s** | **< 5 s** |
| P99 / max | 117.0 s | 25.8 s / 117.4 s | — |

**Verdict: SLO missed, but the congestion collapse is fixed and the remaining gap is fully diagnosed.** Three findings, all metric-grounded:

1. **Bimodal latency (P50 2.25 s vs P95 18 s).** ~85% of runs are a single fast-path call (~2 s); the tail is the runs whose SQL *errors* and falls into verify→revise. P95 < 5 s requires that slow-path population below ~5% — a `generate_sql` grounding improvement we did not make.
2. **~15% errors, unmoved by retries.** Non-transient, almost certainly **context overflow** (prompt > `--max-model-len 8192` on large-schema DBs). This fails the error budget *and* caps goodput at 7.84 — it is the single highest-leverage unfixed item.
3. **117 s max persists** after cutting retries, so that pathological run is the error path itself, not a retry storm.

_Artifacts: `screenshots/grafana_before.png` (collapse — vLLM idle, requests timing out), `screenshots/grafana_after.png` (post-fix — KV-cache rising, queue draining), `screenshots/grafana_eval_run.png`._

---

## Agent value

Judged by the **per-iteration pass rate**, not intuition: the verify→revise loop **does not earn its keep on this eval set**. Iter-0 and final pass rates are both 30% — a **+0 pt** lift (`results/eval_baseline.json`). The loop fires (avg 1.6 iterations/run) but never converts a miss to a hit, because the misses are semantic errors that execute cleanly rather than the broken-SQL cases the loop catches. The right default is therefore the **single-call fast path**, with the loop preserved only for execution errors — which is cheap and occasionally useful — rather than an always-on tax. The fast path is what took P95 from 112 s to 18 s and goodput from 1.67 to 7.84 RPS at **no measured accuracy cost** (post-tuning eval 33.3% vs baseline 30%).

---

## What I'd do with more time

In priority order, to close the SLO gap (P95 18 s → < 5 s, goodput 7.84 → 10+):

1. **Diagnose and fix the ~15% errors.** Log the exception class behind each HTTP 500 during a load run to confirm they are context overflow; then either raise `--max-model-len` (costs KV headroom) or, better, **trim the schema** to the tables likely relevant to the question so prompts stay well under 8192 tokens. This fixes the error budget *and* lifts goodput toward 10 — the biggest single win available.
2. **Shrink the slow-path population at the source.** Improve `generate_sql` grounding (targeted few-shot on the error patterns the eval surfaces; inject exact categorical values from the DB so filters use the right literals/casing) so fewer first attempts error. Fewer slow-path runs pulls P95 down into the fast path far more fundamentally than capping iterations.
3. **A deterministic, non-LLM verifier.** Cheap Python checks (NULL aggregate, zero rows for a non-aggregate question, duplicate rows without `DISTINCT`, aggregate-shape mismatch) that emit a *specific* revise hint. This would move loop gain above 0 — making the loop earn its keep — at essentially no latency, unlike the generic LLM verifier.
4. **Push past 10 RPS once errors are fixed.** The async event loop has headroom the sync version never did; with the error budget under control, re-run the sweep to find the real throughput ceiling.
5. **Richer eval signal.** Per-DB and per-error-type breakdown (syntax vs. schema-grounding vs. semantic) so failures point at a cause, plus a held-out question set to avoid overfitting prompts to these 30.
