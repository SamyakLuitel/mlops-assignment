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

_TODO: overall pass rate, per-iteration pass rate, commentary on whether the verify→revise loop earns its keep._

---

## Phase 6 — Hitting the SLO

_Baseline performance vs. SLO, then the iteration log._

| # | Saw | Hypothesized | Changed | Result |
|---|---|---|---|---|
| 1 | | | | |

_TODO: final numbers, whether quality survived (compare `results/eval_baseline.json` vs `results/eval_after_tuning.json`)._

---

## Agent value

_TODO: one paragraph — did the verify→revise loop help? Cite the per-iteration pass rate._

---

## What I'd do with more time

_TODO: be specific._
