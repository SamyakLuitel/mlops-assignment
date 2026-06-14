#!/usr/bin/env bash
#
# Tuned vLLM launch for the text-to-SQL workload (Phase 1 starting point).
# Original untouched launcher lives in scripts/start_vllm.sh.
#
# Workload profile (from README):
#   - Model: Qwen3-30B-A3B (MoE, ~3B active/token) on 1x H100 80GB
#   - Prompts: ~1.5-3K tokens (schema + question)
#   - Outputs: short, structured SQL (a few hundred tokens)
#   - ~2-3 dependent calls per agent run; SLO P95 < 5s @ 10+ RPS
#
# Every flag below has a one-line rationale -> copy these into REPORT.md.
# Re-tune the numbers in Phase 6 against the load test; this is the baseline.
#
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    `# Cap context to just above real prompt+output size. Model default is 262144;` \
    `# we never use it, and the smaller window frees KV cache for more concurrency.` \
    --max-model-len 8192 \
    `# Claim more of the 80GB for KV cache -> more requests resident -> higher RPS.` \
    --gpu-memory-utilization 0.92 \
    `# Concurrency ceiling. Start moderate; raise in Phase 6 if KV cache has headroom,` \
    `# lower if P95 latency blows past the 5s SLO under load.` \
    --max-num-seqs 64 \
    `# Prefill batch budget. Keeps 1.5-3K-token prefills from monopolizing a step` \
    `# while decode of in-flight requests continues (pairs with chunked prefill).` \
    --max-num-batched-tokens 8192 \
    `# Interleave long-prompt prefill with ongoing decode so one big prompt doesn't` \
    `# stall everyone -> smoother tail latency for our 1.5-3K-token prompts.` \
    --enable-chunked-prefill \
    `# fp8 KV cache packs ~2x more sequences into the same VRAM -> more concurrency` \
    `# for the 10+ RPS target. Drop this flag if you see a quality regression in evals.` \
    --kv-cache-dtype fp8
