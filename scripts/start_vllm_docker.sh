#!/usr/bin/env bash
#
# Launch vLLM via the official Docker image.
#
# Why Docker: the local uv venv resolved transformers 5.x, which vLLM 0.10.2
# can't use (crashes in the tokenizer with "Qwen2Tokenizer has no attribute
# all_special_tokens_extended"). The official image ships a self-consistent,
# compatible dependency set, so it sidesteps the broken venv entirely.
#
# Networking: Prometheus (in docker-compose) scrapes host.docker.internal:8000,
# i.e. the HOST's port 8000. Publishing -p 8000:8000 below keeps that scrape
# working, so the Grafana dashboard in Phase 2 sees /metrics as before.
#
# Tuned flags target the text-to-SQL workload (see REPORT.md Phase 1).
# Re-tune the numbers in Phase 6 against the load test.

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
IMAGE="vllm/vllm-openai:v0.10.2"

exec docker run --rm \
    --gpus all \
    --ipc=host \
    -p 8000:8000 \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "$IMAGE" \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8
