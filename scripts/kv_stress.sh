#!/usr/bin/env bash
#
# KV-cache stress load against vLLM (Phase 2).
#
# Unlike smoke_load.sh (short, near-identical prompts -> high prefix-cache hit,
# low KV usage), this fires LONG + UNIQUE prompts with LARGE max_tokens at high
# concurrency. That maximizes GPU KV-cache utilization and pushes "waiting" > 0,
# which is what makes the KV-cache panel climb toward its eviction threshold.
#
# Run smoke_load.sh and kv_stress.sh back to back to contrast the two regimes:
#   smoke  -> prefix-cache hit rate HIGH, KV usage LOW
#   stress -> prefix-cache hit rate LOW,  KV usage HIGH
#
# Usage:
#   bash scripts/kv_stress.sh                # defaults: 120 requests, 40 in flight
#   bash scripts/kv_stress.sh 240 60         # push harder
#   URL=http://localhost:8003/v1/chat/completions bash scripts/kv_stress.sh   # from laptop
#
# Env overrides:
#   URL        full chat-completions endpoint (default VM: localhost:8000)
#   MODEL      served model id
#   MAX_TOKENS output length per request (default 1200; bigger = more KV held)

set -uo pipefail

URL="${URL:-http://localhost:8000/v1/chat/completions}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MAX_TOKENS="${MAX_TOKENS:-1200}"
TOTAL="${1:-120}"        # total requests to fire
CONCURRENCY="${2:-40}"   # how many in flight at once

echo "KV stress: $TOTAL requests ($CONCURRENCY in flight), max_tokens=$MAX_TOKENS at $URL"
echo "Watch Grafana: 'vLLM serving' -> GPU KV-cache utilization + Concurrency (running vs waiting)"

for i in $(seq 1 "$TOTAL"); do
  # Build a long, UNIQUE schema per request so each gets its own KV blocks
  # (defeats prefix caching, which would otherwise keep KV usage low).
  COLS=""
  for c in $(seq 1 40); do COLS="${COLS}col_${i}_${c} TEXT, "; done

  curl -s "$URL" -H "Content-Type: application/json" -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role":"system","content":"You are a verbose data analyst. Explain your reasoning step by step in detail, then give the SQL."},
      {"role":"user","content":"Schema:\nCREATE TABLE big_'"$i"' ('"$COLS"' id INT);\nQuestion: Walk through, in detail, how you would compute year-over-year growth per column group, then write the SQL. Be thorough. Run '"$i"'."}
    ], "temperature":0.8, "max_tokens":'"$MAX_TOKENS"'
  }' > /dev/null &

  # throttle: cap concurrent in-flight requests
  if (( i % CONCURRENCY == 0 )); then
    wait
  fi
done
wait

echo "KV stress done ($TOTAL requests)"
