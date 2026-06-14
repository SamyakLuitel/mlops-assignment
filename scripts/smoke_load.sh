#!/usr/bin/env bash
#
# Smoke load against vLLM to make the Grafana panels react (Phase 2).
# Fires text-to-SQL chat completions directly at vLLM (not the agent), so it
# works with just vLLM running. Good for grabbing screenshots/grafana_serving.png.
#
# Usage:
#   bash scripts/smoke_load.sh                 # defaults: 60 requests, 10 in flight
#   bash scripts/smoke_load.sh 200 30          # 200 requests, 30 in flight (push KV cache)
#   URL=http://localhost:8003/v1/chat/completions bash scripts/smoke_load.sh   # from laptop
#
# Env overrides:
#   URL    full chat-completions endpoint (default host VM: localhost:8000)
#   MODEL  served model id

set -uo pipefail

URL="${URL:-http://localhost:8000/v1/chat/completions}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TOTAL="${1:-60}"        # total requests to fire
CONCURRENCY="${2:-10}"  # how many in flight at once

echo "Firing $TOTAL requests ($CONCURRENCY in flight) at $URL"
echo "Watch Grafana: http://localhost:3000 -> 'vLLM serving' (time range: last 15m)"

for i in $(seq 1 "$TOTAL"); do
  curl -s "$URL" -H "Content-Type: application/json" -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role":"system","content":"You are a text-to-SQL assistant. Reply with one SQL query only."},
      {"role":"user","content":"Schema:\nCREATE TABLE sales (id INT, product TEXT, region TEXT, amount REAL, sold_at TEXT);\nCREATE TABLE products (id INT, name TEXT, category TEXT);\nQuestion: Monthly total sales per region for the last year, ordered by month. Variation '"$i"'."}
    ], "temperature":0.7, "max_tokens":400
  }' > /dev/null &

  # throttle: cap concurrent in-flight requests
  if (( i % CONCURRENCY == 0 )); then
    wait
  fi
done
wait

echo "burst done ($TOTAL requests)"
