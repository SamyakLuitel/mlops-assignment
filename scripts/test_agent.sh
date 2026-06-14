#!/usr/bin/env bash
#
# Phase 3 smoke test: fire eval questions at the running agent server and report,
# per question, how many iterations it took and whether the verify->revise loop
# fired. Confirms the Phase 3 deliverable "at least one question triggers a revise".
#
# Prereqs: agent server up (uvicorn agent.server:app --port 8001) and vLLM running.
#
# Usage:
#   bash scripts/test_agent.sh             # all questions in the eval set
#   LIMIT=5 bash scripts/test_agent.sh     # only the first 5
#
# Env overrides:
#   AGENT_URL  agent /answer endpoint (default: http://localhost:8001/answer)
#   EVAL_FILE  eval set path          (default: evals/eval_set.jsonl)
#   LIMIT      max questions to fire   (default: all)

set -uo pipefail

AGENT_URL="${AGENT_URL:-http://localhost:8001/answer}"
EVAL_FILE="${EVAL_FILE:-evals/eval_set.jsonl}"
LIMIT="${LIMIT:-0}"

AGENT_URL="$AGENT_URL" EVAL_FILE="$EVAL_FILE" LIMIT="$LIMIT" python3 - <<'PY'
import json, os, urllib.request

agent = os.environ["AGENT_URL"]
path = os.environ["EVAL_FILE"]
limit = int(os.environ.get("LIMIT", "0"))

rows = [json.loads(l) for l in open(path) if l.strip()]
if limit > 0:
    rows = rows[:limit]

revised_idx, failed = [], 0
print(f"Firing {len(rows)} questions at {agent}\n")

for i, r in enumerate(rows):
    q = r.get("question")
    db = r.get("db_id") or r.get("db")          # handle either key name
    payload = json.dumps({"question": q, "db": db}).encode()
    req = urllib.request.Request(
        agent, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=120))
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"Q{i:2d} db={db} REQUEST FAILED: {e}")
        continue
    nodes = [h.get("node") for h in resp.get("history", [])]
    revised = "revise" in nodes
    if revised:
        revised_idx.append(i)
    print(f"Q{i:2d} db={db:<22} iters={resp.get('iterations')} "
          f"ok={resp.get('ok')} revised={revised}")

print(f"\n{len(revised_idx)} / {len(rows)} questions triggered a revise: {revised_idx}")
if failed:
    print(f"{failed} request(s) failed - is the agent server (port 8001) and vLLM up?")
PY
