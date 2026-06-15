#! /usr/bin/env bash
# Start vLLM server for Phase 3 (vLLM + LangGraph + LangChain).

uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001