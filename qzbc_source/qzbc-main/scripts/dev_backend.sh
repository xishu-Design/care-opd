#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/backend"

export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
export KNOWLEDGE_BACKEND="${KNOWLEDGE_BACKEND:-lightrag}"
export LIGHTRAG_ENABLED="${LIGHTRAG_ENABLED:-true}"
export LIGHTRAG_MODE="${LIGHTRAG_MODE:-api}"
export LIGHTRAG_API_BASE_URL="${LIGHTRAG_API_BASE_URL:-http://127.0.0.1:9621}"
export LIGHTRAG_API_KEY="${LIGHTRAG_API_KEY:-guardtcm-local-change-me}"
export LIGHTRAG_WORKING_DIR="${LIGHTRAG_WORKING_DIR:-$ROOT_DIR/runtime/lightrag/staging/current}"
export LIGHTRAG_INPUT_DIR="${LIGHTRAG_INPUT_DIR:-$ROOT_DIR/runtime/lightrag/input}"
export QA_KNOWLEDGE_ENABLED="${QA_KNOWLEDGE_ENABLED:-true}"
export QA_KNOWLEDGE_INDEX_PATH="${QA_KNOWLEDGE_INDEX_PATH:-$ROOT_DIR/runtime/qa_knowledge/qa_index.db}"
export GRAPH_BACKEND="${GRAPH_BACKEND:-graphiti}"
export MEMORY_BACKEND="${MEMORY_BACKEND:-mem0}"
export GRAPHITI_DB_PATH="${GRAPHITI_DB_PATH:-$ROOT_DIR/runtime/graphiti/guardtcm.db}"
export GRAPHITI_TELEMETRY_ENABLED="${GRAPHITI_TELEMETRY_ENABLED:-false}"
export MEM0_DIR="${MEM0_DIR:-$ROOT_DIR/runtime/mem0/meta}"
export MEM0_QDRANT_PATH="${MEM0_QDRANT_PATH:-$ROOT_DIR/runtime/mem0/qdrant}"
export MEM0_HISTORY_DB_PATH="${MEM0_HISTORY_DB_PATH:-$ROOT_DIR/runtime/mem0/history.db}"
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-false}"
export CONVERSATION_CONTEXT_ENABLED="${CONVERSATION_CONTEXT_ENABLED:-true}"
export CONVERSATION_RECENT_TURNS="${CONVERSATION_RECENT_TURNS:-6}"
export CONVERSATION_CONTEXT_MAX_CHARS="${CONVERSATION_CONTEXT_MAX_CHARS:-6000}"

"$ROOT_DIR/.conda/bin/alembic" upgrade head
"$ROOT_DIR/.conda/bin/python" scripts/seed_demo.py
if [[ ! -f "$QA_KNOWLEDGE_INDEX_PATH" || "$ROOT_DIR/backend/guardtcm_dev.db" -nt "$QA_KNOWLEDGE_INDEX_PATH" ]]; then
  "$ROOT_DIR/.conda/bin/python" scripts/build_qa_knowledge_index.py
fi
"$ROOT_DIR/.conda/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port "${BACKEND_PORT:-8000}" --reload
