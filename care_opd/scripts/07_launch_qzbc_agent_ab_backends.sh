#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

require_dir "${QZBC_ROOT}/backend"

QZBC_PYTHON="${QZBC_PYTHON:-python3}"
BASE_AGENT_PORT="${BASE_AGENT_PORT:-8001}"
TRAINED_AGENT_PORT="${TRAINED_AGENT_PORT:-8002}"
OLLAMA_EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL:-qwen3-embedding:4b}"
LIGHTRAG_API_BASE_URL="${LIGHTRAG_API_BASE_URL:-http://127.0.0.1:9621}"
OLLAMA_GENERATE_TIMEOUT_SECONDS="${OLLAMA_GENERATE_TIMEOUT_SECONDS:-600}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
AB_RUNTIME="${AB_RUNTIME:-${QZBC_RUN_ROOT}/agent_ab_runtime_${run_stamp}}"
BACKEND_DIR="${QZBC_ROOT}/backend"

mkdir -p -- \
  "${AB_RUNTIME}/base/graphiti" \
  "${AB_RUNTIME}/base/mem0/qdrant" \
  "${AB_RUNTIME}/base/mem0/meta" \
  "${AB_RUNTIME}/trained/graphiti" \
  "${AB_RUNTIME}/trained/mem0/qdrant" \
  "${AB_RUNTIME}/trained/mem0/meta"

run_module care_opd.qzbc_bridge check-ollama \
  --ollama-base-url "${OLLAMA_BASE_URL}" \
  --model "${BASE_OLLAMA_MODEL}" \
  --model "${TRAINED_OLLAMA_MODEL}"

arm_env() {
  local arm="$1"
  local model="$2"
  shift 2
  env \
    APP_ENV=development \
    DATABASE_URL="sqlite:///${AB_RUNTIME}/${arm}/qzbc.db" \
    LLM_PROVIDER=ollama \
    LLM_ENABLED=true \
    LLM_ALLOW_FALLBACK=false \
    OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
    OLLAMA_MODEL="${model}" \
    OLLAMA_EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL}" \
    OLLAMA_GENERATE_TIMEOUT_SECONDS="${OLLAMA_GENERATE_TIMEOUT_SECONDS}" \
    KNOWLEDGE_BACKEND=lightrag \
    LIGHTRAG_ENABLED=true \
    LIGHTRAG_MODE=api \
    LIGHTRAG_API_BASE_URL="${LIGHTRAG_API_BASE_URL}" \
    GRAPH_BACKEND=graphiti \
    GRAPHITI_DB_PATH="${AB_RUNTIME}/${arm}/graphiti/guardtcm.db" \
    GRAPHITI_GROUP_PREFIX="qzbc_ab_${arm}" \
    MEMORY_BACKEND=mem0 \
    MEM0_DIR="${AB_RUNTIME}/${arm}/mem0/meta" \
    MEM0_QDRANT_PATH="${AB_RUNTIME}/${arm}/mem0/qdrant" \
    MEM0_HISTORY_DB_PATH="${AB_RUNTIME}/${arm}/mem0/history.db" \
    MEM0_COLLECTION_NAME="qzbc_ab_${arm}_memories" \
    LANGGRAPH_THREAD_ID_PREFIX="qzbc_ab_${arm}" \
    "$@"
}

initialize_arm() {
  local arm="$1"
  local model="$2"
  (
    cd -- "${BACKEND_DIR}"
    arm_env "${arm}" "${model}" "${QZBC_PYTHON}" -m alembic upgrade head
    arm_env "${arm}" "${model}" "${QZBC_PYTHON}" scripts/seed_demo.py
  )
}

initialize_arm base "${BASE_OLLAMA_MODEL}"
initialize_arm trained "${TRAINED_OLLAMA_MODEL}"

start_arm() {
  local arm="$1"
  local model="$2"
  local port="$3"
  local log="${AB_RUNTIME}/${arm}/backend.log"
  local pid_file="${AB_RUNTIME}/${arm}/backend.pid"
  (
    cd -- "${BACKEND_DIR}"
    nohup env \
      APP_ENV=development \
      DATABASE_URL="sqlite:///${AB_RUNTIME}/${arm}/qzbc.db" \
      LLM_PROVIDER=ollama \
      LLM_ENABLED=true \
      LLM_ALLOW_FALLBACK=false \
      OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
      OLLAMA_MODEL="${model}" \
      OLLAMA_EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL}" \
      OLLAMA_GENERATE_TIMEOUT_SECONDS="${OLLAMA_GENERATE_TIMEOUT_SECONDS}" \
      KNOWLEDGE_BACKEND=lightrag \
      LIGHTRAG_ENABLED=true \
      LIGHTRAG_MODE=api \
      LIGHTRAG_API_BASE_URL="${LIGHTRAG_API_BASE_URL}" \
      GRAPH_BACKEND=graphiti \
      GRAPHITI_DB_PATH="${AB_RUNTIME}/${arm}/graphiti/guardtcm.db" \
      GRAPHITI_GROUP_PREFIX="qzbc_ab_${arm}" \
      MEMORY_BACKEND=mem0 \
      MEM0_DIR="${AB_RUNTIME}/${arm}/mem0/meta" \
      MEM0_QDRANT_PATH="${AB_RUNTIME}/${arm}/mem0/qdrant" \
      MEM0_HISTORY_DB_PATH="${AB_RUNTIME}/${arm}/mem0/history.db" \
      MEM0_COLLECTION_NAME="qzbc_ab_${arm}_memories" \
      LANGGRAPH_THREAD_ID_PREFIX="qzbc_ab_${arm}" \
      "${QZBC_PYTHON}" -u -m uvicorn app.main:app \
      --host 127.0.0.1 --port "${port}" >"${log}" 2>&1 &
    printf '%s\n' "$!" >"${pid_file}"
  )
}

start_arm base "${BASE_OLLAMA_MODEL}" "${BASE_AGENT_PORT}"
start_arm trained "${TRAINED_OLLAMA_MODEL}" "${TRAINED_AGENT_PORT}"

wait_for_health() {
  local name="$1"
  local port="$2"
  local log="${AB_RUNTIME}/${name}/backend.log"
  for _ in $(seq 1 90); do
    if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
      return
    fi
    sleep 2
  done
  printf 'qzbc %s backend did not become healthy; inspect %s\n' "${name}" "${log}" >&2
  exit 3
}

wait_for_health base "${BASE_AGENT_PORT}"
wait_for_health trained "${TRAINED_AGENT_PORT}"

printf '%s\n' "${AB_RUNTIME}" >"${QZBC_RUN_ROOT}/LATEST_AGENT_AB_RUNTIME.txt"
printf 'Base Agent:    http://127.0.0.1:%s (%s)\n' \
  "${BASE_AGENT_PORT}" "${BASE_OLLAMA_MODEL}"
printf 'Trained Agent: http://127.0.0.1:%s (%s)\n' \
  "${TRAINED_AGENT_PORT}" "${TRAINED_OLLAMA_MODEL}"
printf 'Runtime and logs: %s\n' "${AB_RUNTIME}"
printf '%s\n' \
  'Run scripts/04_evaluate_formal.sh with QZBC_BASE_AGENT_URL and QZBC_TRAINED_AGENT_URL set to these URLs.'
