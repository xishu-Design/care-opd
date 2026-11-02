#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

TEST_REQUESTS="${QZBC_DATA_ROOT}/compiled/test/qzbc_on_policy_requests.jsonl"
EVAL_ROOT="${QZBC_RUN_ROOT}/evaluation"
QZBC_URL="${QZBC_URL:-http://127.0.0.1:8000}"
QZBC_BASE_AGENT_URL="${QZBC_BASE_AGENT_URL:-}"
QZBC_TRAINED_AGENT_URL="${QZBC_TRAINED_AGENT_URL:-}"
require_file "${TEST_REQUESTS}"
mkdir -p -- "${EVAL_ROOT}"

overwrite_args=()
if [[ "${OVERWRITE_EVAL:-0}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

# Main model-boundary evaluation: exact qzbc messages, exact Ollama JSON schema,
# same held-out examples, and only the model tag changes.
run_module care_opd.qzbc_native_eval \
  --requests "${TEST_REQUESTS}" \
  --base-model "${BASE_OLLAMA_MODEL}" \
  --trained-model "${TRAINED_OLLAMA_MODEL}" \
  --ollama-base-url "${OLLAMA_BASE_URL}" \
  --output "${EVAL_ROOT}/native_heldout.json" \
  "${overwrite_args[@]}"

# qzbc's built-in 12-case LLM provider/model integration comparison.  This
# calls LLMExperimentRunner/LLMService directly; it does not execute the full
# Agent graph, tools, retrieval, or memory path.
skip_llm_experiment="${SKIP_QZBC_LLM_EXPERIMENT_AB:-${SKIP_QZBC_AGENT_AB:-0}}"
if [[ "${skip_llm_experiment}" != "1" ]]; then
  if [[ -z "${QZBC_TOKEN:-}" && -z "${QZBC_PASSWORD:-}" ]]; then
    printf '%s\n' \
      'Skip qzbc built-in LLM integration comparison: set QZBC_TOKEN or QZBC_PASSWORD to enable it.' >&2
  else
    run_module care_opd.qzbc_bridge compare \
      --qzbc-url "${QZBC_URL}" \
      --base-model "${BASE_OLLAMA_MODEL}" \
      --trained-model "${TRAINED_OLLAMA_MODEL}" \
      --provider ollama \
      --suite full \
      --output "${EVAL_ROOT}/qzbc_llm_experiment_ab.json" \
      "${overwrite_args[@]}"
  fi
fi

# Optional real Agent integration smoke.  It requires two qzbc processes
# because each process caches OLLAMA_MODEL.  Both processes must use the same
# source/configuration and equivalent seeded data, but isolated writable
# SQLite/Graphiti/Mem0 stores.
if [[ -n "${QZBC_BASE_AGENT_URL}" || -n "${QZBC_TRAINED_AGENT_URL}" ]]; then
  if [[ -z "${QZBC_BASE_AGENT_URL}" || -z "${QZBC_TRAINED_AGENT_URL}" ]]; then
    printf '%s\n' \
      'Set both QZBC_BASE_AGENT_URL and QZBC_TRAINED_AGENT_URL for full-Agent A/B.' >&2
    exit 2
  fi
  agent_reset_args=()
  if [[ "${QZBC_AGENT_RESET_DEMO:-1}" == "1" ]]; then
    agent_reset_args+=(--reset-demo)
  fi
  run_module care_opd.qzbc_agent_ab \
    --base-url "${QZBC_BASE_AGENT_URL}" \
    --trained-url "${QZBC_TRAINED_AGENT_URL}" \
    --base-model "${BASE_OLLAMA_MODEL}" \
    --trained-model "${TRAINED_OLLAMA_MODEL}" \
    --provider ollama \
    --password "${QZBC_AGENT_PASSWORD:-${QZBC_PASSWORD:-password123}}" \
    --output "${EVAL_ROOT}/qzbc_full_agent_ab.json" \
    "${agent_reset_args[@]}" \
    "${overwrite_args[@]}"
else
  printf '%s\n' \
    'Full-Agent A/B not requested; set QZBC_BASE_AGENT_URL and QZBC_TRAINED_AGENT_URL to enable it.' >&2
fi

printf 'Evaluation outputs: %s\n' "${EVAL_ROOT}"
