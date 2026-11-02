#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
QZBC_ROOT="${QZBC_ROOT:-${PROJECT_ROOT}/../qzbc_source/qzbc-main}"
CARE_CONFIG="${CARE_CONFIG:-${PROJECT_ROOT}/configs/qzbc_formal.yaml}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3.5-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ROLLOUT_NPROC="${ROLLOUT_NPROC:-${NPROC_PER_NODE}}"
ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-}"

QZBC_DATA_ROOT="${QZBC_DATA_ROOT:-${PROJECT_ROOT}/data/qzbc_formal}"
QZBC_RUN_ROOT="${QZBC_RUN_ROOT:-${PROJECT_ROOT}/runs/qzbc_base_model}"
QZBC_SEED_TOTAL="${QZBC_SEED_TOTAL:-12000}"
QZBC_DEV_RATIO="${QZBC_DEV_RATIO:-0.10}"
QZBC_TEST_RATIO="${QZBC_TEST_RATIO:-0.10}"
ONLINE_ROUNDS="${ONLINE_ROUNDS:-3}"
ROLLOUTS_PER_ROUND="${ROLLOUTS_PER_ROUND:-12000}"
ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-1200}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
TRAIN_SEED="${TRAIN_SEED:-42}"

BASE_OLLAMA_MODEL="${BASE_OLLAMA_MODEL:-qwen3.5:4b}"
TRAINED_OLLAMA_MODEL="${TRAINED_OLLAMA_MODEL:-qwen3.5-4b-care-opd:latest}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PROJECT_ROOT QZBC_ROOT CARE_CONFIG QZBC_DATA_ROOT QZBC_RUN_ROOT

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'Missing required file: %s\n' "${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    printf 'Missing required directory: %s\n' "${path}" >&2
    exit 2
  fi
}

run_module() {
  local module="$1"
  shift
  "${PYTHON_BIN}" -m "${module}" "$@"
}

run_qzbc_rollout() {
  local parallel_args=(--num-workers "${ROLLOUT_NPROC}")
  if [[ -n "${ROLLOUT_DEVICES}" ]]; then
    parallel_args+=(--devices "${ROLLOUT_DEVICES}")
  fi
  run_module care_opd.qzbc_rollout "${parallel_args[@]}" "$@"
}

json_int_field() {
  local path="$1"
  local field="$2"
  "${PYTHON_BIN}" - "${path}" "${field}" <<'PY'
import json
import sys

path, field = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as handle:
    value = json.load(handle).get(field)
if isinstance(value, bool) or not isinstance(value, int):
    raise SystemExit(f"{path}: {field!r} is not an integer")
print(value)
PY
}

run_train_module() {
  local module="$1"
  shift
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      "--nproc_per_node=${NPROC_PER_NODE}" \
      -m "${module}" "$@"
  else
    "${PYTHON_BIN}" -m "${module}" "$@"
  fi
}

merge_adapter_to_base_model() {
  local base_model="$1"
  local adapter="$2"
  local output="$3"
  local contract_manifest="$4"
  if [[ -f "${output}/config.json" && -f "${output}/export_manifest.json" ]]; then
    printf 'Reuse merged base model: %s\n' "${output}"
    return
  fi
  printf '[pipeline] stage=merge status=start base=%s output=%s\n' \
    "${base_model}" "${output}"
  run_module care_opd.export_model \
    --base-model "${base_model}" \
    --adapter "${adapter}" \
    --output "${output}" \
    --dtype bfloat16 \
    --contract-manifest "${contract_manifest}"
  printf '[pipeline] stage=merge status=complete output=%s\n' "${output}"
}

qzbc_final_model_path() {
  if [[ -f "${QZBC_RUN_ROOT}/FINAL_MODEL.txt" ]]; then
    tr -d '\r\n' <"${QZBC_RUN_ROOT}/FINAL_MODEL.txt"
  else
    printf '%s' "${QZBC_RUN_ROOT}/final_model"
  fi
}
