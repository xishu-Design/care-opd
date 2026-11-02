#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/runs/mechanism_experiments/evidence_counterfactual_v1}"
mkdir -p -- "${OUT_ROOT}/logs"
LOG="${OUT_ROOT}/logs/driver_$(date +%Y%m%d_%H%M%S)_gpu${PHYSICAL_GPUS:-4,5,6}.log"
nohup env \
  PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}" \
  QZBC_ROOT="${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}" \
  SOURCE_REQUESTS="${SOURCE_REQUESTS:-${PROJECT_ROOT}/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl}" \
  SOURCE_SUITE="${SOURCE_SUITE:-${PROJECT_ROOT}/runs/public_eval/qzbc_grouped_identity_4gpu_v1/model_suite.json}" \
  OUT_ROOT="${OUT_ROOT}" \
  PHYSICAL_GPUS="${PHYSICAL_GPUS:-4,5,6}" \
  GROUPS_PER_FAMILY="${GROUPS_PER_FAMILY:-50}" \
  SELECTION_SEED="${SELECTION_SEED:-20260813}" \
  GENERATION_SEED="${GENERATION_SEED:-20260814}" \
  BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-10000}" \
  BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260813}" \
  bash "${SCRIPT_DIR}/run_gpu4_5_6.sh" \
  >"${LOG}" 2>&1 </dev/null &
PID=$!
printf '%s\n' "${PID}" >"${OUT_ROOT}/driver.pid"
printf 'pid=%s\nlog=%s\n' "${PID}" "${LOG}"
