#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QZBC_RUN_ROOT="${QZBC_RUN_ROOT:-${PROJECT_ROOT}/runs/qzbc_base_model}"
CARE_NOHUP_LOG="${CARE_NOHUP_LOG:-${QZBC_RUN_ROOT}/train.nohup.log}"
CARE_PID_FILE="${CARE_PID_FILE:-${QZBC_RUN_ROOT}/train.pid}"
export QZBC_RUN_ROOT

mkdir -p -- "${QZBC_RUN_ROOT}"
if [[ -f "${CARE_PID_FILE}" ]]; then
  existing_pid="$(<"${CARE_PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    printf 'qzbc base-model training is already running (pid=%s).\n' \
      "${existing_pid}"
    exit 0
  fi
fi

nohup bash "${SCRIPT_DIR}/run_all_formal.sh" "$@" \
  >>"${CARE_NOHUP_LOG}" 2>&1 </dev/null &
pipeline_pid=$!
printf '%s\n' "${pipeline_pid}" >"${CARE_PID_FILE}"
printf 'Started qzbc base-model post-training (pid=%s).\n' "${pipeline_pid}"
printf 'Log: %s\n' "${CARE_NOHUP_LOG}"
printf 'Watch: tail -f %q\n' "${CARE_NOHUP_LOG}"
