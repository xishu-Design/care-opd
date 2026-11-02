#!/usr/bin/env bash
set -Eeuo pipefail

# Offline, deterministic selector. It performs no generation or model loading.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
MECHANISM_ROOT="${MECHANISM_ROOT:-${PROJECT_ROOT}/runs/mechanism_eval/qzbc_care_opd_v1}"
FORMAL_ROOT="${FORMAL_ROOT:-${PROJECT_ROOT}/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal}"
OUTPUT_DIR="${OUTPUT_DIR:-${MECHANISM_ROOT}/qualitative_cases}"
SELECTION_SEED="${SELECTION_SEED:-20260813}"
MAX_PREVIEW_CHARACTERS="${MAX_PREVIEW_CHARACTERS:-1200}"

CLEAN_ROLLOUTS="${CLEAN_ROLLOUTS:-${FORMAL_ROOT}/family_clean_views/clean_sft.jsonl}"
GKD_ROLLOUTS="${GKD_ROLLOUTS:-${FORMAL_ROOT}/family_clean_views/gkd_full.jsonl}"
CARE_ROLLOUTS="${CARE_ROLLOUTS:-${FORMAL_ROOT}/family_clean_views/care_opd.jsonl}"
FAILURE_BANK="${FAILURE_BANK:-${MECHANISM_ROOT}/heldout_failure_bank.jsonl}"
MARGIN_JSONL="${MARGIN_JSONL:-${MECHANISM_ROOT}/recovery_margin/recovery_margin.jsonl}"

for required in \
  "${PYTHON_BIN}" "${CLEAN_ROLLOUTS}" "${GKD_ROLLOUTS}" \
  "${CARE_ROLLOUTS}" "${FAILURE_BANK}" "${MARGIN_JSONL}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing qualitative-case input: %s\n' "${required}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/select_qualitative_cases.py" \
  --clean-rollouts "${CLEAN_ROLLOUTS}" \
  --gkd-rollouts "${GKD_ROLLOUTS}" \
  --care-rollouts "${CARE_ROLLOUTS}" \
  --failure-bank "${FAILURE_BANK}" \
  --margin-jsonl "${MARGIN_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --selection-seed "${SELECTION_SEED}" \
  --max-preview-characters "${MAX_PREVIEW_CHARACTERS}" \
  --overwrite

printf 'Qualitative case panel complete: %s\n' "${OUTPUT_DIR}"
