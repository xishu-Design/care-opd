#!/usr/bin/env bash
set -Eeuo pipefail

# Teacher-forced parameter-preference scoring.  It loads one checkpoint at a
# time, performs no generation and no optimization, and can resume by model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/runs/mechanism_eval/qzbc_care_opd_v1}"
FAILURE_BANK="${FAILURE_BANK:-${OUT_ROOT}/heldout_failure_bank.jsonl}"
BASELINE_ROOT="${BASELINE_ROOT:-${PROJECT_ROOT}/runs/qzbc_baselines_3gpu}"
CARE_RUN_ROOT="${CARE_RUN_ROOT:-${PROJECT_ROOT}/runs/qzbc_base_model_3gpu}"
CLEAN_SFT_MODEL="${CLEAN_SFT_MODEL:-${BASELINE_ROOT}/warm_start/merged}"
GKD_MODEL="${GKD_MODEL:-${BASELINE_ROOT}/baselines/gkd_full/final_model}"
CARE_MODEL="${CARE_MODEL:-${CARE_RUN_ROOT}/final_model}"
SOURCE_SUITE="${SOURCE_SUITE:-${PROJECT_ROOT}/runs/public_eval/qzbc_grouped_identity_4gpu_v1/model_suite.json}"
PHYSICAL_GPU="${PHYSICAL_GPU:-4}"
DEVICE="${DEVICE:-cuda:0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
SELECTION_SEED="${SELECTION_SEED:-20260813}"

for required in \
  "${PYTHON_BIN}" "${FAILURE_BANK}" "${CLEAN_SFT_MODEL}" \
  "${GKD_MODEL}" "${CARE_MODEL}" "${SOURCE_SUITE}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing recovery-margin input: %s\n' "${required}" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p -- "${OUT_ROOT}/recovery_margin"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/recovery_margin.py" \
  --failure-bank "${FAILURE_BANK}" \
  --model "clean_sft=${CLEAN_SFT_MODEL}" \
  --model "gkd_full=${GKD_MODEL}" \
  --model "care_opd=${CARE_MODEL}" \
  --model-suite "${SOURCE_SUITE}" \
  --output-dir "${OUT_ROOT}/recovery_margin" \
  --device "${DEVICE}" \
  --dtype bfloat16 \
  --max-length 4096 \
  --response-chunk-size 768 \
  --max-examples "${MAX_EXAMPLES}" \
  --selection-seed "${SELECTION_SEED}"

printf 'Recovery-margin scoring complete: %s\n' \
  "${OUT_ROOT}/recovery_margin"
