#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
OUT_ROOT="${OUT_ROOT:-/opt/care-workspace/care_opd/runs/public_eval/cmb_exam_v1}"
BUNDLED_DATA="${SCRIPT_DIR}/data"
DATA_DIR="${OUT_ROOT}/data"
BASELINE_ROOT="${BASELINE_ROOT:-/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu}"
CARE_ROOT="${CARE_ROOT:-/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu}"
FROZEN_MODEL="${FROZEN_MODEL:-/opt/care-workspace/models/Qwen3.5-4B}"
EVALUATOR="${SCRIPT_DIR}/evaluate_cmb_exam.py"
PREPARE="${SCRIPT_DIR}/prepare_data.py"
PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK:-${SCRIPT_DIR}/../PREREGISTRATION_LOCK.json}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICES="${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

IFS=',' read -r -a physical_gpu_list <<<"${PHYSICAL_GPUS}"
if [[ "${NUM_WORKERS}" != 4 || "${DEVICES}" != "cuda:0,cuda:1,cuda:2,cuda:3" ]]; then
  printf 'Formal CMB-Exam layout requires NUM_WORKERS=4 and DEVICES=cuda:0,cuda:1,cuda:2,cuda:3\n' >&2
  exit 2
fi
if [[ "${#physical_gpu_list[@]}" -ne 4 ]]; then
  printf 'PHYSICAL_GPUS must contain exactly four comma-separated GPU IDs: %s\n' "${PHYSICAL_GPUS}" >&2
  exit 2
fi

mkdir -p -- "${OUT_ROOT}/logs" "${DATA_DIR}"
exec 9>"${OUT_ROOT}/.suite.lock"
if ! flock -n 9; then
  printf 'Another CMB-Exam suite owns %s/.suite.lock\n' "${OUT_ROOT}" >&2
  exit 3
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python environment is missing or not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -s "${PREREGISTRATION_LOCK}" ]]; then
  printf 'Formal preregistration lock is missing or empty: %s\n' "${PREREGISTRATION_LOCK}" >&2
  exit 2
fi

model_ready() {
  local path="$1"
  local require_export="${2:-yes}"
  [[ -d "${path}" && -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] || return 1
  if [[ "${require_export}" == yes && ! -f "${path}/export_manifest.json" ]]; then
    return 1
  fi
  [[ -f "${path}/model.safetensors" \
      || -f "${path}/model.safetensors.index.json" \
      || -f "${path}/pytorch_model.bin" \
      || -f "${path}/pytorch_model.bin.index.json" ]]
}

canonical_ready() {
  local path="$1"
  local require_export="${2:-yes}"
  path="$(readlink -f -- "${path}")"
  model_ready "${path}" "${require_export}" || {
    printf 'Model is not READY: %s\n' "${path}" >&2
    return 1
  }
  printf '%s' "${path}"
}

resolve_care_model() {
  local pointer candidate
  pointer="${CARE_ROOT}/FINAL_MODEL.txt"
  if [[ -s "${pointer}" ]]; then
    candidate="$(tr -d '\r\n' <"${pointer}")"
    if candidate="$(canonical_ready "${candidate}" yes 2>/dev/null)"; then
      printf '%s' "${candidate}"
      return 0
    fi
  fi
  canonical_ready "${CARE_ROOT}/final_model" yes
}

labels=(frozen clean_sft recovery_sft recovery_dpo gkd_full care_opd)
models=(
  "$(canonical_ready "${FROZEN_MODEL}" no)"
  "$(canonical_ready "${BASELINE_ROOT}/warm_start/merged" yes)"
  "$(canonical_ready "${BASELINE_ROOT}/baselines/recovery_sft/merged" yes)"
  "$(canonical_ready "${BASELINE_ROOT}/baselines/recovery_dpo/merged" yes)"
  "$(canonical_ready "${BASELINE_ROOT}/baselines/gkd_full/final_model" yes)"
  "$(resolve_care_model)"
)
for baseline_index in 0 1 2 3 4; do
  if [[ "${models[$baseline_index]}" == "${models[5]}" ]]; then
    printf 'CARE model unexpectedly aliases baseline %s: %s\n' \
      "${labels[$baseline_index]}" "${models[5]}" >&2
    exit 2
  fi
done

check_gpu_idle() {
  local processes
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is unavailable; cannot validate physical GPUs %s\n' "${PHYSICAL_GPUS}" >&2
    return 1
  fi
  if ! processes="$(nvidia-smi -i "${PHYSICAL_GPUS}" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)"; then
    printf 'Failed to query physical GPUs %s with nvidia-smi\n' "${PHYSICAL_GPUS}" >&2
    return 1
  fi
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    printf 'At least one physical GPU in %s is busy; refusing to race another job:\n%s\n' "${PHYSICAL_GPUS}" "${processes}" >&2
    return 1
  fi
}

"${PYTHON_BIN}" "${PREPARE}" \
  --output-dir "${DATA_DIR}" \
  --question-source "${BUNDLED_DATA}/CMB-test-choice-question-merge.json" \
  --answer-source "${BUNDLED_DATA}/CMB-test-choice-answer.json" \
  --overlap-report "${BUNDLED_DATA}/qzbc_overlap_report.json" \
  | tee -a "${OUT_ROOT}/logs/prepare_data.log"
"${PYTHON_BIN}" "${EVALUATOR}" inspect-data --data-dir "${DATA_DIR}" \
  | tee -a "${OUT_ROOT}/logs/inspect_data.log"

suite_args=()
for index in "${!labels[@]}"; do
  suite_args+=(--model "${labels[$index]}=${models[$index]}")
done
"${PYTHON_BIN}" "${EVALUATOR}" freeze-suite \
  --suite "${OUT_ROOT}/model_suite.json" "${suite_args[@]}" \
  | tee -a "${OUT_ROOT}/logs/freeze_suite.log"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for label in "${labels[@]}"; do
  check_gpu_idle
  log="${OUT_ROOT}/logs/official_${label}.log"
  printf '[suite] track=official+controlled_robustness_v1 model=%s status=start log=%s\n' \
    "${label}" "${log}" | tee -a "${OUT_ROOT}/logs/suite.log"
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}" "${PYTHON_BIN}" -u "${EVALUATOR}" evaluate \
    --data-dir "${DATA_DIR}" \
    --suite "${OUT_ROOT}/model_suite.json" \
    --model-label "${label}" \
    --output-root "${OUT_ROOT}" \
    --dtype bfloat16 \
    --max-new-tokens 8 \
    --batch-size 8 \
    --num-workers "${NUM_WORKERS}" \
    --devices "${DEVICES}" \
    --preregistration-lock "${PREREGISTRATION_LOCK}" \
    2>&1 | tee -a "${log}"
  printf '[suite] model=%s status=complete\n' \
    "${label}" | tee -a "${OUT_ROOT}/logs/suite.log"
done

"${PYTHON_BIN}" "${EVALUATOR}" summarize-suite \
  --data-dir "${DATA_DIR}" \
  --suite "${OUT_ROOT}/model_suite.json" \
  --output-root "${OUT_ROOT}" \
  --preregistration-lock "${PREREGISTRATION_LOCK}" \
  | tee -a "${OUT_ROOT}/logs/summarize_suite.log"

printf 'CMB-Exam suite complete: %s\n' "${OUT_ROOT}"
