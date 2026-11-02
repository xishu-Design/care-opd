#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
OUT_ROOT="${OUT_ROOT:-/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3}"
BUNDLED_DATA="${SCRIPT_DIR}/data/TCMEval-PA.xlsx"
COPY_BUNDLED_DATA=no
if [[ -n "${DATA_FILE:-}" ]]; then
  DATA_FILE="${DATA_FILE}"
elif [[ -f "${BUNDLED_DATA}" ]]; then
  DATA_FILE="${OUT_ROOT}/data/TCMEval-PA.xlsx"
  COPY_BUNDLED_DATA=yes
else
  DATA_FILE="${OUT_ROOT}/data/TCMEval-PA.xlsx"
fi
PROMPT_MODES="${PROMPT_MODES:-official}"
BASELINE_ROOT="${BASELINE_ROOT:-/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu}"
CARE_ROOT="${CARE_ROOT:-/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu}"
FROZEN_MODEL="${FROZEN_MODEL:-/opt/care-workspace/models/Qwen3.5-4B}"
EVALUATOR="${SCRIPT_DIR}/evaluate_tcmeval_pa.py"
PREPARE="${SCRIPT_DIR}/prepare_data.py"

mkdir -p -- "${OUT_ROOT}/logs" "${OUT_ROOT}/data"
if [[ "${COPY_BUNDLED_DATA}" == yes && ! -f "${DATA_FILE}" ]]; then
  cp -- "${BUNDLED_DATA}" "${DATA_FILE}"
fi
exec 9>"${OUT_ROOT}/.suite.lock"
if ! flock -n 9; then
  printf 'Another TCMEval-PA suite owns %s/.suite.lock\n' "${OUT_ROOT}" >&2
  exit 3
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python environment is missing or not executable: %s\n' "${PYTHON_BIN}" >&2
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
  for pointer in "${CARE_ROOT}/FINAL_MODEL.txt"; do
    if [[ -s "${pointer}" ]]; then
      candidate="$(tr -d '\r\n' <"${pointer}")"
      if candidate="$(canonical_ready "${candidate}" yes 2>/dev/null)"; then
        printf '%s' "${candidate}"
        return 0
      fi
    fi
  done
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
    printf 'nvidia-smi is unavailable; cannot validate physical GPU 4/5\n' >&2
    return 1
  fi
  if ! processes="$(nvidia-smi -i 4,5 --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)"; then
    printf 'Failed to query physical GPU 4/5 with nvidia-smi\n' >&2
    return 1
  fi
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    printf 'Physical GPU 4 or 5 is busy; refusing to race another job:\n%s\n' "${processes}" >&2
    return 1
  fi
}

if [[ -f "${DATA_FILE}" ]]; then
  "${PYTHON_BIN}" "${PREPARE}" --output "${DATA_FILE}" --verify-only \
    | tee -a "${OUT_ROOT}/logs/prepare_data.log"
else
  "${PYTHON_BIN}" "${PREPARE}" --output "${DATA_FILE}" \
    | tee -a "${OUT_ROOT}/logs/prepare_data.log"
fi
"${PYTHON_BIN}" "${EVALUATOR}" inspect-data --data "${DATA_FILE}" \
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

for prompt_mode in ${PROMPT_MODES}; do
  case "${prompt_mode}" in
    official|label_blind) ;;
    *) printf 'Unsupported PROMPT_MODES entry: %s\n' "${prompt_mode}" >&2; exit 2 ;;
  esac
  for label in "${labels[@]}"; do
    check_gpu_idle
    log="${OUT_ROOT}/logs/${prompt_mode}_${label}.log"
    printf '[suite] mode=%s model=%s status=start log=%s\n' \
      "${prompt_mode}" "${label}" "${log}" | tee -a "${OUT_ROOT}/logs/suite.log"
    CUDA_VISIBLE_DEVICES=4,5 "${PYTHON_BIN}" -u "${EVALUATOR}" evaluate \
      --data "${DATA_FILE}" \
      --suite "${OUT_ROOT}/model_suite.json" \
      --model-label "${label}" \
      --output-root "${OUT_ROOT}" \
      --prompt-mode "${prompt_mode}" \
      --dtype bfloat16 \
      --max-new-tokens 8 \
      --num-workers 2 \
      --devices cuda:0,cuda:1 \
      2>&1 | tee -a "${log}"
    printf '[suite] mode=%s model=%s status=complete\n' \
      "${prompt_mode}" "${label}" | tee -a "${OUT_ROOT}/logs/suite.log"
  done
  "${PYTHON_BIN}" "${EVALUATOR}" summarize-suite \
    --data "${DATA_FILE}" \
    --suite "${OUT_ROOT}/model_suite.json" \
    --output-root "${OUT_ROOT}" \
    --prompt-mode "${prompt_mode}" \
    | tee -a "${OUT_ROOT}/logs/summarize_${prompt_mode}.log"
done

printf 'TCMEval-PA suite complete: %s\n' "${OUT_ROOT}"
