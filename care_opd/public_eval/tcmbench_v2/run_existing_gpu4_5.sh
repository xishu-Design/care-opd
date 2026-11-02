#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
TCMBENCH_REPO="${TCMBENCH_REPO:-/opt/care-workspace/TCMBench}"
DATA_ROOT="${DATA_ROOT:-${TCMBENCH_REPO}/data}"
OUT_ROOT="${OUT_ROOT:-/opt/care-workspace/care_opd/runs/public_eval/tcmbench_v2}"
PROTOCOL="${PROTOCOL:-answer_only}"
TRACK="${TRACK:-robustness}"
BASELINE_ROOT="${BASELINE_ROOT:-/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu}"
CARE_ROOT="${CARE_ROOT:-/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu}"
FROZEN_MODEL="${FROZEN_MODEL:-/opt/care-workspace/models/Qwen3.5-4B}"
EVALUATOR="${SCRIPT_DIR}/evaluate_tcmbench_v2.py"
OVERLAP_EXCLUSIONS="${SCRIPT_DIR}/overlap_exclusions.json"
PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK:-${PROJECT_ROOT}/public_eval/PREREGISTRATION_LOCK.json}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICES="${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

IFS=',' read -r -a physical_gpu_list <<<"${PHYSICAL_GPUS}"
if [[ "${NUM_WORKERS}" != 4 || "${DEVICES}" != "cuda:0,cuda:1,cuda:2,cuda:3" ]]; then
  printf 'Formal TCMBench layout requires NUM_WORKERS=4 and DEVICES=cuda:0,cuda:1,cuda:2,cuda:3\n' >&2
  exit 2
fi
if [[ "${#physical_gpu_list[@]}" -ne 4 ]]; then
  printf 'PHYSICAL_GPUS must contain exactly four comma-separated GPU IDs: %s\n' "${PHYSICAL_GPUS}" >&2
  exit 2
fi

case "${PROTOCOL}" in
  answer_only) MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}" ;;
  reconstructed_official_cot) MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}" ;;
  *) printf 'Unsupported PROTOCOL=%s\n' "${PROTOCOL}" >&2; exit 2 ;;
esac
case "${TRACK}" in
  primary_sdt|robustness|all13) ;;
  *) printf 'Unsupported TRACK=%s\n' "${TRACK}" >&2; exit 2 ;;
esac

mkdir -p -- "${OUT_ROOT}/logs" "${OUT_ROOT}/inspection"
exec 9>"${OUT_ROOT}/.suite.lock"
if ! flock -n 9; then
  printf 'Another TCMBench V2 suite owns %s/.suite.lock\n' "${OUT_ROOT}" >&2
  exit 3
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python environment is missing: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -s "${PREREGISTRATION_LOCK}" ]]; then
  printf 'Shared preregistration lock is missing: %s\n' "${PREREGISTRATION_LOCK}" >&2
  printf 'Create it before any formal predictions with public_eval/create_preregistration_lock.py\n' >&2
  exit 2
fi

model_ready() {
  local path="$1" require_export="${2:-yes}"
  [[ -d "${path}" && -f "${path}/config.json" && -f "${path}/tokenizer_config.json" ]] || return 1
  if [[ "${require_export}" == yes && ! -f "${path}/export_manifest.json" ]]; then return 1; fi
  [[ -f "${path}/model.safetensors" \
      || -f "${path}/model.safetensors.index.json" \
      || -f "${path}/pytorch_model.bin" \
      || -f "${path}/pytorch_model.bin.index.json" ]]
}

canonical_ready() {
  local path="$1" require_export="${2:-yes}"
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
    printf 'CARE unexpectedly aliases %s: %s\n' "${labels[$baseline_index]}" "${models[5]}" >&2
    exit 2
  fi
done

check_gpu_idle() {
  local processes
  command -v nvidia-smi >/dev/null 2>&1 || {
    printf 'nvidia-smi unavailable\n' >&2; return 1;
  }
  processes="$(nvidia-smi -i "${PHYSICAL_GPUS}" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)" || {
    printf 'Cannot query physical GPUs %s\n' "${PHYSICAL_GPUS}" >&2; return 1;
  }
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    printf 'At least one physical GPU in %s is busy:\n%s\n' "${PHYSICAL_GPUS}" "${processes}" >&2
    return 1
  fi
}

"${PYTHON_BIN}" "${EVALUATOR}" inspect-data \
  --data-root "${DATA_ROOT}" \
  --track "${TRACK}" \
  --overlap-exclusions "${OVERLAP_EXCLUSIONS}" \
  --output "${OUT_ROOT}/inspection/data_inspection.json" \
  --pair-manifest-output "${OUT_ROOT}/inspection/sdt_pair_manifest.json" \
  2>&1 | tee -a "${OUT_ROOT}/logs/inspect_data.log"

suite_args=()
for index in "${!labels[@]}"; do
  suite_args+=(--model "${labels[$index]}=${models[$index]}")
done
"${PYTHON_BIN}" "${EVALUATOR}" freeze-suite \
  --suite "${OUT_ROOT}/model_suite.json" "${suite_args[@]}" \
  2>&1 | tee -a "${OUT_ROOT}/logs/freeze_suite.log"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for label in "${labels[@]}"; do
  check_gpu_idle
  log="${OUT_ROOT}/logs/${PROTOCOL}_${TRACK}_${label}.log"
  printf '[suite] protocol=%s track=%s model=%s status=start log=%s\n' \
    "${PROTOCOL}" "${TRACK}" "${label}" "${log}" | tee -a "${OUT_ROOT}/logs/suite.log"
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}" "${PYTHON_BIN}" -u "${EVALUATOR}" evaluate \
    --data-root "${DATA_ROOT}" \
    --suite "${OUT_ROOT}/model_suite.json" \
    --model-label "${label}" \
    --output-root "${OUT_ROOT}" \
    --protocol "${PROTOCOL}" \
    --track "${TRACK}" \
    --overlap-exclusions "${OVERLAP_EXCLUSIONS}" \
    --preregistration-lock "${PREREGISTRATION_LOCK}" \
    --dtype bfloat16 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-workers "${NUM_WORKERS}" \
    --devices "${DEVICES}" \
    2>&1 | tee -a "${log}"
  printf '[suite] protocol=%s track=%s model=%s status=complete\n' \
    "${PROTOCOL}" "${TRACK}" "${label}" | tee -a "${OUT_ROOT}/logs/suite.log"
done

"${PYTHON_BIN}" "${EVALUATOR}" summarize-suite \
  --data-root "${DATA_ROOT}" \
  --suite "${OUT_ROOT}/model_suite.json" \
  --output-root "${OUT_ROOT}" \
  --protocol "${PROTOCOL}" \
  --track "${TRACK}" \
  --overlap-exclusions "${OVERLAP_EXCLUSIONS}" \
  --preregistration-lock "${PREREGISTRATION_LOCK}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  2>&1 | tee -a "${OUT_ROOT}/logs/summarize_${PROTOCOL}_${TRACK}.log"

printf 'TCMBench V2 suite complete: %s/%s\n' "${OUT_ROOT}" "${PROTOCOL}"
