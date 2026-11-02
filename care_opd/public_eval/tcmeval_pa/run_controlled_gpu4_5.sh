#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
OUT_ROOT="${OUT_ROOT:-/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_controlled_v1}"
OFFICIAL_SUITE_ROOT="${OFFICIAL_SUITE_ROOT:-/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3}"
SOURCE_SUITE="${SOURCE_SUITE:-${OFFICIAL_SUITE_ROOT}/model_suite.json}"
OFFICIAL_RESULTS_ROOT="${OFFICIAL_RESULTS_ROOT:-${OFFICIAL_SUITE_ROOT}/official}"
PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK:-${SCRIPT_DIR}/../PREREGISTRATION_LOCK.json}"
EVALUATOR="${SCRIPT_DIR}/evaluate_controlled_robustness.py"
PREPARE="${SCRIPT_DIR}/prepare_data.py"
LOCKER="${SCRIPT_DIR}/../create_preregistration_lock.py"
BUNDLED_DATA="${SCRIPT_DIR}/data/TCMEval-PA.xlsx"
ORIGIN_REFERENCE="${ORIGIN_REFERENCE:-}"
BERT_MODEL="${BERT_MODEL:-/opt/care-workspace/models/bert-base-chinese}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICES="${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

IFS=',' read -r -a physical_gpu_array <<< "${PHYSICAL_GPUS}"
IFS=',' read -r -a logical_device_array <<< "${DEVICES}"
if ! [[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]] \
  || (( ${#physical_gpu_array[@]} != NUM_WORKERS )) \
  || (( ${#logical_device_array[@]} != NUM_WORKERS )); then
  printf 'Invalid GPU topology: PHYSICAL_GPUS=%s NUM_WORKERS=%s DEVICES=%s\n' \
    "${PHYSICAL_GPUS}" "${NUM_WORKERS}" "${DEVICES}" >&2
  exit 2
fi
for gpu in "${physical_gpu_array[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    printf 'PHYSICAL_GPUS must contain numeric GPU indices: %s\n' \
      "${PHYSICAL_GPUS}" >&2
    exit 2
  fi
done

COPY_BUNDLED_DATA=no
if [[ -n "${DATA_FILE:-}" ]]; then
  DATA_FILE="${DATA_FILE}"
elif [[ -f "${BUNDLED_DATA}" ]]; then
  DATA_FILE="${OUT_ROOT}/data/TCMEval-PA.xlsx"
  COPY_BUNDLED_DATA=yes
else
  DATA_FILE="${OUT_ROOT}/data/TCMEval-PA.xlsx"
fi

mkdir -p -- "${OUT_ROOT}/logs" "${OUT_ROOT}/data"
exec 9>"${OUT_ROOT}/.suite.lock"
if ! flock -n 9; then
  printf 'Another controlled TCMEval-PA suite owns %s/.suite.lock\n' "${OUT_ROOT}" >&2
  exit 3
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python environment is missing or not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${PREREGISTRATION_LOCK}" ]]; then
  printf 'Missing preregistration lock (fail closed): %s\n' "${PREREGISTRATION_LOCK}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_SUITE}" ]]; then
  printf 'Missing completed official model suite: %s\n' "${SOURCE_SUITE}" >&2
  exit 2
fi
if [[ ! -d "${OFFICIAL_RESULTS_ROOT}" ]]; then
  printf 'Missing read-only official result root: %s\n' "${OFFICIAL_RESULTS_ROOT}" >&2
  exit 2
fi

# Recompute the registered contract and validate the existing immutable lock.
# The lock creator refuses to overwrite a differing lock.
locker_args=(
  --project-root "${PROJECT_ROOT}"
  --model-suite "${SOURCE_SUITE}"
  --bert-model "${BERT_MODEL}"
  --output "${PREREGISTRATION_LOCK}"
)
if [[ -n "${ORIGIN_REFERENCE}" ]]; then
  locker_args+=(--origin-reference "${ORIGIN_REFERENCE}")
fi
"${PYTHON_BIN}" "${LOCKER}" "${locker_args[@]}" \
  | tee -a "${OUT_ROOT}/logs/validate_preregistration_lock.log"

if [[ "${COPY_BUNDLED_DATA}" == yes && ! -f "${DATA_FILE}" ]]; then
  cp -- "${BUNDLED_DATA}" "${DATA_FILE}"
fi
if [[ -f "${DATA_FILE}" ]]; then
  "${PYTHON_BIN}" "${PREPARE}" --output "${DATA_FILE}" --verify-only \
    | tee -a "${OUT_ROOT}/logs/prepare_data.log"
else
  "${PYTHON_BIN}" "${PREPARE}" --output "${DATA_FILE}" \
    | tee -a "${OUT_ROOT}/logs/prepare_data.log"
fi

# Copy and validate the existing six-model suite without touching its source.
"${PYTHON_BIN}" "${EVALUATOR}" reuse-suite \
  --source "${SOURCE_SUITE}" \
  --destination "${OUT_ROOT}/model_suite.json" \
  | tee -a "${OUT_ROOT}/logs/reuse_suite.log"

# This command must finish before the first model process starts.  It freezes
# all 328 non-identity maps and their manifest hash under the preregistered lock.
"${PYTHON_BIN}" "${EVALUATOR}" freeze-mapping \
  --data "${DATA_FILE}" \
  --output-root "${OUT_ROOT}" \
  --preregistration-lock "${PREREGISTRATION_LOCK}" \
  | tee -a "${OUT_ROOT}/logs/freeze_mapping.log"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

check_gpu_idle() {
  local processes
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is unavailable; cannot validate physical GPUs %s\n' \
      "${PHYSICAL_GPUS}" >&2
    return 1
  fi
  if ! nvidia-smi -i "${PHYSICAL_GPUS}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
    printf 'Requested physical GPUs are not all available: %s\n' \
      "${PHYSICAL_GPUS}" >&2
    return 1
  fi
  if ! processes="$(nvidia-smi -i "${PHYSICAL_GPUS}" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)"; then
    printf 'Failed to query physical GPUs %s with nvidia-smi\n' \
      "${PHYSICAL_GPUS}" >&2
    return 1
  fi
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    printf 'A requested physical GPU (%s) is busy; refusing to race another job:\n%s\n' \
      "${PHYSICAL_GPUS}" "${processes}" >&2
    return 1
  fi
}

labels=(frozen clean_sft recovery_sft recovery_dpo gkd_full care_opd)
for label in "${labels[@]}"; do
  check_gpu_idle
  log="${OUT_ROOT}/logs/controlled_${label}.log"
  printf '[suite] protocol=controlled model=%s status=start log=%s\n' \
    "${label}" "${log}" | tee -a "${OUT_ROOT}/logs/suite.log"
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}" "${PYTHON_BIN}" -u "${EVALUATOR}" evaluate \
    --data "${DATA_FILE}" \
    --suite "${OUT_ROOT}/model_suite.json" \
    --model-label "${label}" \
    --output-root "${OUT_ROOT}" \
    --preregistration-lock "${PREREGISTRATION_LOCK}" \
    --dtype bfloat16 \
    --max-new-tokens 8 \
    --num-workers "${NUM_WORKERS}" \
    --devices "${DEVICES}" \
    2>&1 | tee -a "${log}"
  printf '[suite] protocol=controlled model=%s status=complete\n' \
    "${label}" | tee -a "${OUT_ROOT}/logs/suite.log"
done

"${PYTHON_BIN}" "${EVALUATOR}" summarize-suite \
  --data "${DATA_FILE}" \
  --suite "${OUT_ROOT}/model_suite.json" \
  --output-root "${OUT_ROOT}" \
  --official-root "${OFFICIAL_RESULTS_ROOT}" \
  --preregistration-lock "${PREREGISTRATION_LOCK}" \
  | tee -a "${OUT_ROOT}/logs/summarize_suite.log"

printf 'Controlled TCMEval-PA suite complete: %s\n' "${OUT_ROOT}"
