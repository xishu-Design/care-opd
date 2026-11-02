#!/usr/bin/env bash
set -Eeuo pipefail

# Formal QZBC grouped rerun. Each checkpoint is evaluated sequentially; within
# a checkpoint, qzbc_rollout defaults to four independent workers on GPUs 4-7.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
QZBC_ROOT="${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}"
DATA_ROOT="${QZBC_DATA_ROOT:-${PROJECT_ROOT}/data/qzbc_formal}"
REQUESTS="${REQUESTS:-${DATA_ROOT}/compiled/test/qzbc_on_policy_requests.jsonl}"
SEED_MANIFEST="${SEED_MANIFEST:-${DATA_ROOT}/seeds/qzbc_native_seeds.manifest.json}"
COMPILED_TRAIN_MANIFEST="${COMPILED_TRAIN_MANIFEST:-${DATA_ROOT}/compiled/train/manifest.json}"
COMPILED_DEV_MANIFEST="${COMPILED_DEV_MANIFEST:-${DATA_ROOT}/compiled/dev/manifest.json}"
COMPILED_TEST_MANIFEST="${COMPILED_TEST_MANIFEST:-${DATA_ROOT}/compiled/test/manifest.json}"
PUBLIC_RUN_ROOT="${PUBLIC_RUN_ROOT:-${PROJECT_ROOT}/runs/public_eval}"
SOURCE_SUITE="${SOURCE_SUITE:-${PUBLIC_RUN_ROOT}/tcmeval_pa_v3/model_suite.json}"
PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK:-${PROJECT_ROOT}/public_eval/PREREGISTRATION_LOCK.json}"
OUT_ROOT="${OUT_ROOT:-${PUBLIC_RUN_ROOT}/qzbc_grouped_identity_v1}"
SUITE="${OUT_ROOT}/model_suite.json"
DATA_AUDIT="${OUT_ROOT}/DATA_AUDIT.json"
PREPARED_MARKER="${OUT_ROOT}/PREPARED.json"
EVALUATOR="${SCRIPT_DIR}/evaluate_qzbc_grouped.py"
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

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/public_eval${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p -- "${OUT_ROOT}"
exec 9>"${OUT_ROOT}/.qzbc_grouped_suite.lock"
if ! flock -n 9; then
  printf 'Another QZBC grouped suite owns %s\n' \
    "${OUT_ROOT}/.qzbc_grouped_suite.lock" >&2
  exit 3
fi

for required in \
  "${PYTHON_BIN}" "${QZBC_ROOT}" "${REQUESTS}" "${SEED_MANIFEST}" \
  "${COMPILED_TRAIN_MANIFEST}" "${COMPILED_DEV_MANIFEST}" \
  "${COMPILED_TEST_MANIFEST}" "${SOURCE_SUITE}" \
  "${PREREGISTRATION_LOCK}" "${EVALUATOR}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing formal QZBC grouped input: %s\n' "${required}" >&2
    exit 2
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is required for the GPU occupancy gate (%s)\n' \
    "${PHYSICAL_GPUS}" >&2
  exit 2
fi
if ! nvidia-smi -i "${PHYSICAL_GPUS}" --query-gpu=index --format=csv,noheader,nounits >/dev/null; then
  printf 'Requested physical GPUs are not all available: %s\n' \
    "${PHYSICAL_GPUS}" >&2
  exit 2
fi
busy_pids="$(
  nvidia-smi -i "${PHYSICAL_GPUS}" --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' || true
)"
if [[ -n "${busy_pids}" ]]; then
  printf 'Requested GPU set %s already has compute PID(s); refusing to start: %s\n' \
    "${PHYSICAL_GPUS}" \
    "$(printf '%s' "${busy_pids}" | tr '\n' ' ')" >&2
  exit 3
fi

data_args=(
  --qzbc-root "${QZBC_ROOT}"
  --requests "${REQUESTS}"
  --seed-manifest "${SEED_MANIFEST}"
  --compiled-train-manifest "${COMPILED_TRAIN_MANIFEST}"
  --compiled-dev-manifest "${COMPILED_DEV_MANIFEST}"
  --compiled-test-manifest "${COMPILED_TEST_MANIFEST}"
  --data-audit "${DATA_AUDIT}"
)

"${PYTHON_BIN}" -u "${EVALUATOR}" prepare \
  --source-suite "${SOURCE_SUITE}" \
  --suite "${SUITE}" \
  --preregistration-lock "${PREREGISTRATION_LOCK}" \
  --prepared-marker "${PREPARED_MARKER}" \
  "${data_args[@]}"

labels=(frozen clean_sft recovery_sft recovery_dpo gkd_full care_opd)
for label in "${labels[@]}"; do
  printf '[qzbc-grouped] model=%s status=start physical_gpus=%s workers=%s logical_devices=%s\n' \
    "${label}" "${PHYSICAL_GPUS}" "${NUM_WORKERS}" "${DEVICES}"
  "${PYTHON_BIN}" -u "${EVALUATOR}" evaluate-model \
    --suite "${SUITE}" \
    --preregistration-lock "${PREREGISTRATION_LOCK}" \
    --output-root "${OUT_ROOT}" \
    --prepared-marker "${PREPARED_MARKER}" \
    --model-label "${label}" \
    --num-workers "${NUM_WORKERS}" \
    --devices "${DEVICES}" \
    "${data_args[@]}"
  printf '[qzbc-grouped] model=%s status=complete\n' "${label}"
done

"${PYTHON_BIN}" -u "${EVALUATOR}" summarize-suite \
  --suite "${SUITE}" \
  --preregistration-lock "${PREREGISTRATION_LOCK}" \
  --output-root "${OUT_ROOT}" \
  --prepared-marker "${PREPARED_MARKER}" \
  "${data_args[@]}"

printf 'Identity-bound QZBC grouped suite complete: %s\n' \
  "${OUT_ROOT}/formal/suite_summary.json"
