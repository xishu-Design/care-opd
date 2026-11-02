#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

require_dir "${QZBC_ROOT}/backend/app/llm"
mkdir -p -- "${QZBC_DATA_ROOT}/seeds" "${QZBC_DATA_ROOT}/compiled"

SEED_FILE="${QZBC_DATA_ROOT}/seeds/qzbc_native_seeds.jsonl"
if [[ "${OVERWRITE_DATA:-0}" == "1" || ! -s "${SEED_FILE}" ]]; then
  printf '[pipeline] stage=seed_build status=start total=%s\n' "${QZBC_SEED_TOTAL}"
  run_module care_opd.qzbc_seed_builder \
    --qzbc-root "${QZBC_ROOT}" \
    --output "${SEED_FILE}" \
    --total "${QZBC_SEED_TOTAL}" \
    --seed "${TRAIN_SEED}" \
    --dev-ratio "${QZBC_DEV_RATIO}" \
    --test-ratio "${QZBC_TEST_RATIO}"
  printf '[pipeline] stage=seed_build status=complete output=%s\n' "${SEED_FILE}"
else
  printf '[pipeline] stage=seed_build status=reuse output=%s\n' "${SEED_FILE}"
fi

for split in train dev test; do
  split_seed="${QZBC_DATA_ROOT}/seeds/qzbc_native_seeds.${split}.jsonl"
  split_output="${QZBC_DATA_ROOT}/compiled/${split}"
  require_file "${split_seed}"
  printf '[pipeline] stage=data_compile split=%s status=start\n' "${split}"
  run_module care_opd.qzbc_dataset \
    --input "${split_seed}" \
    --qzbc-root "${QZBC_ROOT}" \
    --output-dir "${split_output}"
  printf '[pipeline] stage=data_compile split=%s status=complete output=%s\n' \
    "${split}" "${split_output}"
done

require_file "${QZBC_DATA_ROOT}/compiled/train/qzbc_clean_mixed_sft.jsonl"
require_file "${QZBC_DATA_ROOT}/compiled/train/qzbc_on_policy_requests.jsonl"
require_file "${QZBC_DATA_ROOT}/compiled/test/qzbc_on_policy_requests.jsonl"
printf 'qzbc-native formal data ready: %s\n' "${QZBC_DATA_ROOT}"
