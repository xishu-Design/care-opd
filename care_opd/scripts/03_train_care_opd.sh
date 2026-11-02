#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

TRAIN_DATA="${QZBC_DATA_ROOT}/compiled/train"
CLEAN_SFT="${TRAIN_DATA}/qzbc_clean_mixed_sft.jsonl"
ONLINE_REQUESTS="${TRAIN_DATA}/qzbc_on_policy_requests.jsonl"
CONTRACT_MANIFEST="${TRAIN_DATA}/manifest.json"
require_file "${CLEAN_SFT}"
require_file "${ONLINE_REQUESTS}"
require_file "${CONTRACT_MANIFEST}"

mkdir -p -- "${QZBC_RUN_ROOT}"

# Common qzbc-native warm start.  It teaches only the existing qzbc
# Evidence-Packet -> GeneratedResponse boundary; no replacement Agent exists.
WARM_ROOT="${QZBC_RUN_ROOT}/warm_start"
if [[ ! -f "${WARM_ROOT}/final/run_metadata.json" ]]; then
  printf '[pipeline] stage=warm_start status=start output=%s\n' "${WARM_ROOT}"
  run_train_module care_opd.train_sft \
    --config "${CARE_CONFIG}" \
    --model "${STUDENT_MODEL}" \
    --data "${CLEAN_SFT}" \
    --output "${WARM_ROOT}" \
    --baseline clean_sft \
    --target_field target \
    --resume_from_checkpoint auto \
    --seed "${TRAIN_SEED}"
  printf '[pipeline] stage=warm_start status=complete output=%s\n' "${WARM_ROOT}"
else
  printf '[pipeline] stage=warm_start status=reuse output=%s\n' "${WARM_ROOT}"
fi

WARM_MODEL="${WARM_ROOT}/merged"
merge_adapter_to_base_model \
  "${STUDENT_MODEL}" \
  "${WARM_ROOT}/final" \
  "${WARM_MODEL}" \
  "${CONTRACT_MANIFEST}"

current_model="${WARM_MODEL}"
completed_round=0

for round in $(seq 1 "${ONLINE_ROUNDS}"); do
  round_tag="$(printf '%02d' "${round}")"
  round_root="${QZBC_RUN_ROOT}/round_${round_tag}"
  rollout_root="${round_root}/rollouts"
  train_root="${round_root}/opd_train"
  mkdir -p -- "${round_root}"

  if [[ ! -f "${rollout_root}/summary.json" ]]; then
    printf '[pipeline] stage=round_%s_rollout status=start output=%s\n' \
      "${round_tag}" "${rollout_root}"
    run_qzbc_rollout \
      --requests "${ONLINE_REQUESTS}" \
      --output-dir "${rollout_root}" \
      --model "${current_model}" \
      --dtype bfloat16 \
      --max-new-tokens "${ROLLOUT_MAX_NEW_TOKENS}" \
      --temperature "${ROLLOUT_TEMPERATURE}" \
      --top-p "${ROLLOUT_TOP_P}" \
      --seed "$((TRAIN_SEED + round - 1))" \
      --max-examples "${ROLLOUTS_PER_ROUND}"
    printf '[pipeline] stage=round_%s_rollout status=complete output=%s\n' \
      "${round_tag}" "${rollout_root}"
  else
    printf '[pipeline] stage=round_%s_rollout status=reuse output=%s\n' \
      "${round_tag}" "${rollout_root}"
  fi

  failed_rollouts="$(json_int_field "${rollout_root}/summary.json" opd_failures)"
  if [[ "${failed_rollouts}" -eq 0 ]]; then
    printf 'No verifier-failed on-policy samples remain after round %s.\n' "${round}"
    completed_round=$((round - 1))
    break
  fi
  require_file "${rollout_root}/qzbc_opd.jsonl"

  if [[ ! -f "${train_root}/final/run_metadata.json" ]]; then
    printf '[pipeline] stage=round_%s_opd status=start output=%s\n' \
      "${round_tag}" "${train_root}"
    run_train_module care_opd.train_opd \
      --config "${CARE_CONFIG}" \
      --model "${current_model}" \
      --teacher "${TEACHER_MODEL}" \
      --data "${rollout_root}/qzbc_opd.jsonl" \
      --output "${train_root}" \
      --baseline_mode care_opd \
      --resume_from_checkpoint auto \
      --seed "$((TRAIN_SEED + round - 1))"
    printf '[pipeline] stage=round_%s_opd status=complete output=%s\n' \
      "${round_tag}" "${train_root}"
  else
    printf '[pipeline] stage=round_%s_opd status=reuse output=%s\n' \
      "${round_tag}" "${train_root}"
  fi

  if [[ "${round}" -eq "${ONLINE_ROUNDS}" ]]; then
    merged_model="${QZBC_RUN_ROOT}/final_model"
  else
    merged_model="${round_root}/merged_model"
  fi
  merge_adapter_to_base_model \
    "${current_model}" \
    "${train_root}/final" \
    "${merged_model}" \
    "${CONTRACT_MANIFEST}"
  current_model="${merged_model}"
  completed_round="${round}"
done

if [[ ! -f "${QZBC_RUN_ROOT}/final_model/config.json" ]]; then
  # Early convergence: the last standalone merged checkpoint is already the
  # deployable base model.  Keep one stable path without copying its weights.
  final_target="$(realpath "${current_model}")"
  ln -sfn "${final_target}" "${QZBC_RUN_ROOT}/final_model"
fi

final_model="$(realpath "${QZBC_RUN_ROOT}/final_model")"
printf '%s\n' "${final_model}" >"${QZBC_RUN_ROOT}/FINAL_MODEL.txt"
printf '%s\n' "${completed_round}" >"${QZBC_RUN_ROOT}/COMPLETED_ONLINE_ROUNDS.txt"
printf 'Standalone qzbc replacement base model: %s\n' "${final_model}"
