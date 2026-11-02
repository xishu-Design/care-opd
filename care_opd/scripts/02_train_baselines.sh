#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

TRAIN_DATA="${QZBC_DATA_ROOT}/compiled/train"
CLEAN_SFT="${TRAIN_DATA}/qzbc_clean_mixed_sft.jsonl"
RECOVERY_SFT="${TRAIN_DATA}/qzbc_recovery_sft.jsonl"
ONLINE_REQUESTS="${TRAIN_DATA}/qzbc_on_policy_requests.jsonl"
CONTRACT_MANIFEST="${TRAIN_DATA}/manifest.json"
require_file "${CLEAN_SFT}"
require_file "${RECOVERY_SFT}"
require_file "${ONLINE_REQUESTS}"

mkdir -p -- "${QZBC_RUN_ROOT}/baselines"

# Every trainable baseline starts from this exact same qzbc-native clean SFT.
WARM_ROOT="${QZBC_RUN_ROOT}/warm_start"
if [[ ! -f "${WARM_ROOT}/final/run_metadata.json" ]]; then
  run_train_module care_opd.train_sft \
    --config "${CARE_CONFIG}" \
    --model "${STUDENT_MODEL}" \
    --data "${CLEAN_SFT}" \
    --output "${WARM_ROOT}" \
    --baseline clean_sft \
    --target_field target \
    --resume_from_checkpoint auto \
    --seed "${TRAIN_SEED}"
fi
WARM_MODEL="${WARM_ROOT}/merged"
merge_adapter_to_base_model \
  "${STUDENT_MODEL}" \
  "${WARM_ROOT}/final" \
  "${WARM_MODEL}" \
  "${CONTRACT_MANIFEST}"

# One common on-policy collection supplies matched DPO, GKD-full and
# trust-region OPD baselines.  CARE-OPD itself recollects after every round.
SOURCE_ROLLOUTS="${QZBC_RUN_ROOT}/baselines/common_rollouts"
if [[ ! -f "${SOURCE_ROLLOUTS}/summary.json" ]]; then
  run_qzbc_rollout \
    --requests "${ONLINE_REQUESTS}" \
    --output-dir "${SOURCE_ROLLOUTS}" \
    --model "${WARM_MODEL}" \
    --dtype bfloat16 \
    --max-new-tokens "${ROLLOUT_MAX_NEW_TOKENS}" \
    --temperature "${ROLLOUT_TEMPERATURE}" \
    --top-p "${ROLLOUT_TOP_P}" \
    --seed "${TRAIN_SEED}" \
    --max-examples "${ROLLOUTS_PER_ROUND}"
fi

source_failures="$(json_int_field "${SOURCE_ROLLOUTS}/summary.json" opd_failures)"
if [[ "${source_failures}" -eq 0 ]]; then
  printf 'No verifier-failed samples: online baselines have no training rows.\n' >&2
  exit 3
fi
require_file "${SOURCE_ROLLOUTS}/qzbc_opd.jsonl"
require_file "${SOURCE_ROLLOUTS}/qzbc_dpo.jsonl"

# Offline recovery-SFT: sees the same controlled failures, but has no
# on-policy token distributions and no causally localized loss.
RECOVERY_ROOT="${QZBC_RUN_ROOT}/baselines/recovery_sft"
if [[ ! -f "${RECOVERY_ROOT}/final/run_metadata.json" ]]; then
  run_train_module care_opd.train_sft \
    --config "${CARE_CONFIG}" \
    --model "${WARM_MODEL}" \
    --data "${RECOVERY_SFT}" \
    --output "${RECOVERY_ROOT}" \
    --baseline direct_revision_sft \
    --target_field target \
    --resume_from_checkpoint auto \
    --seed "${TRAIN_SEED}"
fi
merge_adapter_to_base_model \
  "${WARM_MODEL}" \
  "${RECOVERY_ROOT}/final" \
  "${RECOVERY_ROOT}/merged" \
  "${CONTRACT_MANIFEST}"

# Recovery-DPO: pair-level preference on the exact same failed student draws.
DPO_ROOT="${QZBC_RUN_ROOT}/baselines/recovery_dpo"
if [[ ! -f "${DPO_ROOT}/final/run_metadata.json" ]]; then
  run_train_module care_opd.train_dpo \
    --config "${CARE_CONFIG}" \
    --model "${WARM_MODEL}" \
    --data "${SOURCE_ROLLOUTS}/qzbc_dpo.jsonl" \
    --output "${DPO_ROOT}" \
    --resume_from_checkpoint auto \
    --seed "${TRAIN_SEED}"
fi
merge_adapter_to_base_model \
  "${WARM_MODEL}" \
  "${DPO_ROOT}/final" \
  "${DPO_ROOT}/merged" \
  "${CONTRACT_MANIFEST}"

for mode in gkd_full tropd_full; do
  method_root="${QZBC_RUN_ROOT}/baselines/${mode}"
  current_model="${WARM_MODEL}"
  for round in $(seq 1 "${ONLINE_ROUNDS}"); do
    round_tag="$(printf '%02d' "${round}")"
    round_root="${method_root}/round_${round_tag}"
    if [[ "${round}" -eq 1 ]]; then
      rollout_root="${SOURCE_ROLLOUTS}"
    else
      rollout_root="${round_root}/rollouts"
      if [[ ! -f "${rollout_root}/summary.json" ]]; then
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
      fi
    fi
    # Freshly recollected rounds must remain usable under the deployed qzbc
    # parser.  Stop before training if generation collapses into formatting
    # failures; round 1 intentionally reuses the historical common collection.
    if [[ "${round}" -gt 1 ]]; then
      rollout_total="$(json_int_field "${rollout_root}/summary.json" rollouts)"
      rollout_passed="$(json_int_field "${rollout_root}/summary.json" passed)"
      rollout_format_failures="$(
        json_int_field "${rollout_root}/summary.json" format_failures_total
      )"
      if [[ "${rollout_total}" -le 0 \
        || "${rollout_passed}" -le 0 \
        || $((rollout_format_failures * 10)) -gt "${rollout_total}" ]]; then
        printf \
          '%s round %s failed rollout health gate: total=%s passed=%s format_failures=%s\n' \
          "${mode}" "${round}" "${rollout_total}" "${rollout_passed}" \
          "${rollout_format_failures}" >&2
        exit 4
      fi
    fi

    round_failures="$(json_int_field "${rollout_root}/summary.json" opd_failures)"
    if [[ "${round_failures}" -eq 0 ]]; then
      printf '%s converged before round %s.\n' "${mode}" "${round}"
      break
    fi
    require_file "${rollout_root}/qzbc_opd.jsonl"

    train_root="${round_root}/opd_train"
    if [[ ! -f "${train_root}/final/run_metadata.json" ]]; then
      run_train_module care_opd.train_opd \
        --config "${CARE_CONFIG}" \
        --model "${current_model}" \
        --teacher "${TEACHER_MODEL}" \
        --data "${rollout_root}/qzbc_opd.jsonl" \
        --output "${train_root}" \
        --baseline_mode "${mode}" \
        --resume_from_checkpoint auto \
        --seed "$((TRAIN_SEED + round - 1))"
    fi
    if [[ "${round}" -eq "${ONLINE_ROUNDS}" ]]; then
      merged_model="${method_root}/final_model"
    else
      merged_model="${round_root}/merged_model"
    fi
    merge_adapter_to_base_model \
      "${current_model}" \
      "${train_root}/final" \
      "${merged_model}" \
      "${CONTRACT_MANIFEST}"
    current_model="${merged_model}"
  done
  if [[ ! -f "${method_root}/final_model/config.json" ]]; then
    ln -sfn "$(realpath "${current_model}")" "${method_root}/final_model"
  fi
done

printf 'Relevant qzbc base-model baselines ready under: %s\n' \
  "${QZBC_RUN_ROOT}/baselines"
