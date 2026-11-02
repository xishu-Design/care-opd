#!/usr/bin/env bash
set -Eeuo pipefail

# CPU-only mechanism diagnostics.  These consume already-completed formal
# rollouts and never load, generate from, or update a language model.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
REQUESTS="${REQUESTS:-${PROJECT_ROOT}/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl}"
FORMAL_ROOT="${FORMAL_ROOT:-${PROJECT_ROOT}/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal}"
CARE_RUN_ROOT="${CARE_RUN_ROOT:-${PROJECT_ROOT}/runs/qzbc_base_model_3gpu}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/runs/mechanism_eval/qzbc_care_opd_v1}"
TOKENIZER="${TOKENIZER:-${PROJECT_ROOT}/runs/qzbc_baselines_3gpu/warm_start/merged}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

for required in \
  "${PYTHON_BIN}" "${REQUESTS}" "${TOKENIZER}" \
  "${FORMAL_ROOT}/family_clean_views/clean_sft.jsonl" \
  "${FORMAL_ROOT}/family_clean_views/gkd_full.jsonl" \
  "${FORMAL_ROOT}/family_clean_views/care_opd.jsonl"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing mechanism input: %s\n' "${required}" >&2
    exit 2
  fi
done

mkdir -p -- "${OUT_ROOT}"

printf '[1/4] build result-blind held-out Clean-SFT failure bank\n'
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/build_heldout_failure_bank.py" \
  --requests "${REQUESTS}" \
  --rollouts "${FORMAL_ROOT}/family_clean_views/clean_sft.jsonl" \
  --output "${OUT_ROOT}/heldout_failure_bank.jsonl"

printf '[2/4] decompose failures and matched rescues\n'
decomposition_args=(
  --requests "${REQUESTS}"
  --reference-label clean_sft
  --output-dir "${OUT_ROOT}/failure_decomposition"
)
for label in frozen clean_sft recovery_sft recovery_dpo gkd_full care_opd; do
  rollout="${FORMAL_ROOT}/family_clean_views/${label}.jsonl"
  if [[ -f "${rollout}" ]]; then
    decomposition_args+=(--rollout "${label}=${rollout}")
  fi
done
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_failure_decomposition.py" \
  "${decomposition_args[@]}" \
  --overwrite

printf '[3/4] audit verifier-localized supervision coverage\n'
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_localization.py" \
  --failure-bank "${OUT_ROOT}/heldout_failure_bank.jsonl" \
  --tokenizer "${TOKENIZER}" \
  --output-dir "${OUT_ROOT}/localization" \
  --local-files-only

printf '[4/4] summarize three-round on-policy dynamics when artifacts exist\n'
round_args=()
for round in 01 02 03; do
  round_dir="${CARE_RUN_ROOT}/round_${round}"
  if [[ -d "${round_dir}" ]]; then
    round_args+=(--round "${round}=${round_dir}")
  fi
done
if (( ${#round_args[@]} == 0 )); then
  printf 'No round directories found under %s; skipping round dynamics.\n' \
    "${CARE_RUN_ROOT}" >&2
else
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_round_dynamics.py" \
    "${round_args[@]}" \
    --output-dir "${OUT_ROOT}/round_dynamics" \
    --overwrite
fi

printf 'CPU mechanism diagnostics complete: %s\n' "${OUT_ROOT}"
