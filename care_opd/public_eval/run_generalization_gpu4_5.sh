#!/usr/bin/env bash
set -Eeuo pipefail

# Formal, sequential public-generalization suite.  The two additional primary
# public benchmarks are TCMBench V2 and MTCMB Clean9.  CMB-Exam is an optional
# broad medical-knowledge retention appendix because it is expensive and
# overlaps in format with the other multiple-choice evaluations.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"

PUBLIC_RUN_ROOT="${PUBLIC_RUN_ROOT:-/opt/care-workspace/care_opd/runs/public_eval}"
OFFICIAL_TCMEVAL_ROOT="${OFFICIAL_TCMEVAL_ROOT:-${PUBLIC_RUN_ROOT}/tcmeval_pa_v3}"
SOURCE_SUITE="${SOURCE_SUITE:-${OFFICIAL_TCMEVAL_ROOT}/model_suite.json}"
TCMEVAL_OFFICIAL_SUMMARY="${OFFICIAL_TCMEVAL_ROOT}/official/suite_summary.json"
TCMEVAL_DATA="${SCRIPT_DIR}/tcmeval_pa/data/TCMEval-PA.xlsx"
TCMEVAL_EVALUATOR="${SCRIPT_DIR}/tcmeval_pa/evaluate_tcmeval_pa.py"
PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK:-${SCRIPT_DIR}/PREREGISTRATION_LOCK.json}"
ORIGIN_REFERENCE="${ORIGIN_REFERENCE:-}"
LINEAGE_AUDIT="${LINEAGE_AUDIT:-${PUBLIC_RUN_ROOT}/model_lineage_audit.json}"
FROZEN_MODEL="${FROZEN_MODEL:-/opt/care-workspace/models/Qwen3.5-4B}"
CARE_WARM_START="${CARE_WARM_START:-/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged}"
BASELINE_WARM_START="${BASELINE_WARM_START:-/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/warm_start/merged}"

INTERNAL_REQUESTS="${INTERNAL_REQUESTS:-/opt/care-workspace/care_opd/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl}"
INTERNAL_EVAL_ROOT="${INTERNAL_EVAL_ROOT:-/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baseline_hf_evaluation}"
QZBC_ROOT="${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}"
QZBC_DATA_ROOT="${QZBC_DATA_ROOT:-/opt/care-workspace/care_opd/data/qzbc_formal}"
QZBC_GROUPED_OUT="${QZBC_GROUPED_OUT:-${PUBLIC_RUN_ROOT}/qzbc_grouped_identity_v1}"
LEGACY_INTERNAL_OUT="${LEGACY_INTERNAL_OUT:-${PUBLIC_RUN_ROOT}/internal_qzbc_legacy_v1}"
TCMEVAL_CONTROLLED_OUT="${TCMEVAL_CONTROLLED_OUT:-${PUBLIC_RUN_ROOT}/tcmeval_pa_controlled_v1}"
TCMBENCH_OUT="${TCMBENCH_OUT:-${PUBLIC_RUN_ROOT}/tcmbench_v2}"
MTCMB_OUT="${MTCMB_OUT:-${PUBLIC_RUN_ROOT}/mtcmb_clean_v1}"
CMB_OUT="${CMB_OUT:-${PUBLIC_RUN_ROOT}/cmb_exam_v1}"
MAIN_TABLE_OUT="${MAIN_TABLE_OUT:-${PUBLIC_RUN_ROOT}/generalization_main_table_v1}"

TCMBENCH_REPO="${TCMBENCH_REPO:-/opt/care-workspace/TCMBench}"
MTCMB_REPO="${MTCMB_REPO:-/opt/care-workspace/MTCMB}"
MTCMB_TRAIN_SEED_FILE="${MTCMB_TRAIN_SEED_FILE:-/opt/care-workspace/care_opd/data/qzbc_formal/seeds/qzbc_native_seeds.train.jsonl}"
BERT_MODEL="${BERT_MODEL:-/opt/care-workspace/models/bert-base-chinese}"
RUN_CMB="${RUN_CMB:-0}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICES="${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"

# Freeze the execution knobs that wrappers otherwise allow callers to
# override.  Their values are also written into each run contract.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p -- "${PUBLIC_RUN_ROOT}/logs" "${MAIN_TABLE_OUT}"
exec 8>"${PUBLIC_RUN_ROOT}/.generalization_suite.lock"
if ! flock -n 8; then
  printf 'Another formal generalization suite owns %s\n' \
    "${PUBLIC_RUN_ROOT}/.generalization_suite.lock" >&2
  exit 3
fi

for required in "${PYTHON_BIN}" "${SOURCE_SUITE}" \
  "${INTERNAL_REQUESTS}" "${QZBC_ROOT}" "${BERT_MODEL}" \
  "${FROZEN_MODEL}" "${CARE_WARM_START}" "${BASELINE_WARM_START}" \
  "${TCMEVAL_DATA}" "${TCMEVAL_EVALUATOR}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Required formal-evaluation input is missing: %s\n' "${required}" >&2
    exit 2
  fi
done
case "${RUN_CMB}" in
  0|1) ;;
  *) printf 'RUN_CMB must be 0 or 1, got %s\n' "${RUN_CMB}" >&2; exit 2 ;;
esac
if [[ "${PHYSICAL_GPUS}" != "4,5,6,7" || "${NUM_WORKERS}" != "4" || \
      "${DEVICES}" != "cuda:0,cuda:1,cuda:2,cuda:3" ]]; then
  printf 'Formal four-GPU run requires PHYSICAL_GPUS=4,5,6,7, NUM_WORKERS=4 and DEVICES=cuda:0,cuda:1,cuda:2,cuda:3\n' >&2
  exit 2
fi

# Rebuild the observed official aggregate from its validated per-model
# artifacts.  This prevents a same-label summary file from being substituted
# independently of the run directory that the lineage audit checks.
"${PYTHON_BIN}" "${TCMEVAL_EVALUATOR}" summarize-suite \
  --data "${TCMEVAL_DATA}" \
  --suite "${SOURCE_SUITE}" \
  --output-root "${OFFICIAL_TCMEVAL_ROOT}" \
  --prompt-mode official
if [[ ! -s "${TCMEVAL_OFFICIAL_SUMMARY}" ]]; then
  printf 'Validated TCMEval official summary was not created: %s\n' \
    "${TCMEVAL_OFFICIAL_SUMMARY}" >&2
  exit 2
fi

# Rehash the six checkpoints, all existing TCMEval run contracts, both warm
# starts and every export ancestry before freezing the public-evaluation lock.
# Exit 3 is expected when the bytes are locally consistent but the upstream
# origin was not independently attested; local_lineage_status must still PASS.
lineage_args=(
  --suite "${SOURCE_SUITE}"
  --eval-root "${OFFICIAL_TCMEVAL_ROOT}"
  --prompt-mode official
  --expected-frozen "${FROZEN_MODEL}"
  --care-warm-start "${CARE_WARM_START}"
  --baseline-warm-start "${BASELINE_WARM_START}"
  --rehash-suite
  --output "${LINEAGE_AUDIT}"
)
if [[ -n "${ORIGIN_REFERENCE}" ]]; then
  if [[ ! -s "${ORIGIN_REFERENCE}" ]]; then
    printf 'ORIGIN_REFERENCE does not exist or is empty: %s\n' "${ORIGIN_REFERENCE}" >&2
    exit 2
  fi
  lineage_args+=(--origin-reference "${ORIGIN_REFERENCE}")
fi
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_model_lineage.py" "${lineage_args[@]}"
lineage_rc=$?
set -e
if [[ "${lineage_rc}" -ne 0 && "${lineage_rc}" -ne 3 ]]; then
  printf 'Model-lineage audit failed with exit code %s: %s\n' \
    "${lineage_rc}" "${LINEAGE_AUDIT}" >&2
  exit "${lineage_rc}"
fi
"${PYTHON_BIN}" - "${LINEAGE_AUDIT}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("local_lineage_status") != "PASS":
    raise SystemExit(f"local model lineage did not close: {path}")
if not report.get("warm_start_content_identical"):
    raise SystemExit(f"CARE/baseline warm starts differ: {path}")
print(
    "[lineage] local_lineage_status=PASS "
    f"origin_status={report.get('origin_status')} raw_initial_verified="
    f"{report.get('raw_initial_base_verified')}"
)
PY

locker_args=(
  --project-root "${PROJECT_ROOT}"
  --model-suite "${SOURCE_SUITE}"
  --bert-model "${BERT_MODEL}"
  --output "${PREREGISTRATION_LOCK}"
)
if [[ -n "${ORIGIN_REFERENCE}" ]]; then
  locker_args+=(--origin-reference "${ORIGIN_REFERENCE}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/create_preregistration_lock.py" \
  "${locker_args[@]}"

# Preserve the old 4,200-row results as a clearly marked retrospective
# diagnostic when they are present.  They never feed the formal main table.
legacy_ready=1
for label in frozen clean_sft recovery_sft recovery_dpo gkd_full care_opd; do
  [[ -s "${INTERNAL_EVAL_ROOT}/${label}/qzbc_rollouts.jsonl" ]] || legacy_ready=0
done
if [[ "${legacy_ready}" == 1 ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_internal_qzbc.py" \
    --requests "${INTERNAL_REQUESTS}" \
    --eval-root "${INTERNAL_EVAL_ROOT}" \
    --model-suite "${SOURCE_SUITE}" \
    --output-dir "${LEGACY_INTERNAL_OUT}"
else
  printf '[qzbc-legacy] status=skipped reason=incomplete_old_rollouts root=%s\n' \
    "${INTERNAL_EVAL_ROOT}"
fi

# Rerun the internal grouped benchmark against the exact frozen six-model
# suite.  This binds checkpoint content, the current train/dev/test audit,
# inference runtime and completion artifacts; the result-blind family-clean
# view is derived from the same 4,200 predictions without extra inference.
env \
  PYTHON_BIN="${PYTHON_BIN}" \
  QZBC_ROOT="${QZBC_ROOT}" \
  QZBC_DATA_ROOT="${QZBC_DATA_ROOT}" \
  SOURCE_SUITE="${SOURCE_SUITE}" \
  PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK}" \
  OUT_ROOT="${QZBC_GROUPED_OUT}" \
  PHYSICAL_GPUS="${PHYSICAL_GPUS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICES="${DEVICES}" \
  bash "${SCRIPT_DIR}/qzbc_grouped/run_existing_gpu4_5.sh"

env \
  PYTHON_BIN="${PYTHON_BIN}" \
  SOURCE_SUITE="${SOURCE_SUITE}" \
  OFFICIAL_SUITE_ROOT="${OFFICIAL_TCMEVAL_ROOT}" \
  OFFICIAL_RESULTS_ROOT="${OFFICIAL_TCMEVAL_ROOT}/official" \
  OUT_ROOT="${TCMEVAL_CONTROLLED_OUT}" \
  PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK}" \
  ORIGIN_REFERENCE="${ORIGIN_REFERENCE}" \
  BERT_MODEL="${BERT_MODEL}" \
  PHYSICAL_GPUS="${PHYSICAL_GPUS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICES="${DEVICES}" \
  bash "${SCRIPT_DIR}/tcmeval_pa/run_controlled_gpu4_5.sh"

env \
  PYTHON_BIN="${PYTHON_BIN}" \
  TCMBENCH_REPO="${TCMBENCH_REPO}" \
  OUT_ROOT="${TCMBENCH_OUT}" \
  PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK}" \
  PROTOCOL=answer_only \
  TRACK=robustness \
  MAX_NEW_TOKENS=24 \
  PHYSICAL_GPUS="${PHYSICAL_GPUS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICES="${DEVICES}" \
  bash "${SCRIPT_DIR}/tcmbench_v2/run_existing_gpu4_5.sh"

env \
  PYTHON_BIN="${PYTHON_BIN}" \
  MTCMB_REPO="${MTCMB_REPO}" \
  TRAIN_SEED_FILE="${MTCMB_TRAIN_SEED_FILE}" \
  BERT_MODEL="${BERT_MODEL}" \
  OUT_ROOT="${MTCMB_OUT}" \
  PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK}" \
  ORIGIN_REFERENCE="${ORIGIN_REFERENCE}" \
  GENERATION_BATCH_SIZE=8 \
  BERT_BATCH_SIZE=32 \
  PHYSICAL_GPUS="${PHYSICAL_GPUS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICES="${DEVICES}" \
  bash "${SCRIPT_DIR}/mtcmb_clean/run_existing_gpu4_5.sh"

if [[ "${RUN_CMB}" == 1 ]]; then
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    OUT_ROOT="${CMB_OUT}" \
    PREREGISTRATION_LOCK="${PREREGISTRATION_LOCK}" \
    PHYSICAL_GPUS="${PHYSICAL_GPUS}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    DEVICES="${DEVICES}" \
    bash "${SCRIPT_DIR}/cmb_exam/run_existing_gpu4_5.sh"
fi

required_summaries=(
  "${QZBC_GROUPED_OUT}/formal/suite_summary.json"
  "${TCMEVAL_CONTROLLED_OUT}/suite_summary.json"
  "${TCMBENCH_OUT}/answer_only/suite_summary.json"
  "${MTCMB_OUT}/official/suite_summary.json"
)
if [[ "${RUN_CMB}" == 1 ]]; then
  required_summaries+=("${CMB_OUT}/official/suite_summary.json")
fi
for summary in "${required_summaries[@]}"; do
  if [[ ! -s "${summary}" ]]; then
    printf 'Required suite summary is missing after evaluation: %s\n' "${summary}" >&2
    exit 2
  fi
done

summary_args=(
  --internal-qzbc "${QZBC_GROUPED_OUT}/formal/suite_summary.json"
  --tcmeval-official "${TCMEVAL_OFFICIAL_SUMMARY}"
  --tcmeval-controlled "${TCMEVAL_CONTROLLED_OUT}/suite_summary.json"
  --tcmbench "${TCMBENCH_OUT}/answer_only/suite_summary.json"
  --mtcmb "${MTCMB_OUT}/official/suite_summary.json"
  --preregistration-lock "${PREREGISTRATION_LOCK}"
  --output-dir "${MAIN_TABLE_OUT}"
  --require-complete
)
if [[ "${RUN_CMB}" == 1 ]]; then
  summary_args+=(--cmb "${CMB_OUT}/official/suite_summary.json")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_generalization.py" \
  "${summary_args[@]}"

printf 'Formal generalization suite complete. Main table: %s\n' \
  "${MAIN_TABLE_OUT}/main_table.md"
