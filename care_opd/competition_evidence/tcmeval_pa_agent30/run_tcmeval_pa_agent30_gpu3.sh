#!/usr/bin/env bash
set -Eeuo pipefail

# Run the complete Shennong Sanyan project and the three already-frozen local
# baselines sequentially on physical GPU 3.  All four receive the same 30
# frozen TCMEval-PA user payloads.  Keep row-level outputs internal.

umask 077

PHASE="${1:-all}"
case "$PHASE" in
  prepare|preflight|run|verify|score|all) ;;
  *) printf 'usage: %s {prepare|preflight|run|verify|score|all}\n' "$0" >&2; exit 2 ;;
esac

CARE_ROOT="${CARE_ROOT:-/opt/care-workspace/four_agent_expert_blind_bundle/care_opd}"
QZBC_ROOT="${QZBC_ROOT:-/opt/care-workspace/four_agent_expert_blind_bundle/qzbc_private_eval}"
MODEL_ROOT="${MODEL_ROOT:-/opt/care-data/four_agent_expert_blind/models}"
CARE_MODEL="${CARE_MODEL:-/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model}"
EVAL_PY="${EVAL_PY:-$(command -v python || true)}"
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
CARE_VLLM_BIN="${CARE_VLLM_BIN:-$VLLM_BIN}"
BASELINE_VLLM_BIN="${BASELINE_VLLM_BIN:-$VLLM_BIN}"
CARE_SERVER_BACKEND="${CARE_SERVER_BACKEND:-transformers}"
CARE_HF_PY="${CARE_HF_PY:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
BACKEND_PY="${BACKEND_PY:-${QZBC_ROOT}/.venv/bin/python}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-3}"

DATA="${TCMEVAL_DATA:-${CARE_ROOT}/public_eval/tcmeval_pa/data/TCMEval-PA.xlsx}"
TASK_ROOT="${TCMEVAL_TASK_ROOT:-${CARE_ROOT}/runs/public_eval/tcmeval_pa_agent30_fixed_seed20260825}"
TASKS="${TASK_ROOT}/tcmeval_pa_agent30.tasks.jsonl"
SELECTION_MANIFEST="${TASK_ROOT}/tcmeval_pa_agent30.selection_manifest.json"
RUN_ROOT="${TCMEVAL_RUN_ROOT:-${TASK_ROOT}/four_system_runs}"
SCORE_ROOT="${TCMEVAL_SCORE_ROOT:-${TASK_ROOT}/scored}"

PREPARE="${CARE_ROOT}/competition_evidence/tcmeval_pa_agent30/prepare_tcmeval_pa_agent30.py"
SCORER="${CARE_ROOT}/competition_evidence/tcmeval_pa_agent30/score_tcmeval_pa_agent30.py"
FOUR_RUNNER="${CARE_ROOT}/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh"
MODEL_SPEC="${CARE_ROOT}/competition_evidence/local_four_agent_server/model_suite.json"
SELECTION_LOCK="${MODEL_ROOT}/suite_selection_lock.private.json"
DOWNLOAD_MANIFEST="${MODEL_ROOT}/suite_download_manifest.private.json"

require_file() {
  [[ -s "$1" ]] || { printf 'missing required file: %s\n' "$1" >&2; exit 2; }
}

prepare_sample() {
  require_file "$EVAL_PY"
  require_file "$PREPARE"
  require_file "$DATA"
  "$EVAL_PY" "$PREPARE" \
    --data "$DATA" \
    --output-dir "$TASK_ROOT" \
    --run-root "$RUN_ROOT"
}

check_model_downloads() {
  require_file "$DOWNLOAD_MANIFEST"
  require_file "$SELECTION_LOCK"
  "$EVAL_PY" - "$DOWNLOAD_MANIFEST" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("status") == "COMPLETE" and x.get("model_count") == 3, x
print("MODEL_DOWNLOAD_COMPLETE=3")
PY
  for path in \
    "$MODEL_ROOT/huatuogpt_3_8b/config.json" \
    "$MODEL_ROOT/glm_4_9b_0414/config.json" \
    "$MODEL_ROOT/internlm3_8b_instruct/config.json"; do
    require_file "$path"
  done
}

export CARE_ROOT QZBC_ROOT MODEL_ROOT CARE_MODEL EVAL_PY VLLM_BIN
export CARE_VLLM_BIN BASELINE_VLLM_BIN BACKEND_PY
export CARE_SERVER_BACKEND CARE_HF_PY
export PHYSICAL_GPUS DATA TASKS RUN_ROOT
export EXPECTED_TASKS=30
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

case "$PHASE" in
  prepare)
    prepare_sample
    ;;
  preflight)
    prepare_sample
    check_model_downloads
    bash "$FOUR_RUNNER" preflight
    ;;
  run)
    prepare_sample
    check_model_downloads
    bash "$FOUR_RUNNER" run
    ;;
  verify)
    require_file "$TASKS"
    bash "$FOUR_RUNNER" verify
    ;;
  score)
    require_file "$TASKS"
    require_file "$SELECTION_MANIFEST"
    "$EVAL_PY" "$SCORER" \
      --data "$DATA" \
      --tasks "$TASKS" \
      --selection-manifest "$SELECTION_MANIFEST" \
      --run-root "$RUN_ROOT" \
      --output-dir "$SCORE_ROOT"
    ;;
  all)
    prepare_sample
    check_model_downloads
    bash "$FOUR_RUNNER" run
    "$EVAL_PY" "$SCORER" \
      --data "$DATA" \
      --tasks "$TASKS" \
      --selection-manifest "$SELECTION_MANIFEST" \
      --run-root "$RUN_ROOT" \
      --output-dir "$SCORE_ROOT"
    ;;
esac

printf 'TCMEVAL_PA_AGENT30_PHASE=%s COMPLETE\n' "$PHASE"
