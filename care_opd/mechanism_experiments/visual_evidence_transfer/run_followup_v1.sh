#!/usr/bin/env bash
# New output root only. No downloads, training, old-output rewriting or GPU killing.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HF_PY="${HF_PY:-/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python}"
SOURCE_CONFIG="${SOURCE_CONFIG:-/opt/care-data/care_opd_tmm_visual_gpu45_v4/config_validation10.json}"
FOLLOWUP_WORK="${FOLLOWUP_WORK:-/opt/care-data/care_opd_tmm_visual_followup_v1}"
PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
PHASE="${1:-all}"
if [[ ! "$PHYSICAL_GPU" =~ ^[0-9]+$ ]]; then
  echo 'PHYSICAL_GPU must name exactly one physical card' >&2
  exit 2
fi
test -x "$HF_PY" || { echo "Python unavailable: $HF_PY" >&2; exit 2; }
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
umask 077
invoke() { "$HF_PY" -B -u "$SCRIPT_DIR/followup_run.py" "$@" --work "$FOLLOWUP_WORK"; }
case "$PHASE" in
  audit|prepare)
    invoke "$PHASE" --source-config "$SOURCE_CONFIG"
    ;;
  preflight)
    invoke preflight --full-hash
    ;;
  run|score|status)
    invoke "$PHASE"
    ;;
  all)
    invoke prepare --source-config "$SOURCE_CONFIG"
    invoke preflight --full-hash
    invoke run
    ;;
  *)
    echo 'Usage: bash run_followup_v1.sh {all|audit|prepare|preflight|run|score|status}' >&2
    exit 2
    ;;
esac
