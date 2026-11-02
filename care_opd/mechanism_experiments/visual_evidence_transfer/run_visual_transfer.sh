#!/usr/bin/env bash
# Execute with bash; errors exit this script, never the interactive parent shell.
set -euo pipefail
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-help}"
CONFIG="${2:-}"
if [[ "$PHASE" = help || -z "$CONFIG" ]]; then
  echo 'Usage: HF_PY=/absolute/python PHYSICAL_GPU=6 bash run_visual_transfer.sh preflight|all|observe|direct|clean_sft|gkd_full|tropd_full|care_opd|score CONFIG.json'
  exit 0
fi
: "${HF_PY:?Set HF_PY to an existing Python environment supporting qwen3_5_text and qwen2_5_vl}"
[[ -x "$HF_PY" ]] || { echo "Missing interpreter: $HF_PY" >&2; exit 2; }
[[ -s "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU:-6}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# No active-GPU-process check, GPU reset, process termination or service restart.
score() {
  "$HF_PY" "$SCRIPT_DIR/score_results.py" --config "$CONFIG"
}
case "$PHASE" in
  preflight|observe|direct)
    "$HF_PY" "$SCRIPT_DIR/evaluate.py" "$PHASE" --config "$CONFIG" ;;
  clean_sft|gkd_full|tropd_full|care_opd)
    "$HF_PY" "$SCRIPT_DIR/evaluate.py" generate --config "$CONFIG" --arm "$PHASE" ;;
  all)
    "$HF_PY" "$SCRIPT_DIR/evaluate.py" all --config "$CONFIG"
    score ;;
  score) score ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
