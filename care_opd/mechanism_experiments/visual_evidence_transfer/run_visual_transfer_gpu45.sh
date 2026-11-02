#!/usr/bin/env bash
# Explicit dual-device coordinator. No occupancy check, process kill or reset.
set -euo pipefail
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-help}"
CONFIG="${2:-}"
if [[ "$PHASE" == help || -z "$CONFIG" ]]; then
  echo 'Usage: HF_PY=/absolute/python bash run_visual_transfer_gpu45.sh preflight|all CONFIG.json'
  echo 'GPU4: shared observations, then direct + GKD; GPU5: Clean-SFT + TROPD + CARE.'
  exit 0
fi
: "${HF_PY:?Set HF_PY to the proven Qwen3.5 text and Qwen2.5-VL Python environment}"
[[ -x "$HF_PY" ]] || { echo "Missing interpreter: $HF_PY" >&2; exit 2; }
[[ -s "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Each fresh spawned worker explicitly selects its own single physical GPU.
unset CUDA_VISIBLE_DEVICES
exec "$HF_PY" -u "$SCRIPT_DIR/gpu45.py" "$PHASE" --config "$CONFIG"
