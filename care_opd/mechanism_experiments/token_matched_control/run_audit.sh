#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/care-workspace/care_opd}"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
DATA="${DATA:-$ROOT/runs/mechanism_ablation/views/full_selected.jsonl}"
TOKENIZER="${TOKENIZER:-$ROOT/runs/qzbc_base_model_3gpu/warm_start/merged}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/mechanism_experiments/token_matched_control_audit_v1}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -u \
  mechanism_experiments/token_matched_control/audit_token_matching.py \
  --data "$DATA" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUTPUT_DIR" \
  --seed 20260815 \
  --max-length 4096 \
  --max-response-tokens 1200 \
  --add-eos

