#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/care-workspace/care_opd}"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
QZBC_EVAL_ROOT="${QZBC_EVAL_ROOT:-${PROJECT_ROOT}/runs/public_eval/qzbc_grouped_identity_4gpu_v1}"
MODEL_SUITE="${MODEL_SUITE:-${QZBC_EVAL_ROOT}/model_suite.json}"
DATA_AUDIT="${DATA_AUDIT:-${QZBC_EVAL_ROOT}/DATA_AUDIT.json}"
REQUESTS="${REQUESTS:-${PROJECT_ROOT}/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/mechanism_eval/qzbc_care_opd_v1/efficiency_audit}"
PHYSICAL_GPU="${PHYSICAL_GPU:-4}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

env \
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON_BIN" -u \
  mechanism_experiments/efficiency_audit/efficiency_audit.py \
  --suite "$MODEL_SUITE" \
  --data-audit "$DATA_AUDIT" \
  --requests "$REQUESTS" \
  --output-dir "$OUTPUT_DIR" \
  --labels clean_sft,gkd_full,care_opd \
  --device cuda:0 \
  --dtype bfloat16 \
  --sample-size 200 \
  --warmup-requests 5 \
  --max-new-tokens 1200 \
  --seed 20260813 \
  "$@"
