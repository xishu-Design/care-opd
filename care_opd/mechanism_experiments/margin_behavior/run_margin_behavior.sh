#!/usr/bin/env bash
set -Eeuo pipefail

# Pure offline analysis: no model loading, generation, training, or GPU use.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
MECHANISM_ROOT="${MECHANISM_ROOT:-${PROJECT_ROOT}/runs/mechanism_eval/qzbc_care_opd_v1}"
FORMAL_ROOT="${FORMAL_ROOT:-${PROJECT_ROOT}/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal}"
OUTPUT_DIR="${OUTPUT_DIR:-${MECHANISM_ROOT}/margin_behavior}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260813}"

MARGIN_JSONL="${MARGIN_JSONL:-${MECHANISM_ROOT}/recovery_margin/recovery_margin.jsonl}"
CLEAN_ROLLOUTS="${CLEAN_ROLLOUTS:-${FORMAL_ROOT}/family_clean_views/clean_sft.jsonl}"
GKD_ROLLOUTS="${GKD_ROLLOUTS:-${FORMAL_ROOT}/family_clean_views/gkd_full.jsonl}"
CARE_ROLLOUTS="${CARE_ROLLOUTS:-${FORMAL_ROOT}/family_clean_views/care_opd.jsonl}"

for required in \
  "${PYTHON_BIN}" "${MARGIN_JSONL}" "${CLEAN_ROLLOUTS}" \
  "${GKD_ROLLOUTS}" "${CARE_ROLLOUTS}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing margin-behavior input: %s\n' "${required}" >&2
    exit 2
  fi
done

mkdir -p -- "${OUTPUT_DIR}"

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/analyze_margin_behavior.py" \
  --margin-jsonl "${MARGIN_JSONL}" \
  --clean-rollouts "${CLEAN_ROLLOUTS}" \
  --gkd-rollouts "${GKD_ROLLOUTS}" \
  --care-rollouts "${CARE_ROLLOUTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --overwrite

if "${PYTHON_BIN}" -c 'import matplotlib, numpy' >/dev/null 2>&1; then
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/plot_margin_behavior.py" \
    --analysis "${OUTPUT_DIR}/margin_behavior.json" \
    --output-dir "${OUTPUT_DIR}"
else
  printf 'Analysis complete; skipping optional figure (matplotlib/numpy unavailable).\n' >&2
fi

printf 'Margin-to-behavior analysis complete: %s\n' "${OUTPUT_DIR}"
