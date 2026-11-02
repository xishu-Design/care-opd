#!/usr/bin/env bash
# Run this file using bash, never source it into an interactive shell.
set -euo pipefail
umask 077
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PHASE=${1:-status}
WORK=${WORK:-/opt/care-data/care_opd_multimedia_boundary_gpu1_v1}
DATA_ROOT=${DATA_ROOT:-/opt/care-data/care_opd_multimedia_boundary_chartqa_val_v1}
HF_PY=${HF_PY:-/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python}
DATA_PY=${DATA_PY:-/opt/care-data/venvs/care_boundary_data/bin/python}
PHYSICAL_GPU=${PHYSICAL_GPU:-1}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

cpu_python() {
  if test -x "$DATA_PY"; then printf '%s\n' "$DATA_PY"; else printf '%s\n' "$HF_PY"; fi
}
check_package() { "$(cpu_python)" -B "$SCRIPT_DIR/check_package.py"; }
prepare() {
  if test -s "$WORK/selection_manifest.json"; then
    "$(cpu_python)" -B "$SCRIPT_DIR/boundary_run.py" status --work "$WORK"
    return
  fi
  local download_args=()
  if [[ "${IGNORE_PROXY:-0}" == 1 ]]; then download_args+=(--ignore-proxy); fi
  # Download environment applies only to this subprocess. Generation remains offline.
  env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DISABLE_XET=1 \
    "$DATA_PY" -B "$SCRIPT_DIR/boundary_fetch.py" --destination "$DATA_ROOT" \
    --revision "${CHARTQA_REVISION:-main}" "${download_args[@]}"
  "$DATA_PY" -B "$SCRIPT_DIR/boundary_prepare.py" \
    --source "$DATA_ROOT" --format chartqa-parquet --work "$WORK" --blocks 24 --seed 20260917
}
init() {
  if test -s "$WORK/config.json"; then
    echo "CONFIG_EXISTS: keeping frozen configuration $WORK/config.json"
    return
  fi
  local source_config=${SOURCE_CONFIG:-}
  if test -z "$source_config"; then
    for candidate in \
      /opt/care-data/care_opd_tmm_visual_followup_gpu1/experiment.json \
      /opt/care-data/care_opd_tmm_visual_gpu45_v4/config_validation10.json; do
      if test -s "$candidate"; then source_config=$candidate; break; fi
    done
  fi
  local args=()
  if test -n "$source_config"; then
    test -s "$source_config" || { echo "Missing SOURCE_CONFIG: $source_config" >&2; return 2; }
    args+=(--source-config "$source_config")
  fi
  if test -n "${VLM_MODEL:-}"; then args+=(--vlm "$VLM_MODEL"); fi
  "$(cpu_python)" -B "$SCRIPT_DIR/boundary_run.py" init --work "$WORK" "${args[@]}"
}
preflight() {
  "$HF_PY" -B "$SCRIPT_DIR/boundary_run.py" preflight --work "$WORK" --gpu "$PHYSICAL_GPU" --full-hash
}
generate() {
  "$HF_PY" -B "$SCRIPT_DIR/boundary_run.py" run --work "$WORK" --gpu "$PHYSICAL_GPU" "$@"
}
score() { "$(cpu_python)" -B "$SCRIPT_DIR/boundary_run.py" score --work "$WORK"; }

check_package
case "$PHASE" in
  selftest) (cd "$SCRIPT_DIR" && "$(cpu_python)" -B -m unittest discover -s . -p 'test_boundary*.py' -v) ;;
  prepare) prepare ;;
  init) init ;;
  preflight) preflight ;;
  smoke) generate --limit-blocks 1 ;;
  run|resume) generate; score ;;
  score) score ;;
  status) "$(cpu_python)" -B "$SCRIPT_DIR/boundary_run.py" status --work "$WORK" ;;
  all) prepare; init; preflight; generate; score ;;
  *) echo "Usage: bash $0 {selftest|prepare|init|preflight|smoke|run|resume|score|status|all}" >&2; exit 2 ;;
esac
