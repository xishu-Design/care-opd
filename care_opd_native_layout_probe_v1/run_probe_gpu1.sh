#!/usr/bin/env bash
# Execute with bash. Never source this file into an interactive login shell.
set -euo pipefail
umask 077
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PHASE=${1:-status}
HF_PY=${HF_PY:?Set HF_PY to your existing GPU Python executable}
WORK=${WORK:?Set WORK to a NEW output directory}
PHYSICAL_GPU=${PHYSICAL_GPU:-1}
BLOCKS=${BLOCKS:-3}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -x "$HF_PY" || { echo "Missing Python: $HF_PY" >&2; exit 2; }
"$HF_PY" -B "$SCRIPT_DIR/check_package.py"
args=("$PHASE" --work "$WORK" --gpu "$PHYSICAL_GPU" --blocks "$BLOCKS")
case "$PHASE" in
  prepare|all)
    SOURCE_WORK=${SOURCE_WORK:?Set SOURCE_WORK to the completed original boundary pilot}
    args+=(--source "$SOURCE_WORK") ;;
  selftest)
    cd "$SCRIPT_DIR"
    exec "$HF_PY" -B -m unittest discover -s . -p 'test_probe*.py' -v ;;
  check|preflight|run|resume|status|score) ;;
  *) echo "Usage: bash $0 {prepare|check|preflight|run|resume|score|status|all|selftest}" >&2; exit 2 ;;
esac
if [[ "${FULL_HASH:-0}" == 1 ]]; then args+=(--full-hash); fi
exec "$HF_PY" -B -u "$SCRIPT_DIR/probe_run.py" "${args[@]}"
