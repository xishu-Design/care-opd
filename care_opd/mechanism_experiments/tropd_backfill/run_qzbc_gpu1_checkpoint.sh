#!/usr/bin/env bash
# Explicit QZBC-only fresh/restart lane. Never touch visual or real-track roots.
set -Eeuo pipefail
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CARE_ROOT="${CARE_ROOT:-/opt/care-workspace/care_opd}"
export QZBC_ROOT="${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}"
export PY="${PY:-/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python}"
export TROPD_MODEL="${TROPD_MODEL:-$CARE_ROOT/runs/qzbc_baselines_3gpu/baselines/tropd_full/final_model}"
export CARE_MODEL="${CARE_MODEL:-$CARE_ROOT/runs/qzbc_base_model_3gpu/final_model}"
export BACKFILL_ROOT="${GPU1_CHECKPOINT_ROOT:-/opt/care-data/care_opd_tropd_backfill_gpu1_checkpoint_v1}"
export QZBC_OUT_ROOT="$BACKFILL_ROOT/qzbc_grouped_tropd_vs_care_v1"
export PHYSICAL_GPU=1 LOGICAL_DEVICE=cuda:0 QZBC_CHECKPOINT=1 RUN_MODE=fresh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
unset CUDA_VISIBLE_DEVICES STRUCTURED_OUT RAW_OUT
RUNNER="$SCRIPT_DIR/run_tropd_backfill.sh"
PID_FILE="$BACKFILL_ROOT/driver.pid"
LATEST_LOG="$BACKFILL_ROOT/latest_log.txt"

case "${1:-start}" in
  plan)
    env DRY_RUN=1 bash "$RUNNER" qzbc
    ;;
  start)
    test -x "$PY"
    test -s "$RUNNER"
    mkdir -p "$BACKFILL_ROOT"
    exec 8>"$BACKFILL_ROOT/.launcher.lock"
    flock -n 8 || { echo 'another launcher is active' >&2; exit 2; }
    if test -s "$PID_FILE"; then
      previous_pid="$(<"$PID_FILE")"
      if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        echo "PID still exists: $previous_pid; inspect it before starting again. No process was killed."
        ps -fp "$previous_pid" || true
        exit 2
      fi
    fi
    RUN_LOG="$BACKFILL_ROOT/qzbc_gpu1_$(date +%Y%m%d_%H%M%S)_$$.nohup.log"
    nohup env DRY_RUN=0 bash "$RUNNER" qzbc >"$RUN_LOG" 2>&1 </dev/null 8>&- &
    run_pid=$!
    printf '%s\n' "$run_pid" >"$PID_FILE"
    printf '%s\n' "$RUN_LOG" >"$LATEST_LOG"
    disown "$run_pid" 2>/dev/null || true
    printf 'PID=%s\nGPU=1\nROOT=%s\nLOG=%s\n' "$run_pid" "$BACKFILL_ROOT" "$RUN_LOG"
    sleep 3
    ps -fp "$run_pid" || true
    tail -n 80 "$RUN_LOG"
    ;;
  status)
    if test -s "$PID_FILE"; then
      read -r run_pid <"$PID_FILE"
      if [[ "$run_pid" =~ ^[0-9]+$ ]]; then ps -fp "$run_pid" || true; fi
    fi
    if test -s "$LATEST_LOG"; then
      read -r RUN_LOG <"$LATEST_LOG"
      tail -n 100 "$RUN_LOG"
    fi
    for label in tropd_full care_opd; do
      state="$QZBC_OUT_ROOT/formal/$label/checkpoint_progress.json"
      if test -s "$state"; then
        printf '\nCHECKPOINT=%s\n' "$label"
        "$PY" -m json.tool "$state"
      fi
    done
    ;;
  follow)
    read -r RUN_LOG <"$LATEST_LOG"
    exec tail -F "$RUN_LOG"
    ;;
  *) echo 'Usage: bash run_qzbc_gpu1_checkpoint.sh start|status|follow|plan' >&2; exit 2 ;;
esac
