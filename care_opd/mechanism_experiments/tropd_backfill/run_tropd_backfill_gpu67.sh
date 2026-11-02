#!/usr/bin/env bash
# Two independent single-GPU lanes. This is NOT tensor parallelism.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_GPU_RUNNER="${SINGLE_GPU_RUNNER:-$SCRIPT_DIR/run_tropd_backfill.sh}"
PARALLEL_ROOT="${PARALLEL_ROOT:-/opt/care-data/care_opd_tropd_backfill_gpu67}"
export PY="${PY:-/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python}"
REAL_RUN_MODE="${REAL_RUN_MODE:-fresh}"
PHASE="${1:-all}"

case "$PHASE" in
  preflight|all) ;;
  -h|--help)
    echo 'Usage: bash run_tropd_backfill_gpu67.sh preflight|all'
    echo 'GPU6: QZBC TROPD + CARE; GPU7: structured then raw TROPD + CARE.'
    echo 'PARALLEL_ROOT selects a new isolated output parent. DRY_RUN=1 only prints commands.'
    exit 0 ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
[[ "$REAL_RUN_MODE" == fresh || "$REAL_RUN_MODE" == resume ]] || {
  echo 'REAL_RUN_MODE must be fresh or resume' >&2; exit 2;
}
[[ -s "$SINGLE_GPU_RUNNER" ]] || { echo "Missing runner: $SINGLE_GPU_RUNNER" >&2; exit 2; }
[[ "$PARALLEL_ROOT" == /* && "$PARALLEL_ROOT" != / ]] || {
  echo 'PARALLEL_ROOT must be an absolute, dedicated experiment directory' >&2; exit 2;
}

run_one() {
  local gpu="$1" stage="$2" lane mode
  if [[ "$gpu" == 6 ]]; then
    lane="$PARALLEL_ROOT/gpu6_qzbc"
    mode=fresh
  else
    lane="$PARALLEL_ROOT/gpu7_real"
    mode="$REAL_RUN_MODE"
  fi
  # Discard old output overrides: the two lanes MUST NOT share output/lock paths.
  # Model/data paths and Python may still be set by the caller.
  env -u CUDA_VISIBLE_DEVICES -u QZBC_OUT_ROOT -u STRUCTURED_OUT -u RAW_OUT -u MODEL_LOCK \
    BACKFILL_ROOT="$lane" PHYSICAL_GPU="$gpu" LOGICAL_DEVICE=cuda:0 RUN_MODE="$mode" \
    bash "$SINGLE_GPU_RUNNER" "$stage"
}

verify_lane_models() {
  "$PY" - "$PARALLEL_ROOT/gpu6_qzbc/MODEL_LOCK.json" "$PARALLEL_ROOT/gpu7_real/MODEL_LOCK.json" <<'PY'
import hashlib
import json
import sys

fingerprints = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        lock = json.load(handle)
    stable = {k: v for k, v in lock.items() if k not in {"contract_fingerprint", "created_at_utc"}}
    actual = hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if lock.get("contract_fingerprint") != actual:
        raise SystemExit("MODEL_LOCK_INVALID " + path)
    if set(lock.get("models", {})) != {"tropd_full", "care_opd"}:
        raise SystemExit("MODEL_SET_INVALID " + path)
    fingerprints.append(actual)
if fingerprints[0] != fingerprints[1]:
    raise SystemExit("GPU67_MODEL_MISMATCH: two lanes must use identical checkpoints")
print("GPU67_MODEL_IDENTITY=PASS", fingerprints[0])
PY
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "GPU67_PLAN GPU6=QZBC GPU7=structured_then_raw root=$PARALLEL_ROOT"
  if [[ "$PHASE" == preflight ]]; then
    run_one 6 preflight
    run_one 7 preflight
  else
    run_one 6 qzbc
    run_one 7 structured
    run_one 7 raw
  fi
  exit 0
fi

mkdir -p "$PARALLEL_ROOT"
command -v flock >/dev/null 2>&1 || { echo 'flock is required' >&2; exit 2; }
exec 8>"$PARALLEL_ROOT/.gpu67_coordinator.lock"
flock -n 8 || { echo "GPU67 run already active in $PARALLEL_ROOT" >&2; exit 2; }

# These validate inputs/dependencies, not GPU occupancy. No process is killed.
run_one 6 preflight
run_one 7 preflight
verify_lane_models
echo "GPU67_PREFLIGHT=PASS root=$PARALLEL_ROOT"
[[ "$PHASE" == all ]] || exit 0

run_real_lane() {
  if run_one 7 structured; then
    run_one 7 raw
  else
    local failed_rc=$?
    echo "GPU7_STRUCTURED_FAILED rc=$failed_rc; raw not started" >&2
    return "$failed_rc"
  fi
}

run_one 6 qzbc >>"$PARALLEL_ROOT/gpu6_qzbc.log" 2>&1 &
pid6=$!
printf '%s\n' "$pid6" >"$PARALLEL_ROOT/gpu6_qzbc.pid"
run_real_lane >>"$PARALLEL_ROOT/gpu7_real.log" 2>&1 &
pid7=$!
printf '%s\n' "$pid7" >"$PARALLEL_ROOT/gpu7_real.pid"
printf 'GPU67_LAUNCHED pid6=%s pid7=%s root=%s\n' "$pid6" "$pid7" "$PARALLEL_ROOT"

# Always reap both lanes, even if one fails. Successful sibling results remain intact.
rc6=0
rc7=0
wait "$pid6" || rc6=$?
wait "$pid7" || rc7=$?
printf 'GPU67_RESULT gpu6_rc=%s gpu7_rc=%s\n' "$rc6" "$rc7"
if (( rc6 != 0 || rc7 != 0 )); then
  echo 'GPU67_INCOMPLETE: inspect lane logs; do not relabel partial results as complete' >&2
  exit 2
fi
verify_lane_models
echo "GPU67_COMPLETE root=$PARALLEL_ROOT"
