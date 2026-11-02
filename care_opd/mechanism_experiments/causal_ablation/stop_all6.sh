#!/usr/bin/env bash
# Safely stop only the detached causal-ablation process group recorded by a run.

set -euo pipefail

CARE_OPD_ROOT="${CARE_OPD_ROOT:-/opt/care-workspace/care_opd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CARE_OPD_ROOT/runs/mechanism_ablation/single_round_all6_v2}"
WAIT_SECONDS=30
FORCE=false

usage() {
  printf '%s\n' \
    'Usage: bash stop_all6.sh [--output-root PATH] [--wait-seconds N] [--force]' \
    '' \
    'The script validates PID, session ID, command, and plan path before it' \
    'signals the recorded process group. --force permits SIGKILL after timeout.'
}

while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$WAIT_SECONDS" =~ ^[0-9]+$ ]]; then
  printf '%s\n' '--wait-seconds must be a non-negative integer.' >&2
  exit 2
fi

OUTPUT_ROOT="$(readlink -m "$OUTPUT_ROOT")"
PID_FILE="$OUTPUT_ROOT/launcher.pid"
PLAN="$OUTPUT_ROOT/experiment_plan.json"

if [[ ! -f "$PID_FILE" ]]; then
  printf 'No launcher PID file: %s\n' "$PID_FILE" >&2
  exit 2
fi

pid="$(tr -d '[:space:]' <"$PID_FILE")"
if ! [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Invalid launcher PID file: %s\n' "$PID_FILE" >&2
  exit 2
fi
if ! kill -0 "$pid" 2>/dev/null; then
  printf 'Launcher is not running (stale pid=%s); no signal sent.\n' "$pid"
  exit 0
fi

cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
session_id="$(ps -o sid= -p "$pid" | tr -d '[:space:]')"
if [[ "$session_id" != "$pid" ]]; then
  printf 'Refusing to signal: pid=%s is not its own session leader (sid=%s).\n' \
    "$pid" "$session_id" >&2
  exit 2
fi
if [[ "$cmdline" != *"mechanism_experiments/causal_ablation/launch_parallel.py"* ]]; then
  printf 'Refusing to signal unexpected command for pid=%s: %s\n' "$pid" "$cmdline" >&2
  exit 2
fi
if [[ "$cmdline" != *"$PLAN"* ]]; then
  printf 'Refusing to signal launcher bound to another plan: %s\n' "$cmdline" >&2
  exit 2
fi

printf 'Sending SIGTERM to causal-ablation process group %s\n' "$pid"
kill -TERM -- "-$pid"

deadline=$((SECONDS + WAIT_SECONDS))
while kill -0 "$pid" 2>/dev/null && ((SECONDS < deadline)); do
  sleep 1
done

if kill -0 "$pid" 2>/dev/null; then
  if [[ "$FORCE" == true ]]; then
    printf 'Grace period expired; sending SIGKILL to process group %s\n' "$pid" >&2
    kill -KILL -- "-$pid"
  else
    printf 'Process group is still alive after %ss; no SIGKILL sent. Re-run with --force if required.\n' \
      "$WAIT_SECONDS" >&2
    exit 1
  fi
fi

printf 'ALL6_STOP=COMPLETE pid=%s output_root=%s\n' "$pid" "$OUTPUT_ROOT"
