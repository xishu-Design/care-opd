#!/usr/bin/env bash
# Build and detach a fresh, equal-budget six-arm causal-ablation run.

set -euo pipefail

CARE_OPD_ROOT="${CARE_OPD_ROOT:-/opt/care-workspace/care_opd}"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CARE_OPD_ROOT/runs/mechanism_ablation/single_round_all6_v2}"
VIEWS_DIR="${VIEWS_DIR:-$CARE_OPD_ROOT/runs/mechanism_ablation/views}"
STUDENT_MODEL="${STUDENT_MODEL:-$CARE_OPD_ROOT/runs/qzbc_base_model_3gpu/warm_start/merged}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
REFERENCE_MODEL="${REFERENCE_MODEL:-$STUDENT_MODEL}"
GPUS="${GPUS:-4,5,6,7}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
PLAN_ONLY=false

usage() {
  printf '%s\n' \
    'Usage: bash launch_all6_detached.sh [options]' \
    '' \
    'Options:' \
    '  --output-root PATH   Fresh output root (default: .../single_round_all6_v2)' \
    '  --python PATH        Python executable' \
    '  --views-dir PATH     Frozen causal-view directory' \
    '  --student PATH       Common initial student checkpoint' \
    '  --teacher ID_OR_PATH Common teacher checkpoint' \
    '  --reference PATH     Common frozen retention reference' \
    '  --gpus LIST          Physical GPU list (default: 4,5,6,7)' \
    '  --plan-only          Build and audit the plan without starting training' \
    '  -h, --help           Show this help'
}

while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --views-dir) VIEWS_DIR="$2"; shift 2 ;;
    --student) STUDENT_MODEL="$2"; shift 2 ;;
    --teacher) TEACHER_MODEL="$2"; shift 2 ;;
    --reference) REFERENCE_MODEL="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python executable is missing or not executable: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi

OUTPUT_ROOT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_ROOT")"
CARE_OPD_ROOT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CARE_OPD_ROOT")"

if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing to mix runs: output root already exists: %s\n' "$OUTPUT_ROOT" >&2
  printf 'Choose a new --output-root, or archive the interrupted directory first.\n' >&2
  exit 2
fi

MATRIX="$CARE_OPD_ROOT/mechanism_experiments/causal_ablation/ablation_matrix.json"
CONFIG="$CARE_OPD_ROOT/configs/qzbc_formal.yaml"
for required in "$MATRIX" "$CONFIG" "$VIEWS_DIR/full_selected.jsonl" "$VIEWS_DIR/outside_only.jsonl"; do
  if [[ ! -f "$required" ]]; then
    printf 'Required input is missing: %s\n' "$required" >&2
    exit 2
  fi
done

cd "$CARE_OPD_ROOT"

"$PYTHON_BIN" mechanism_experiments/causal_ablation/run_single_round.py \
  --matrix "$MATRIX" \
  --views-dir "$VIEWS_DIR" \
  --train-config "$CONFIG" \
  --student-model "$STUDENT_MODEL" \
  --teacher-model "$TEACHER_MODEL" \
  --reference-model "$REFERENCE_MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --max-steps 199 \
  --save-steps 100 \
  --seed 42 \
  --nproc-per-node 1 \
  --python "$PYTHON_BIN" \
  --resume

PLAN="$OUTPUT_ROOT/experiment_plan.json"
"$PYTHON_BIN" -c '
import json, sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = [
    "full_selected", "without_retention", "matched_full_response",
    "outside_only", "without_risk", "without_trust",
]
actual = [arm.get("name") for arm in plan.get("arms", [])]
if actual != expected:
    raise SystemExit(f"six-arm closure failed: expected={expected!r}, actual={actual!r}")
budget = plan.get("budget_contract", {})
wanted = {
    "max_steps_per_arm": 199,
    "checkpoint_save_steps": 100,
    "seed": 42,
    "nproc_per_node": 1,
    "same_row_ids_and_order": True,
}
for key, value in wanted.items():
    if budget.get(key) != value:
        raise SystemExit(f"budget mismatch for {key}: {budget.get(key)!r} != {value!r}")
for arm in plan["arms"]:
    name = arm["name"]
    command = arm["command"]
    for flag, value in (("--max_steps", "199"), ("--save_steps", "100"), ("--seed", "42")):
        try:
            actual_value = command[command.index(flag) + 1]
        except (ValueError, IndexError):
            raise SystemExit(f"{name}: missing {flag}")
        if actual_value != value:
            raise SystemExit(f"{name}: {flag}={actual_value!r}, expected {value!r}")
    if "--resume_from_checkpoint" not in command:
        raise SystemExit(f"{name}: missing --resume_from_checkpoint")
print("ALL6_PLAN_AUDIT=PASS")
' "$PLAN"

if [[ "$PLAN_ONLY" == true ]]; then
  printf 'PLAN_ONLY=true\nplan=%s\n' "$PLAN"
  exit 0
fi

if ! command -v setsid >/dev/null 2>&1; then
  printf 'setsid is required for terminal-independent launch but was not found.\n' >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
LAUNCHER_LOG="$OUTPUT_ROOT/launcher.log"
PID_FILE="$OUTPUT_ROOT/launcher.pid"

# setsid creates an independent session/process group; nohup ignores terminal
# hangup, and all arm processes inherit that group.  This intentionally does
# not inspect or refuse GPU occupancy.
nohup setsid "$PYTHON_BIN" mechanism_experiments/causal_ablation/launch_parallel.py \
  --plan "$PLAN" \
  --gpus "$GPUS" \
  --max-parallel "$MAX_PARALLEL" \
  --log-dir "$OUTPUT_ROOT/logs" \
  --execute \
  >"$LAUNCHER_LOG" 2>&1 </dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" >"$PID_FILE"

sleep 1
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  printf 'Detached launcher exited during startup; inspect %s\n' "$LAUNCHER_LOG" >&2
  tail -n 40 "$LAUNCHER_LOG" >&2 || true
  exit 1
fi

launcher_sid="$(ps -o sid= -p "$launcher_pid" | tr -d '[:space:]')"
if [[ "$launcher_sid" != "$launcher_pid" ]]; then
  printf 'Detached launcher is not its own session leader (pid=%s sid=%s); stopping it.\n' \
    "$launcher_pid" "$launcher_sid" >&2
  kill -TERM "$launcher_pid" 2>/dev/null || true
  exit 1
fi

disown "$launcher_pid" 2>/dev/null || true
printf '%s\n' \
  'ALL6_LAUNCH=STARTED' \
  "pid=$launcher_pid" \
  "output_root=$OUTPUT_ROOT" \
  "launcher_log=$LAUNCHER_LOG" \
  "arm_logs=$OUTPUT_ROOT/logs" \
  "stop=bash mechanism_experiments/causal_ablation/stop_all6.sh --output-root $OUTPUT_ROOT"
