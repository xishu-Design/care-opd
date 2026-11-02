#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/care-workspace/care_opd}"
PY="${PY:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
VIEWS="${VIEWS:-$PROJECT_ROOT/runs/mechanism_ablation/views}"
MATRIX="${MATRIX:-$PROJECT_ROOT/mechanism_experiments/causal_ablation/ablation_matrix.json}"
CONTRASTS_FILE="${CONTRASTS_FILE:-}"
PREREQUISITE_AUDIT="${PREREQUISITE_AUDIT:-}"
STUDENT="${STUDENT:-$PROJECT_ROOT/runs/qzbc_base_model_3gpu/warm_start/merged}"
TEACHER="${TEACHER:-Qwen/Qwen3.5-9B}"
REFERENCE="${REFERENCE:-$STUDENT}"
SEEDS="${SEEDS:-42,43,44}"
GPUS="${GPUS:-4,5,6,7}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
MAX_STEPS="${MAX_STEPS:-199}"
SAVE_STEPS="${SAVE_STEPS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/mechanism_ablation/multiseed_ablation_v1}"

while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --matrix) MATRIX="$2"; shift 2 ;;
    --contrasts-file) CONTRASTS_FILE="$2"; shift 2 ;;
    --prerequisite-audit) PREREQUISITE_AUDIT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"

contrast_args=()
if [[ -n "$CONTRASTS_FILE" ]]; then
  contrast_args+=(--contrasts-file "$CONTRASTS_FILE")
fi
audit_args=()
if [[ -n "$PREREQUISITE_AUDIT" ]]; then
  audit_args+=(--prerequisite-audit "$PREREQUISITE_AUDIT")
fi

"$PY" mechanism_experiments/multiseed_ablation/plan_multiseed.py \
  --matrix "$MATRIX" \
  --views-dir "$VIEWS" \
  --train-config configs/qzbc_formal.yaml \
  --student-model "$STUDENT" \
  --teacher-model "$TEACHER" \
  --reference-model "$REFERENCE" \
  --output-root "$OUTPUT_ROOT" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --seeds "$SEEDS" \
  --python "$PY" \
  --resume \
  "${contrast_args[@]}" \
  "${audit_args[@]}"

PID_FILE="$OUTPUT_ROOT/launcher.pid"
if [[ -s "$PID_FILE" ]]; then
  old_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "refusing to start: recorded launcher is alive (pid=$old_pid)" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_ROOT/logs"
nohup setsid "$PY" mechanism_experiments/multiseed_ablation/launch_queue.py \
  --plan "$OUTPUT_ROOT/multiseed_plan.json" \
  --gpus "$GPUS" \
  --max-parallel "$MAX_PARALLEL" \
  --log-dir "$OUTPUT_ROOT/logs" \
  --status "$OUTPUT_ROOT/queue_status.json" \
  --skip-completed \
  --execute \
  > "$OUTPUT_ROOT/launcher.log" 2>&1 < /dev/null &
launcher_pid=$!
pid_tmp="$PID_FILE.tmp.$$"
printf '%s\n' "$launcher_pid" > "$pid_tmp"
mv "$pid_tmp" "$PID_FILE"

echo "MULTISEED_LAUNCHED=1"
echo "pid=$launcher_pid"
echo "root=$OUTPUT_ROOT"
echo "tail -f $OUTPUT_ROOT/launcher.log"
