#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CARE_ROOT="${CARE_ROOT:-/opt/care-workspace/care_opd}"
QZBC_ROOT="${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}"
PY="${PY:-/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python}"
TROPD_MODEL="${TROPD_MODEL:-$CARE_ROOT/runs/qzbc_baselines_3gpu/baselines/tropd_full/final_model}"
CARE_MODEL="${CARE_MODEL:-$CARE_ROOT/runs/qzbc_base_model_3gpu/final_model}"
BACKFILL_ROOT="${BACKFILL_ROOT:-/opt/care-data/care_opd_tropd_backfill}"
PHYSICAL_GPU="${PHYSICAL_GPU:-6}"
LOGICAL_DEVICE="${LOGICAL_DEVICE:-cuda:0}"
RUN_MODE="${RUN_MODE:-fresh}"
DRY_RUN="${DRY_RUN:-0}"
QZBC_CHECKPOINT="${QZBC_CHECKPOINT:-0}"

QZBC_DATA_ROOT="${QZBC_DATA_ROOT:-$CARE_ROOT/data/qzbc_formal}"
QZBC_REQUESTS="${QZBC_REQUESTS:-$QZBC_DATA_ROOT/compiled/test/qzbc_on_policy_requests.jsonl}"
QZBC_SEED_MANIFEST="${QZBC_SEED_MANIFEST:-$QZBC_DATA_ROOT/seeds/qzbc_native_seeds.manifest.json}"
QZBC_TRAIN_MANIFEST="${QZBC_TRAIN_MANIFEST:-$QZBC_DATA_ROOT/compiled/train/manifest.json}"
QZBC_DEV_MANIFEST="${QZBC_DEV_MANIFEST:-$QZBC_DATA_ROOT/compiled/dev/manifest.json}"
QZBC_TEST_MANIFEST="${QZBC_TEST_MANIFEST:-$QZBC_DATA_ROOT/compiled/test/manifest.json}"
QZBC_OUT_ROOT="${QZBC_OUT_ROOT:-$BACKFILL_ROOT/qzbc_grouped_tropd_vs_care_v1}"

STRUCTURED_BENCH="${STRUCTURED_BENCH:-$CARE_ROOT/runs/private_real_thyroid_external/benchmark_v5_structured}"
RAW_BENCH="${RAW_BENCH:-$CARE_ROOT/runs/private_real_thyroid_external/benchmark_v5_raw}"
STRUCTURED_OUT="${STRUCTURED_OUT:-$BACKFILL_ROOT/real_thyroid_v5_structured/tropd_vs_care_v1}"
RAW_OUT="${RAW_OUT:-$BACKFILL_ROOT/real_thyroid_v5_raw/tropd_vs_care_v1}"
REAL_EVAL="$CARE_ROOT/mechanism_experiments/real_thyroid_external/thyroid_external_eval.py"
GUARD="$SCRIPT_DIR/protocol_guard.py"
MODEL_LOCK="$BACKFILL_ROOT/MODEL_LOCK.json"

export PYTHONPATH="$CARE_ROOT/src:$CARE_ROOT/public_eval:$CARE_ROOT/public_eval/qzbc_grouped${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

usage() {
  cat <<'EOF'
Usage: bash run_tropd_backfill.sh {preflight|qzbc|structured|raw|all}

Environment:
  PHYSICAL_GPU=6       one physical GPU; no occupancy check is performed
  BACKFILL_ROOT=...    must be a new, isolated output root
  RUN_MODE=fresh       fresh (default) or resume for an interrupted real track
  QZBC_CHECKPOINT=1    durable QZBC row checkpoints; repeat same root to resume
  DRY_RUN=1            print the exact inference commands without touching assets

The formal run always evaluates TROP-D and a fresh CARE-OPD reference under the
same protocol. It performs inference only; it never trains or downloads models.
EOF
}

die() {
  printf 'TROPD_BACKFILL_ERROR: %s\n' "$*" >&2
  exit 2
}

require_file() {
  [[ -s "$1" ]] || die "missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing required directory: $1"
}

require_model() {
  local root="$1"
  [[ ! -e "$root/adapter_config.json" ]] || \
    die "adapter-only/PEFT checkpoint is not a merged causal-LM checkpoint: $root"
  require_file "$root/config.json"
  require_file "$root/tokenizer_config.json"
  find "$root" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) \
    -print -quit | grep -q . || die "model has no local weights: $root"
}

print_dry_run() {
  local phase="$1"
  printf 'DRY_RUN=PASS phase=%s device=%s physical_gpu=%s\n' \
    "$phase" "$LOGICAL_DEVICE" "$PHYSICAL_GPU"
  if [[ "$phase" == qzbc || "$phase" == all ]]; then
    for label in tropd_full care_opd; do
      local model="$TROPD_MODEL"
      [[ "$label" == care_opd ]] && model="$CARE_MODEL"
      printf 'QZBC_CMD env CUDA_VISIBLE_DEVICES=%q %q -u -m care_opd.qzbc_rollout --requests %q --output-dir %q --model %q --num-workers 1 --devices %q --dtype bfloat16 --max-new-tokens 1200 --greedy --seed 42 --max-examples -1 --max-format-failure-fraction 0.25' \
        "$PHYSICAL_GPU" "$PY" "$QZBC_REQUESTS" "$QZBC_OUT_ROOT/formal/$label" "$model" "$LOGICAL_DEVICE"
      if [[ "$QZBC_CHECKPOINT" == 1 ]]; then
        printf ' --checkpoint-contract %q' "$QZBC_OUT_ROOT/protocol_lock.json"
      fi
      printf '\n'
    done
  fi
  for track in structured raw; do
    if [[ "$phase" == "$track" || "$phase" == all ]]; then
      local bench="$STRUCTURED_BENCH"
      local out="$STRUCTURED_OUT"
      [[ "$track" == raw ]] && bench="$RAW_BENCH" && out="$RAW_OUT"
      printf 'REAL_CMD[%s] env CUDA_VISIBLE_DEVICES=%q %q -u %q --seeds %q --field-gold %q --qzbc-root %q --model tropd_full=%q --model care_opd=%q --reference-model care_opd --output-dir %q --device %q --max-input-tokens 32768 --max-new-tokens 2048 --generation-seed 42 --bootstrap-replicates 10000 --bootstrap-seed 20260819 --acknowledge-private-clinical-data --overwrite\n' \
        "$track" "$PHYSICAL_GPU" "$PY" "$REAL_EVAL" \
        "$bench/locked_external_qzbc.private.jsonl" \
        "$bench/field_gold.private.jsonl" "$QZBC_ROOT" \
        "$TROPD_MODEL" "$CARE_MODEL" "$out" "$LOGICAL_DEVICE"
    fi
  done
}

prepare_qzbc_lock() {
  mkdir -p "$QZBC_OUT_ROOT"
  local checkpoint_args=()
  [[ "$QZBC_CHECKPOINT" == 0 ]] || checkpoint_args=(--durable-checkpoint)
  env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$PY" -u "$GUARD" prepare-qzbc \
    --project-root "$CARE_ROOT" \
    --qzbc-root "$QZBC_ROOT" \
    --requests "$QZBC_REQUESTS" \
    --seed-manifest "$QZBC_SEED_MANIFEST" \
    --compiled-train-manifest "$QZBC_TRAIN_MANIFEST" \
    --compiled-dev-manifest "$QZBC_DEV_MANIFEST" \
    --compiled-test-manifest "$QZBC_TEST_MANIFEST" \
    --model-lock "$MODEL_LOCK" \
    --physical-gpu "$PHYSICAL_GPU" \
    --logical-device "$LOGICAL_DEVICE" \
    --output-dir "$QZBC_OUT_ROOT" "${checkpoint_args[@]}"
}

prepare_model_lock() {
  mkdir -p "$BACKFILL_ROOT"
  "$PY" -u "$GUARD" prepare-model-lock \
    --project-root "$CARE_ROOT" \
    --tropd-model "$TROPD_MODEL" \
    --care-model "$CARE_MODEL" \
    --output "$MODEL_LOCK"
}

check_benchmark() {
  local track="$1"
  local bench="$2"
  require_dir "$bench"
  for file in locked_external_qzbc.private.jsonl field_gold.private.jsonl benchmark_manifest.json SHA256SUMS.private; do
    require_file "$bench/$file"
  done
  (cd "$bench" && sha256sum -c SHA256SUMS.private)
  "$PY" -u "$GUARD" inspect-real-inputs \
    --track "$track" \
    --project-root "$CARE_ROOT" \
    --qzbc-root "$QZBC_ROOT" \
    --seeds "$bench/locked_external_qzbc.private.jsonl" \
    --field-gold "$bench/field_gold.private.jsonl"
}

preflight_common() {
  [[ "$QZBC_CHECKPOINT" == 0 || "$QZBC_CHECKPOINT" == 1 ]] || \
    die "QZBC_CHECKPOINT must be 0 or 1"
  [[ "$RUN_MODE" == fresh || "$RUN_MODE" == resume ]] || \
    die "RUN_MODE must be fresh or resume"
  [[ "$PHYSICAL_GPU" =~ ^[0-9]+$ ]] || die "PHYSICAL_GPU must be one numeric index"
  [[ "$LOGICAL_DEVICE" == cuda:0 ]] || \
    die "formal GPU protocol requires LOGICAL_DEVICE=cuda:0"
  require_file "$PY"
  require_file "$GUARD"
  require_file "$REAL_EVAL"
  require_dir "$QZBC_ROOT"
  require_model "$TROPD_MODEL"
  require_model "$CARE_MODEL"
  env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$PY" - "$TROPD_MODEL" "$CARE_MODEL" <<'PY'
import sys
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("CUDA mapping check failed: expected exactly one visible CUDA device")
for model in sys.argv[1:]:
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if type(config) not in AutoModelForCausalLM._model_mapping:
        raise SystemExit(f"unsupported causal-LM architecture: {model} ({type(config)})")
print(
    "INFERENCE_STACK=PASS",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"device={torch.cuda.get_device_name(0)}",
)
PY
  "$PY" "$REAL_EVAL" --help >/dev/null
  prepare_model_lock
}

preflight_qzbc() {
  for file in "$QZBC_REQUESTS" "$QZBC_SEED_MANIFEST" "$QZBC_TRAIN_MANIFEST" "$QZBC_DEV_MANIFEST" "$QZBC_TEST_MANIFEST"; do
    require_file "$file"
  done
  prepare_qzbc_lock
}

preflight() {
  preflight_common
  preflight_qzbc
  check_benchmark structured "$STRUCTURED_BENCH"
  check_benchmark raw "$RAW_BENCH"
  printf 'TROPD_BACKFILL_PREFLIGHT=PASS gpu=%s root=%s\n' "$PHYSICAL_GPU" "$BACKFILL_ROOT"
}

run_qzbc_model() {
  local label="$1"
  local model="$2"
  local out="$QZBC_OUT_ROOT/formal/$label"
  local lock="$QZBC_OUT_ROOT/protocol_lock.json"
  if [[ -s "$out/COMPLETE.json" ]]; then
    "$PY" -u "$GUARD" verify-qzbc-model \
      --lock "$lock" --requests "$QZBC_REQUESTS" \
      --label "$label" --output-dir "$out"
    printf 'QZBC_MODEL_SKIP_VALIDATED label=%s\n' "$label"
    return
  fi
  if [[ -e "$out" ]]; then
    [[ "$QZBC_CHECKPOINT" == 1 && -d "$out" && -s "$out/checkpoint_manifest.json" ]] || \
      die "partial QZBC output has no enabled durable checkpoint manifest: $out; preserve it and choose a new QZBC_OUT_ROOT"
  fi
  # The checkpoint collector validates all existing manifests/rows and refuses
  # legacy or malformed partials before generating any further outputs.
  local checkpoint_args=()
  [[ "$QZBC_CHECKPOINT" == 0 ]] || checkpoint_args=(--checkpoint-contract "$lock")
  mkdir -p "$out"
  env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    "$PY" -u -m care_opd.qzbc_rollout \
      --requests "$QZBC_REQUESTS" \
      --output-dir "$out" \
      --model "$model" \
      --num-workers 1 \
      --devices "$LOGICAL_DEVICE" \
      --dtype bfloat16 \
      --max-new-tokens 1200 \
      --greedy \
      --seed 42 \
      --max-examples -1 \
      --max-format-failure-fraction 0.25 "${checkpoint_args[@]}"
  "$PY" -u "$GUARD" finalize-qzbc-model \
    --lock "$lock" --requests "$QZBC_REQUESTS" \
    --label "$label" --output-dir "$out"
}

run_qzbc() {
  run_qzbc_model tropd_full "$TROPD_MODEL"
  run_qzbc_model care_opd "$CARE_MODEL"
  "$PY" -u "$GUARD" summarize-qzbc \
    --project-root "$CARE_ROOT" \
    --lock "$QZBC_OUT_ROOT/protocol_lock.json" \
    --data-audit "$QZBC_OUT_ROOT/DATA_AUDIT.json" \
    --requests "$QZBC_REQUESTS" \
    --eval-root "$QZBC_OUT_ROOT" \
    --output-dir "$QZBC_OUT_ROOT/formal"
}

run_real_track() {
  local track="$1"
  local bench="$2"
  local out="$3"
  local summary="$out/public/summary.json"
  if [[ -s "$summary" ]]; then
    "$PY" -u "$GUARD" verify-real \
      --track "$track" --output-dir "$out" \
      --seeds "$bench/locked_external_qzbc.private.jsonl" \
      --field-gold "$bench/field_gold.private.jsonl" \
      --model-lock "$MODEL_LOCK"
    printf 'REAL_TRACK_SKIP_VALIDATED track=%s\n' "$track"
    return
  fi
  local mode_arg="--overwrite"
  if [[ "$RUN_MODE" == fresh ]]; then
    [[ ! -e "$out" ]] || die "real-track output already exists: $out; use RUN_MODE=resume only for an identical interrupted v5 run"
  else
    [[ -d "$out" ]] || die "RUN_MODE=resume requires an existing partial output: $out"
    mode_arg="--resume"
  fi
  mkdir -p "$out"
  local tee_args=("$out/driver.log")
  [[ "$mode_arg" == --resume ]] && tee_args=(-a "$out/driver.log")
  env CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    "$PY" -u "$REAL_EVAL" \
      --seeds "$bench/locked_external_qzbc.private.jsonl" \
      --field-gold "$bench/field_gold.private.jsonl" \
      --qzbc-root "$QZBC_ROOT" \
      --model "tropd_full=$TROPD_MODEL" \
      --model "care_opd=$CARE_MODEL" \
      --reference-model care_opd \
      --output-dir "$out" \
      --device "$LOGICAL_DEVICE" \
      --max-input-tokens 32768 \
      --max-new-tokens 2048 \
      --generation-seed 42 \
      --bootstrap-replicates 10000 \
      --bootstrap-seed 20260819 \
      --acknowledge-private-clinical-data \
      "$mode_arg" 2>&1 | tee "${tee_args[@]}"
  "$PY" -u "$GUARD" verify-real \
    --track "$track" --output-dir "$out" \
    --seeds "$bench/locked_external_qzbc.private.jsonl" \
    --field-gold "$bench/field_gold.private.jsonl" \
    --model-lock "$MODEL_LOCK"
}

phase="${1:-}"
case "$phase" in
  preflight|qzbc|structured|raw|all) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "$DRY_RUN" == 1 ]]; then
  print_dry_run "$phase"
  exit 0
fi

mkdir -p "$BACKFILL_ROOT"
command -v flock >/dev/null 2>&1 || die "flock is required for the exclusive backfill-root lock"
exec 9>"$BACKFILL_ROOT/.tropd_backfill.lock"
flock -n 9 || die "another TROP-D backfill process already owns $BACKFILL_ROOT"

case "$phase" in
  preflight)
    preflight
    ;;
  qzbc)
    preflight_common
    preflight_qzbc
    run_qzbc
    ;;
  structured)
    preflight_common
    check_benchmark structured "$STRUCTURED_BENCH"
    run_real_track structured "$STRUCTURED_BENCH" "$STRUCTURED_OUT"
    ;;
  raw)
    preflight_common
    check_benchmark raw "$RAW_BENCH"
    run_real_track raw "$RAW_BENCH" "$RAW_OUT"
    ;;
  all)
    preflight_common
    preflight_qzbc
    check_benchmark structured "$STRUCTURED_BENCH"
    check_benchmark raw "$RAW_BENCH"
    run_qzbc
    run_real_track structured "$STRUCTURED_BENCH" "$STRUCTURED_OUT"
    run_real_track raw "$RAW_BENCH" "$RAW_OUT"
    ;;
esac

printf 'TROPD_BACKFILL_PHASE=COMPLETE phase=%s root=%s\n' "$phase" "$BACKFILL_ROOT"
