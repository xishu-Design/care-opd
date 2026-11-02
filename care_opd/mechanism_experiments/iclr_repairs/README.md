# ICLR repair experiments: complete server commands

This package closes two high-priority attribution gaps with three independent
training seeds (`42,43,44`):

1. hard-oracle CE attribution, including matched CARE/GKD oracle controls;
2. semantic localization versus an exact tokenizer-level random-token control.

All planners are plan-only. Training starts only at the explicit
`launch_queue.py --execute` command. Evaluation is frozen only after all final
adapters are complete. Do not run the two evaluation queues simultaneously on
the same four GPUs.

## 0. Common paths and minimal development check

Run these definitions at the start of every new shell:

```bash
cd /opt/care-workspace/care_opd
set -euo pipefail

ROOT=/opt/care-workspace/care_opd
PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
VIEWS="$ROOT/runs/mechanism_ablation/views"
STUDENT="$ROOT/runs/qzbc_base_model_3gpu/warm_start/merged"
TEACHER=Qwen/Qwen3.5-9B
REFERENCE="$STUDENT"
REQUESTS="$ROOT/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl"
FAMILY_IDS="$ROOT/runs/public_eval/qzbc_grouped_identity_4gpu_v1/DATA_AUDIT.json"
SEEDS=42,43,44
MAX_STEPS=199
SAVE_STEPS=100
GPUS=4,5,6,7

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m py_compile \
  mechanism_experiments/iclr_repairs/verify_results.py \
  mechanism_experiments/multiseed_ablation/plan_multiseed.py \
  mechanism_experiments/multiseed_ablation/plan_evaluation.py \
  mechanism_experiments/multiseed_ablation/summarize_multiseed.py

"$PY" -m unittest discover \
  -s mechanism_experiments/iclr_repairs -p 'test_*.py' -v

"$PY" -m pytest tests/test_opd_training.py -q \
  -k 'random_token_matched or oracle_risk'
```

This check is CPU-only and does not load model weights.

## 1. Hard-oracle attribution, three training seeds

Choose a fresh root and freeze all 15 training jobs:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
HO_MATRIX="$ROOT/mechanism_experiments/hard_oracle_attribution/oracle_attribution_matrix.json"
HO_CONTRASTS="$ROOT/mechanism_experiments/multiseed_ablation/hard_oracle_contrasts.json"
HO_RUN="$ROOT/runs/mechanism_ablation/hard_oracle_multiseed_${STAMP}"
HO_EVAL="${HO_RUN}_eval"

echo "HO_RUN=$HO_RUN"
echo "HO_EVAL=$HO_EVAL"

"$PY" mechanism_experiments/multiseed_ablation/plan_multiseed.py \
  --matrix "$HO_MATRIX" \
  --contract hard_oracle \
  --contrasts-file "$HO_CONTRASTS" \
  --views-dir "$VIEWS" \
  --train-config configs/qzbc_formal.yaml \
  --student-model "$STUDENT" \
  --teacher-model "$TEACHER" \
  --reference-model "$REFERENCE" \
  --output-root "$HO_RUN" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --seeds "$SEEDS" \
  --python "$PY" \
  --resume
```

Expected planning sentinels include `PLAN_ONLY=True`. Start the bounded queue
on physical GPUs 4-7:

```bash
mkdir -p "$HO_RUN/logs"

nohup setsid "$PY" mechanism_experiments/multiseed_ablation/launch_queue.py \
  --plan "$HO_RUN/multiseed_plan.json" \
  --gpus "$GPUS" \
  --max-parallel 4 \
  --log-dir "$HO_RUN/logs" \
  --status "$HO_RUN/queue_status.json" \
  --skip-completed \
  --execute \
  > "$HO_RUN/launcher.log" 2>&1 < /dev/null &

echo $! > "$HO_RUN/launcher.pid"
echo "launcher_pid=$(cat "$HO_RUN/launcher.pid")"
tail -f "$HO_RUN/launcher.log"
```

After the queue is terminal, audit the five arms under every seed:

```bash
cat "$HO_RUN/queue_status.json"

for S in 42 43 44; do
  "$PY" mechanism_experiments/hard_oracle_attribution/audit_completed_run.py \
    --experiment-root "$HO_RUN/seed_${S}/train"
done
```

Each iteration must print `HARD_ORACLE_TRAINING_AUDIT=PASS`.

## 2. Hard-oracle common evaluation and cross-seed summary

Freeze one common 4,200-request evaluation protocol. The reference arm is the
no-oracle GKD control; paired hard-oracle contrasts remain left-minus-right
within the same training seed.

```bash
"$PY" mechanism_experiments/multiseed_ablation/plan_evaluation.py \
  --training-plan "$HO_RUN/multiseed_plan.json" \
  --requests "$REQUESTS" \
  --family-clean-ids "$FAMILY_IDS" \
  --output-root "$HO_EVAL" \
  --python "$PY" \
  --num-workers 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --dtype bfloat16 \
  --max-new-tokens 1200 \
  --generation-seed 42 \
  --reference gkd_without_oracle_ce

export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup setsid bash "$HO_EVAL/commands.sh" \
  > "$HO_EVAL/evaluation.log" 2>&1 < /dev/null &

echo $! > "$HO_EVAL/evaluation.pid"
echo "evaluation_pid=$(cat "$HO_EVAL/evaluation.pid")"
tail -f "$HO_EVAL/evaluation.log"
```

After `commands.sh` exits successfully, aggregate training-seed variability:

```bash
"$PY" mechanism_experiments/multiseed_ablation/summarize_multiseed.py \
  --evaluation-plan "$HO_EVAL/multiseed_evaluation_plan.json" \
  --output-dir "$HO_EVAL/aggregate"

"$PY" mechanism_experiments/iclr_repairs/verify_results.py \
  --matrix "$HO_MATRIX" \
  --training-plan "$HO_RUN/multiseed_plan.json" \
  --evaluation-plan "$HO_EVAL/multiseed_evaluation_plan.json" \
  --aggregate "$HO_EVAL/aggregate/multiseed_summary.json" \
  --expected-seeds "$SEEDS" \
  --expected-requests 4200
```

The final command must end with `ICLR_REPAIR_COMPLETION_AUDIT=PASS`.

## 3. Exact token-matched control audit and three training seeds

The audit must use the same `full_selected.jsonl`, tokenizer, truncation limits,
and random-mask seed used by training.

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
TM_MATRIX="$ROOT/mechanism_experiments/token_matched_control/token_matched_matrix.json"
TM_CONTRASTS="$ROOT/mechanism_experiments/multiseed_ablation/token_control_contrasts.json"
TM_AUDIT_ROOT="$ROOT/runs/mechanism_ablation/token_matched_audit_${STAMP}"
TM_AUDIT="$TM_AUDIT_ROOT/token_matching_audit.json"
TM_RUN="$ROOT/runs/mechanism_ablation/token_matched_multiseed_${STAMP}"
TM_EVAL="${TM_RUN}_eval"

DATA="$VIEWS/full_selected.jsonl" \
TOKENIZER="$STUDENT" \
OUTPUT_DIR="$TM_AUDIT_ROOT" \
PYTHON_BIN="$PY" \
bash mechanism_experiments/token_matched_control/run_audit.sh

test -s "$TM_AUDIT"
echo "TM_AUDIT=$TM_AUDIT"
echo "TM_RUN=$TM_RUN"
echo "TM_EVAL=$TM_EVAL"
```

The audit must print `TOKEN_MATCHING_AUDIT=COMPLETE`. Freeze all six training
jobs, with the audit hash-bound into every seed plan:

```bash
"$PY" mechanism_experiments/multiseed_ablation/plan_multiseed.py \
  --matrix "$TM_MATRIX" \
  --contract token_matched \
  --prerequisite-audit "$TM_AUDIT" \
  --contrasts-file "$TM_CONTRASTS" \
  --views-dir "$VIEWS" \
  --train-config configs/qzbc_formal.yaml \
  --student-model "$STUDENT" \
  --teacher-model "$TEACHER" \
  --reference-model "$REFERENCE" \
  --output-root "$TM_RUN" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --seeds "$SEEDS" \
  --python "$PY" \
  --resume

mkdir -p "$TM_RUN/logs"
nohup setsid "$PY" mechanism_experiments/multiseed_ablation/launch_queue.py \
  --plan "$TM_RUN/multiseed_plan.json" \
  --gpus "$GPUS" \
  --max-parallel 4 \
  --log-dir "$TM_RUN/logs" \
  --status "$TM_RUN/queue_status.json" \
  --skip-completed \
  --execute \
  > "$TM_RUN/launcher.log" 2>&1 < /dev/null &

echo $! > "$TM_RUN/launcher.pid"
echo "launcher_pid=$(cat "$TM_RUN/launcher.pid")"
tail -f "$TM_RUN/launcher.log"
```

## 4. Token-control common evaluation and cross-seed summary

Run only after the token-control queue is terminal and every job is complete:

```bash
cat "$TM_RUN/queue_status.json"

"$PY" mechanism_experiments/multiseed_ablation/plan_evaluation.py \
  --training-plan "$TM_RUN/multiseed_plan.json" \
  --requests "$REQUESTS" \
  --family-clean-ids "$FAMILY_IDS" \
  --output-root "$TM_EVAL" \
  --python "$PY" \
  --num-workers 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --dtype bfloat16 \
  --max-new-tokens 1200 \
  --generation-seed 42 \
  --reference random_token_kd_only

export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup setsid bash "$TM_EVAL/commands.sh" \
  > "$TM_EVAL/evaluation.log" 2>&1 < /dev/null &

echo $! > "$TM_EVAL/evaluation.pid"
echo "evaluation_pid=$(cat "$TM_EVAL/evaluation.pid")"
tail -f "$TM_EVAL/evaluation.log"
```

After evaluation finishes:

```bash
"$PY" mechanism_experiments/multiseed_ablation/summarize_multiseed.py \
  --evaluation-plan "$TM_EVAL/multiseed_evaluation_plan.json" \
  --output-dir "$TM_EVAL/aggregate"

"$PY" mechanism_experiments/iclr_repairs/verify_results.py \
  --matrix "$TM_MATRIX" \
  --training-plan "$TM_RUN/multiseed_plan.json" \
  --evaluation-plan "$TM_EVAL/multiseed_evaluation_plan.json" \
  --aggregate "$TM_EVAL/aggregate/multiseed_summary.json" \
  --expected-seeds "$SEEDS" \
  --expected-requests 4200 \
  --token-audit "$TM_AUDIT"
```

Again, the required terminal sentinel is
`ICLR_REPAIR_COMPLETION_AUDIT=PASS`.

## 5. Result files to use

For each suite, use these outputs rather than training loss:

- `EVAL/aggregate/multiseed_summary.json`: authoritative cross-seed result;
- `EVAL/aggregate/contrast_per_seed.csv`: paired delta within each training seed;
- `EVAL/aggregate/contrast_aggregate.csv`: mean, sample SD, range, and worst-seed paired delta;
- `EVAL/aggregate/multiseed_summary.md`: compact paper-facing rendering;
- `EVAL/seed_*/summary/summary.json`: request/seed-cluster bootstrap uncertainty conditional on one checkpoint;
- `RUN/queue_status.json` and `RUN/seed_*/train/*/final/run_metadata.json`: completion and effective-switch evidence;
- token suite only: `TM_AUDIT_ROOT/token_matching_audit.json`.

Compute arm deltas within each training seed before averaging. Do not pool the
three request sets as if they were independent training seeds, and do not use a
request-level bootstrap interval as a substitute for training-seed variance.
