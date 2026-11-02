# CARE-OPD multi-training-seed ablations

This package repeats a frozen causal-ablation matrix across independent
training seeds and separates two uncertainty sources that must not be mixed:

- **between-training-seed variability**: mean, sample standard deviation
  (`ddof=1`), and direction-aware worst seed across independently trained
  checkpoints;
- **conditional evaluation-sample uncertainty**: the paired seed-ID-cluster
  bootstrap already stored inside each seed's causal `summary.json`.

The second is not an estimator of the first. With the default three training
seeds, the cross-seed output is descriptive and deliberately reports no
population confidence interval.

All Python entry points are plan-only unless `launch_queue.py --execute` or a
generated command file is explicitly run. Plans use atomic writes and are
frozen: regenerating identical content is allowed, but changed content must use
a new output root.

## Output contract

```text
multiseed_all6_v1/
├── multiseed_plan.json
├── launcher.pid
├── launcher.log
├── queue_status.json
├── logs/seed_{42,43,44}/ARM.log
└── seed_{42,43,44}/train/
    ├── experiment_plan.json
    ├── commands.sh
    └── ARM/final/...

multiseed_all6_v1_eval/
├── multiseed_evaluation_plan.json
├── commands.sh
├── evaluation.log
├── seed_{42,43,44}/
│   ├── evaluation_plan.json
│   ├── commands.sh
│   ├── rollouts/ARM/qzbc_rollouts.jsonl
│   └── summary/{summary.json,summary.csv,summary.md}
└── aggregate/
    ├── multiseed_summary.json
    ├── aggregate.csv
    ├── per_seed.csv
    └── multiseed_summary.md
```

## 1. Freeze and launch the default 42/43/44 training queue

The following is the complete server command. It creates 18 jobs for the
current six-arm matrix, runs at most four jobs concurrently on physical GPUs
4–7, and survives terminal exit. It does not use `nvidia-smi` as a lock; check
that the selected GPUs are available first.

```bash
cd /opt/care-workspace/care_opd

export PROJECT_ROOT=/opt/care-workspace/care_opd
export PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
export VIEWS=/opt/care-workspace/care_opd/runs/mechanism_ablation/views
export STUDENT=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged
export TEACHER=Qwen/Qwen3.5-9B
export REFERENCE="$STUDENT"
export SEEDS=42,43,44
export GPUS=4,5,6,7
export MAX_PARALLEL=4
export MAX_STEPS=199
export SAVE_STEPS=100
export OUTPUT_ROOT=/opt/care-workspace/care_opd/runs/mechanism_ablation/multiseed_all6_v1

bash mechanism_experiments/multiseed_ablation/launch_multiseed_detached.sh
```

The wrapper first freezes every per-seed plan and the suite plan. It then uses
`nohup setsid` to launch the GPU queue. `--skip-completed` skips an arm only
when `final/run_metadata.json` matches both the planned training seed and
`max_steps`, both non-empty adapter files exist, and `metrics.jsonl` contains a
matching `complete` event. A final artifact from another seed or step budget is
rejected as a contaminated output root. Incomplete same-identity arms retain
the existing `--resume_from_checkpoint auto` contract.

For the exact token-location control, set `MATRIX` to
`mechanism_experiments/token_matched_control/token_matched_matrix.json` and set
`PREREQUISITE_AUDIT` to the completed `token_matching_audit.json`. The planner
then automatically uses the token-control contract and binds that audit into
every seed plan; it does not fall back to generic matrix planning.

Monitor without attaching to the process group:

```bash
RUN=/opt/care-workspace/care_opd/runs/mechanism_ablation/multiseed_all6_v1
tail -f "$RUN/launcher.log"
cat "$RUN/queue_status.json"
tail -f "$RUN/logs/seed_43/full_selected.log"
```

A successful terminal queue has `"terminal": true` and every job state is
`completed` or `skipped_complete`. A zero launcher exit alone is not a substitute
for inspecting that status contract and the final metadata.

### Plan only, without starting training

```bash
cd /opt/care-workspace/care_opd

PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
RUN=/opt/care-workspace/care_opd/runs/mechanism_ablation/multiseed_all6_v1
STUDENT=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged

"$PY" mechanism_experiments/multiseed_ablation/plan_multiseed.py \
  --matrix mechanism_experiments/causal_ablation/ablation_matrix.json \
  --views-dir /opt/care-workspace/care_opd/runs/mechanism_ablation/views \
  --train-config configs/qzbc_formal.yaml \
  --student-model "$STUDENT" \
  --teacher-model Qwen/Qwen3.5-9B \
  --reference-model "$STUDENT" \
  --output-root "$RUN" \
  --max-steps 199 \
  --save-steps 100 \
  --seeds 42,43,44 \
  --python "$PY" \
  --resume

"$PY" mechanism_experiments/multiseed_ablation/launch_queue.py \
  --plan "$RUN/multiseed_plan.json" \
  --gpus 4,5,6,7 \
  --max-parallel 4 \
  --log-dir "$RUN/logs"
```

The last command prints `PLAN_ONLY=True` and starts nothing because
`--execute` is absent.

## 2. Freeze the common evaluation and per-seed summary plan

Run this only after all planned training adapters have completed. Greedy
generation uses one fixed generation seed for all checkpoints; the independent
variable remains the training seed.

```bash
cd /opt/care-workspace/care_opd

PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
RUN=/opt/care-workspace/care_opd/runs/mechanism_ablation/multiseed_all6_v1
EVAL=/opt/care-workspace/care_opd/runs/mechanism_ablation/multiseed_all6_v1_eval
REQUESTS=/opt/care-workspace/care_opd/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl
FAMILY_IDS=/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/DATA_AUDIT.json

"$PY" mechanism_experiments/multiseed_ablation/plan_evaluation.py \
  --training-plan "$RUN/multiseed_plan.json" \
  --requests "$REQUESTS" \
  --family-clean-ids "$FAMILY_IDS" \
  --output-root "$EVAL" \
  --python "$PY" \
  --num-workers 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --dtype bfloat16 \
  --max-new-tokens 1200 \
  --generation-seed 42 \
  --reference full_selected
```

Inspect `$EVAL/multiseed_evaluation_plan.json` and `$EVAL/commands.sh`. The
global command file evaluates one checkpoint at a time, with each checkpoint
sharded over the same four GPUs, then immediately writes that training seed's
causal `summary.json`. Launch the frozen evaluation sequence detached:

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup setsid bash "$EVAL/commands.sh" \
  > "$EVAL/evaluation.log" 2>&1 < /dev/null &
EVAL_PID=$!
printf '%s\n' "$EVAL_PID" > "$EVAL/evaluation.pid"
echo "evaluation_pid=$EVAL_PID"
tail -f "$EVAL/evaluation.log"
```

Do not run multiple copies of this sequence on the same output root. The
per-seed rollout and summary commands fail closed on missing or malformed
inputs; the cross-seed aggregator below also requires every planned seed.

## 3. Aggregate across independently trained checkpoints

```bash
cd /opt/care-workspace/care_opd

PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
EVAL=/opt/care-workspace/care_opd/runs/mechanism_ablation/multiseed_all6_v1_eval

"$PY" mechanism_experiments/multiseed_ablation/summarize_multiseed.py \
  --evaluation-plan "$EVAL/multiseed_evaluation_plan.json" \
  --output-dir "$EVAL/aggregate"
```

`multiseed_summary.json` is the authoritative machine-readable artifact.
`aggregate.csv` is long-form mean/SD/min/max/worst-seed output; `per_seed.csv`
retains every source value; `multiseed_summary.md` is the paper-facing compact
view. Higher-is-better metrics use the minimum as the worst seed, while
regression metrics use the maximum.

## Claim boundary

This package supports a statement about stability over the exact training
seeds, initialization, data, matrix, optimizer-step budget, and evaluation
protocol bound into the plans. Three seeds do not justify a normality claim or
a precise population confidence interval. A request-level bootstrap CI cannot
be used as evidence that retraining the method is stable.

## Basic development tests

These tests create only temporary tiny plans and fake summaries. They do not
load model weights or start training/inference.

```bash
cd /opt/care-workspace/care_opd
PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python

"$PY" -m unittest discover \
  -s mechanism_experiments/multiseed_ablation \
  -p 'test_*.py' -v
```
