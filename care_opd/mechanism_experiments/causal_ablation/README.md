# CARE-OPD single-round causal ablations

This directory closes a basic mechanism experiment loop without changing
`care_opd.train_opd`, the QZBC parser, or the QZBC verifier. Scripts are
**plan-only by default**; training/evaluation starts only with `--execute` (or
by manually running the generated `commands.sh`).

## What is compared

All formal arms start from one identical checkpoint and use identical row IDs,
row order, seed, epoch cap, and explicit optimizer-step cap:

| Arm | Selected KD region | Retention | Risk | Trust |
|---|---|---:|---:|---:|
| `full_selected` | verifier-selected causal span | on | on | on |
| `without_retention` | verifier-selected causal span | off | on | on |
| `matched_full_response` | full response | off | on | on |
| `outside_only` | complement of causal span | off | on | on |
| `without_risk` | verifier-selected causal span | on | off | on |
| `without_trust` | verifier-selected causal span | on | on | off |

`without_retention`, `matched_full_response`, and `outside_only` form the clean
localisation contrast: retention is disabled in all three. “Equal budget” here
means equal examples and optimizer updates. It does **not** mean equal selected
tokens: full-response/outside masks intentionally change token mass.

`build_views.py` also emits a character-offset `random_equal_length.jsonl` for
diagnostics, but it is deliberately excluded from the formal matrix. Unicode
character equality is not tokenizer-token equality, and many QZBC spans leave
too little outside region for a valid outside-only equal-length control. It
must not be described or trained as a token-matched control. A future formal
random arm should sample response token positions after tokenizer offset
projection and publish an exact-token-count audit.

## 1. Build causal views

```bash
cd /opt/care-workspace/care_opd

PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
ROUND=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/round_01
VIEWS=/opt/care-workspace/care_opd/runs/mechanism_ablation/views

"$PY" mechanism_experiments/causal_ablation/build_views.py causal \
  --input "$ROUND/rollouts/qzbc_opd.jsonl" \
  --output-dir "$VIEWS" \
  --seed 42
```

The manifest binds source/view SHA-256, row count, ID order, and degeneracies.
No held-out outcome is read when constructing a view.

## 2. Generate the equal-budget one-round training plan

```bash
STUDENT=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged
TEACHER=Qwen/Qwen3.5-9B
# This matches the completed formal run. If the server cache is unavailable,
# replace it with the local completed 9B checkpoint path before freezing the plan.
RUN=/opt/care-workspace/care_opd/runs/mechanism_ablation/single_round_all6_v2

# The retention-enabled arms require a frozen reference under LoRA training.
# Using the common initialization makes this explicit and keeps every arm
# anchored to the same Clean-SFT policy.
REFERENCE="$STUDENT"

"$PY" mechanism_experiments/causal_ablation/run_single_round.py \
  --matrix mechanism_experiments/causal_ablation/ablation_matrix.json \
  --views-dir "$VIEWS" \
  --train-config configs/qzbc_formal.yaml \
  --student-model "$STUDENT" \
  --teacher-model "$TEACHER" \
  --reference-model "$REFERENCE" \
  --output-root "$RUN" \
  --max-steps 199 \
  --save-steps 100 \
  --seed 42 \
  --nproc-per-node 1 \
  --python "$PY" \
  --resume
```

Inspect `experiment_plan.json` and `commands.sh`. Add `--execute` only after
the plan is frozen. `baseline_mode=care_opd` is intentional: named core
baseline bundles overwrite switches. Core metadata therefore says
`method=care_opd`; the unique arm directory and hash-bearing experiment plan
are the authoritative ablation labels.

`199 = ceil(about 6,350 OPD rows / gradient_accumulation_steps 32)` is the
one-epoch update cap for the current round-1 bank. Recompute it from the view
manifest if row count changes; a larger `--max-steps` does not create a second
epoch because this experiment fixes `num_train_epochs=1`.

`--save-steps 100` is applied identically to all arms. It leaves one recovery
checkpoint before the 199-step endpoint without multiplying storage use across
six large arms. `--resume` is embedded in every arm command; it is a no-op for
a fresh root and permits an explicitly filtered interrupted arm to resume from
its checkpoint. Neither setting changes the 199-update training budget.

### Recommended: launch all six arms detached

The wrapper below freezes and audits exactly the six arms in
`ablation_matrix.json`, uses seed 42 / 199 updates / 100-step checkpoints for
every arm, schedules four one-GPU jobs on physical GPUs 4,5,6,7, and starts
the remaining two when slots finish. It refuses an existing output root so an
interrupted run cannot be silently mixed with a fresh comparison. `nohup` plus
an independent `setsid` session prevents SSH/terminal exit or a later Ctrl-C
in the invoking terminal from propagating to the training group:

```bash
cd /opt/care-workspace/care_opd

bash mechanism_experiments/causal_ablation/launch_all6_detached.sh
```

Defaults write to
`/opt/care-workspace/care_opd/runs/mechanism_ablation/single_round_all6_v2`. Override
only when a new clean directory is needed:

```bash
bash mechanism_experiments/causal_ablation/launch_all6_detached.sh \
  --output-root /opt/care-workspace/care_opd/runs/mechanism_ablation/single_round_all6_v3
```

The wrapper deliberately performs no GPU-occupancy check. Inspect startup and
per-arm logs with:

```bash
RUN=/opt/care-workspace/care_opd/runs/mechanism_ablation/single_round_all6_v2
tail -f "$RUN/launcher.log"
tail -f "$RUN/logs/full_selected.log"
```

To stop only this recorded detached batch, with PID/session/command/plan
validation before signalling its process group:

```bash
bash mechanism_experiments/causal_ablation/stop_all6.sh \
  --output-root /opt/care-workspace/care_opd/runs/mechanism_ablation/single_round_all6_v2
```

SIGTERM is the default. The script does not escalate to SIGKILL unless
`--force` is explicitly supplied after the grace period.

### Manual foreground launcher

With physical GPUs 4,5,6,7, run at most four one-GPU arms concurrently; the
remaining arms start as slots finish. This foreground form is useful for
debugging but remains attached to the terminal:

```bash
"$PY" mechanism_experiments/causal_ablation/launch_parallel.py \
  --plan "$RUN/experiment_plan.json" \
  --gpus 4,5,6,7 --max-parallel 4 \
  --log-dir "$RUN/logs" --execute
```

The same command schedules the two-arm fresh/stale plan concurrently. Build
both plans with `--nproc-per-node 1`; every arm still has the same update and
sample budget.

## 3. Fresh versus stale replay

At round *r*, use the same round-*r* initialization in both arms. The fresh
bank must be collected from that policy; stale is a prior-policy bank.

```bash
FRESH_VIEWS=/opt/care-workspace/care_opd/runs/mechanism_ablation/freshness_views

"$PY" mechanism_experiments/causal_ablation/build_views.py freshness \
  --fresh /opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/round_02/rollouts/qzbc_opd.jsonl \
  --stale /opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/round_01/rollouts/qzbc_opd.jsonl \
  --output-dir "$FRESH_VIEWS" \
  --seed 42 \
  --min-rows 100

FRESH_ROWS="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$FRESH_VIEWS/manifest.json")"
FRESH_STEPS="$(( (FRESH_ROWS + 31) / 32 ))"

"$PY" mechanism_experiments/causal_ablation/run_single_round.py \
  --matrix mechanism_experiments/causal_ablation/freshness_matrix.json \
  --views-dir "$FRESH_VIEWS" \
  --train-config configs/qzbc_formal.yaml \
  --student-model /opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/round_01/merged_model \
  --teacher-model "$TEACHER" \
  --reference-model /opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/round_01/merged_model \
  --output-root /opt/care-workspace/care_opd/runs/mechanism_ablation/fresh_vs_stale \
  --max-steps "$FRESH_STEPS" --seed 42 --nproc-per-node 1 --python "$PY"

FRESH_RUN=/opt/care-workspace/care_opd/runs/mechanism_ablation/fresh_vs_stale

"$PY" mechanism_experiments/causal_ablation/launch_parallel.py \
  --plan "$FRESH_RUN/experiment_plan.json" \
  --gpus 4,5,6,7 --max-parallel 2 \
  --log-dir "$FRESH_RUN/logs" --execute
```

Matching is a seeded exact-ID intersection and does not use held-out results.
It excludes clean controls and therefore evaluates only the
**persistent-failure cohort** (historically expected to be roughly 259 rows,
or roughly 9 updates at accumulation 32, but the actual manifest/calculated
step cap is authoritative). Round-2 rollouts were generated by the round-1
merged model, so both arms intentionally start from `round_01/merged_model`.
This supports a conditional stale
replay claim, not an all-request freshness effect. The script records hashes
but cannot infer chronology; input labels must be assigned honestly.

## 4. Reuse QZBC rollout evaluation and summarize 3,900 rows

```bash
EVAL=/opt/care-workspace/care_opd/runs/mechanism_ablation/eval
REQUESTS=/opt/care-workspace/care_opd/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl
export CUDA_VISIBLE_DEVICES=4,5,6,7

PYTHONPATH=/opt/care-workspace/care_opd/src \
"$PY" mechanism_experiments/causal_ablation/evaluate_arms.py \
  --experiment-plan "$RUN/experiment_plan.json" \
  --requests "$REQUESTS" \
  --output-root "$EVAL" \
  --num-workers 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --python "$PY" \
  --execute
```

The evaluator runs the six arms sequentially; each arm shards the 4,200
requests across physical GPUs 4,5,6,7. Omit `--execute` when you only want to
freeze and inspect `evaluation_plan.json` / `commands.sh`. After all arms have
finished, summarize. `DATA_AUDIT.json` is accepted directly as the ID source:

```bash
ARGS=()
for arm in full_selected without_retention matched_full_response outside_only without_risk without_trust; do
  ARGS+=(--rollout "$arm=$EVAL/$arm/qzbc_rollouts.jsonl")
done

"$PY" mechanism_experiments/causal_ablation/summarize_arms.py \
  --requests "$REQUESTS" \
  --family-clean-ids /opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/DATA_AUDIT.json \
  "${ARGS[@]}" \
  --reference full_selected \
  --output-dir "$EVAL/summary"
```

Evaluate and summarize the fresh-versus-stale plan in the same way:

```bash
FRESH_RUN=/opt/care-workspace/care_opd/runs/mechanism_ablation/fresh_vs_stale
FRESH_EVAL=/opt/care-workspace/care_opd/runs/mechanism_ablation/fresh_vs_stale_eval
export CUDA_VISIBLE_DEVICES=4,5,6,7

PYTHONPATH=/opt/care-workspace/care_opd/src \
"$PY" mechanism_experiments/causal_ablation/evaluate_arms.py \
  --experiment-plan "$FRESH_RUN/experiment_plan.json" \
  --requests "$REQUESTS" \
  --output-root "$FRESH_EVAL" \
  --num-workers 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --python "$PY" \
  --execute

"$PY" mechanism_experiments/causal_ablation/summarize_arms.py \
  --requests "$REQUESTS" \
  --family-clean-ids /opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/DATA_AUDIT.json \
  --rollout "fresh_on_policy=$FRESH_EVAL/fresh_on_policy/qzbc_rollouts.jsonl" \
  --rollout "stale_replay=$FRESH_EVAL/stale_replay/qzbc_rollouts.jsonl" \
  --reference fresh_on_policy \
  --output-dir "$FRESH_EVAL/summary"
```

Outputs are JSON/CSV/Markdown with clean TSR, perturbed TSR, conditional
retention, regression, rescue, and paired seed-cluster bootstrap intervals.
The summarizer imports the existing QZBC paired metric implementation rather
than defining a second verifier. The 3,900 view remains a retrospective
family-clean contamination-sensitivity analysis, not external OOD evidence.

## Basic tests

```bash
"$PY" -m unittest discover \
  -s mechanism_experiments/causal_ablation \
  -p 'test_*.py' -v
```
