# Hard-oracle attribution controls

This package tests whether CARE-OPD's gain is explained by the auxiliary
hard-oracle cross-entropy (CE), rather than by verifier-localized dense
distillation. It reuses the existing `care_opd.train_opd`, single-round plan,
QZBC rollout evaluator, and production verifier. It does not introduce a new
training loop or metric implementation.

Planning is fail-closed and **never starts training**. Actual training begins
only when the already generated plan is passed to the existing parallel
launcher with `--execute`.

## Frozen five-arm design

| Arm | KD | KD region | Risk-weighted KD | Trust | Retention | Oracle CE |
|---|---|---|---:|---:|---:|---:|
| `care_with_oracle_ce` | on | verifier span | yes | yes | 0.20 | 0.25, risk weighted |
| `care_without_oracle_ce` | on | verifier span | yes | yes | 0.20 | 0 |
| `gkd_without_oracle_ce` | on | full response | no | no | 0 | 0 |
| `gkd_plus_same_oracle_ce` | on | full response | no | no | 0 | 0.25, risk weighted |
| `oracle_ce_only_on_policy` | off | N/A | N/A | no | 0 | 0.25, risk weighted |

The first two and middle two arms are exact intervention pairs: each pair
differs only in `hard_oracle_ce_weight`. The GKD+oracle arm keeps ordinary GKD
unweighted while using the same risk-weighted oracle objective as CARE. This
requires `oracle_risk_weighting_mode=enabled`; the legacy global risk switch
cannot express that control because it couples KD and oracle weights.

The fifth arm is a sufficiency control. It consumes the same on-policy rows and
optimizer-update budget but its only nonzero loss coefficient is oracle CE.
All arms deliberately use `baseline_mode=care_opd` so explicit switches are
not overwritten by a named baseline bundle. Arm identity and semantics are
bound by `experiment_plan.json`, not by the legacy `method` string in metrics.

This is a one-round, one-training-seed intervention. Even a clean result does
not measure training-seed variance and is not evidence of cross-domain
generality.

## 0. Basic development verification

```bash
cd /opt/care-workspace/care_opd
PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python

"$PY" -m unittest discover \
  -s mechanism_experiments/hard_oracle_attribution \
  -p 'test_*.py' -v
```

## 1. Generate and audit the plan only

Use the same frozen round-1 view and initialization as the earlier single-round
ablation. The view audit requires every non-clean row to contain a
host-certified oracle; otherwise planning stops.

```bash
cd /opt/care-workspace/care_opd

PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
VIEWS=/opt/care-workspace/care_opd/runs/mechanism_ablation/views
STUDENT=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged
TEACHER=Qwen/Qwen3.5-9B
REFERENCE="$STUDENT"
RUN=/opt/care-workspace/care_opd/runs/mechanism_ablation/hard_oracle_seed42_$(date +%Y%m%d_%H%M%S)

echo "RUN=$RUN"

"$PY" mechanism_experiments/hard_oracle_attribution/plan_experiment.py \
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

echo "Inspect before launch:"
echo "$RUN/experiment_plan.json"
echo "$RUN/commands.sh"
```

Expected terminal sentinels are `PLAN_ONLY=True` and `CONTRACT=PASS`. The plan
contains matrix/config/data hashes, exact effective overrides, the common
row-order hash, and a pre-tokenization oracle-availability audit.

## 2. Start all five arms detached

The following survives terminal/SSH exit. Four arms run on physical GPUs
4-7; the fifth starts when a slot becomes free.

```bash
mkdir -p "$RUN/logs"

nohup setsid "$PY" mechanism_experiments/causal_ablation/launch_parallel.py \
  --plan "$RUN/experiment_plan.json" \
  --gpus 4,5,6,7 \
  --max-parallel 4 \
  --log-dir "$RUN/logs" \
  --execute \
  > "$RUN/launcher.log" 2>&1 < /dev/null &

echo $! > "$RUN/launcher.pid"
echo "launcher_pid=$(cat "$RUN/launcher.pid")"
```

Follow progress without affecting the jobs:

```bash
tail -f "$RUN/launcher.log"
```

Per-arm logs:

```bash
tail -f "$RUN/logs/care_with_oracle_ce.log"
tail -f "$RUN/logs/care_without_oracle_ce.log"
tail -f "$RUN/logs/gkd_without_oracle_ce.log"
tail -f "$RUN/logs/gkd_plus_same_oracle_ce.log"
tail -f "$RUN/logs/oracle_ce_only_on_policy.log"
```

## 3. Prove matched training completion

Do not infer completion from `tail` alone. The auditor checks the frozen
contract, effective final switches, equal final step, non-empty adapter, JSONL
completion sentinel, and actual nonzero oracle token/loss events in every arm
whose oracle coefficient is positive.

```bash
"$PY" mechanism_experiments/hard_oracle_attribution/audit_completed_run.py \
  --experiment-root "$RUN"
```

Expected sentinel: `HARD_ORACLE_TRAINING_AUDIT=PASS`. This proves training
completion only; it is not an effect result.

## 4. Generate and run the common evaluation plan

```bash
REQUESTS=/opt/care-workspace/care_opd/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl
EVAL="${RUN}_eval"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/opt/care-workspace/care_opd/src:/opt/care-workspace/care_opd${PYTHONPATH:+:$PYTHONPATH}

"$PY" mechanism_experiments/causal_ablation/evaluate_arms.py \
  --experiment-plan "$RUN/experiment_plan.json" \
  --requests "$REQUESTS" \
  --output-root "$EVAL" \
  --num-workers 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-new-tokens 1200 \
  --seed 42 \
  --python "$PY"
```

Inspect `$EVAL/evaluation_plan.json` and `$EVAL/commands.sh`, then start the
frozen sequential evaluation detached:

```bash
nohup setsid bash "$EVAL/commands.sh" \
  > "$EVAL/evaluation.log" 2>&1 < /dev/null &

echo $! > "$EVAL/evaluation.pid"
echo "evaluation_pid=$(cat "$EVAL/evaluation.pid")"
tail -f "$EVAL/evaluation.log"
```

## 5. Produce the prespecified paired attribution table

Run this only after all five evaluation directories contain complete
`qzbc_rollouts.jsonl` and `summary.json` files.

```bash
FAMILY_IDS=/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/DATA_AUDIT.json

"$PY" mechanism_experiments/hard_oracle_attribution/summarize_attribution.py \
  --requests "$REQUESTS" \
  --family-clean-ids "$FAMILY_IDS" \
  --rollout "care_with_oracle_ce=$EVAL/care_with_oracle_ce/qzbc_rollouts.jsonl" \
  --rollout "care_without_oracle_ce=$EVAL/care_without_oracle_ce/qzbc_rollouts.jsonl" \
  --rollout "gkd_without_oracle_ce=$EVAL/gkd_without_oracle_ce/qzbc_rollouts.jsonl" \
  --rollout "gkd_plus_same_oracle_ce=$EVAL/gkd_plus_same_oracle_ce/qzbc_rollouts.jsonl" \
  --rollout "oracle_ce_only_on_policy=$EVAL/oracle_ce_only_on_policy/qzbc_rollouts.jsonl" \
  --bootstrap-seed 20260815 \
  --bootstrap-replicates 10000 \
  --output-dir "$EVAL/attribution_summary"
```

Authoritative outputs:

- `attribution_summary/attribution_summary.json`: all prespecified contrasts,
  paired cluster intervals, counts, and claim boundary.
- `attribution_summary/attribution_summary.csv`: paper-table values.
- `attribution_summary/attribution_summary.md`: readable table.
- `attribution_summary/summary.json`: reused full family-clean arm summary.

Interpret the result as follows:

- `care_with_oracle_ce - care_without_oracle_ce`: oracle CE's incremental
  effect inside CARE.
- `gkd_plus_same_oracle_ce - gkd_without_oracle_ce`: the same oracle CE's
  incremental effect inside GKD.
- `care_without_oracle_ce - gkd_without_oracle_ce`: whether the localized
  CARE package retains an advantage without direct oracle CE.
- `care_with_oracle_ce - gkd_plus_same_oracle_ce`: whether the CARE package
  retains an advantage after direct oracle supervision is matched.
- `oracle_ce_only_on_policy`: whether oracle CE alone is sufficient; it must
  not be described as standard SFT because it uses the on-policy failure bank
  and verifier-selected oracle fields.

No result should be reported before the JSON summary exists. Training loss is
not an accuracy metric and cannot substitute for the common rollout/verifier
evaluation.
