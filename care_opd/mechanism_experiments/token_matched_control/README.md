# Exact token-matched mask control

This experiment separates the **semantic location** of CARE-OPD's verifier mask
from the mere amount of supervised text.  The random arm is constructed after
tokenization and preserves, for every training row:

- the exact selected response-token count;
- the exact multiset of selected-token loss weights;
- the same data rows, row order, optimizer updates, model initialization, and
  random training seed.

Only the selected token positions change.  EOS and other zero-width special
tokens are excluded because character spans cannot select them.  Both formal
arms disable teacher-trust filtering, retention KL, and hard-oracle CE, so the
exact pre-trust match is also an exact effective-KD-token match.  Both retain
the same risk-weight multiset.

## 1. Audit the mask contract before training

```bash
cd /opt/care-workspace/care_opd
export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
bash mechanism_experiments/token_matched_control/run_audit.sh
```

The run must report `TOKEN_MATCHING_AUDIT=COMPLETE`; the JSON must show both
`exact_count_match_rate=1.0` and `exact_weight_multiset_match_rate=1.0`.
The audit is bound to the exact training JSONL, tokenizer fingerprint, mask
implementation, and the frozen 4096/1200/EOS tokenization settings. When the
student is a local checkpoint, planning recomputes its tokenizer fingerprint;
remote model IDs are bound by their exact ID plus the audit fingerprint without
triggering a network fetch.

## 2. Plan one matched training seed

```bash
cd /opt/care-workspace/care_opd
PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
RUN=/opt/care-workspace/care_opd/runs/mechanism_experiments/token_matched_control_seed42_v1
AUDIT=/opt/care-workspace/care_opd/runs/mechanism_experiments/token_matched_control_audit_v1/token_matching_audit.json

test ! -e "$RUN" || { echo "REFUSE stale root: $RUN" >&2; exit 2; }

"$PY" mechanism_experiments/token_matched_control/plan_experiment.py \
  --token-audit "$AUDIT" \
  --views-dir runs/mechanism_ablation/views \
  --train-config configs/qzbc_formal.yaml \
  --student-model runs/qzbc_base_model_3gpu/warm_start/merged \
  --teacher-model Qwen/Qwen3.5-9B \
  --reference-model runs/qzbc_base_model_3gpu/warm_start/merged \
  --output-root "$RUN" \
  --max-steps 199 --save-steps 100 --seed 42 \
  --nproc-per-node 1 --python "$PY" --resume
```

Expected sentinels are `PLAN_ONLY=True` and
`TOKEN_MATCHED_CONTRACT=PASS`. Training has not started at this point. Launch
the resulting `experiment_plan.json` with the common causal-ablation launcher
only after inspecting the plan.

Use the multiseed experiment package for the formal 42/43/44 run and unified
evaluation.  The formal contrast is
`localized_kd_only - random_token_kd_only`.  Do not compare training losses;
compare frozen family-clean rollouts, perturbed TSR, conditional retention,
rescue/regression, and their paired intervals.  Training-seed dispersion must
be reported separately from request/seed-cluster bootstrap uncertainty.
