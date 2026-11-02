# Margin-to-behavior closure

This is a zero-training, zero-inference mechanism experiment. It joins the
already-completed 200-example recovery-margin assay to the formal QZBC rollout
archives by request ID and asks whether frozen parameter preference is
associated with actual verifier-judged CARE rescue.

## Input contract

The analysis refuses to run unless all of the following hold:

1. `recovery_margin.jsonl` contains exactly one record for each
   `(request ID, clean_sft/gkd_full/care_opd)` pair.
2. Every selected ID exists in all three formal rollout archives.
3. Every selected ID is a failed Clean-SFT rollout. This preserves the
   result-blind failure-conditioned cohort used to select the margin assay.
4. Localized scores are included only when the existing margin diagnostic
   marked the projection `localized_complete`; fallback/partial rows are not
   silently mixed into localized estimates.

Default server inputs:

```text
/opt/care-workspace/care_opd/runs/mechanism_eval/qzbc_care_opd_v1/recovery_margin/recovery_margin.jsonl
/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal/family_clean_views/clean_sft.jsonl
/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal/family_clean_views/gkd_full.jsonl
/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal/family_clean_views/care_opd.jsonl
```

## Run

```bash
cd /opt/care-workspace/care_opd

export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
export MECHANISM_ROOT=/opt/care-workspace/care_opd/runs/mechanism_eval/qzbc_care_opd_v1
export FORMAL_ROOT=/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal
export OUTPUT_DIR="$MECHANISM_ROOT/margin_behavior"

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  PYTHON_BIN="$PYTHON_BIN" \
  MECHANISM_ROOT="$MECHANISM_ROOT" \
  FORMAL_ROOT="$FORMAL_ROOT" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  BOOTSTRAP_REPLICATES=10000 \
  BOOTSTRAP_SEED=20260813 \
  bash /opt/care-workspace/care_opd/mechanism_experiments/margin_behavior/run_margin_behavior.sh \
  >"$LOG" 2>&1 </dev/null &

echo $! > "$OUTPUT_DIR/run.pid"
echo "$LOG"
tail -f "$LOG"
```

The statistical analysis uses only the Python standard library. The launcher
also creates a 300-DPI PNG and vector PDF when `matplotlib` and `numpy` are
available; otherwise it prints a warning and retains all numeric outputs.

## Outputs

```text
margin_behavior/
├── margin_behavior.json          # full machine-readable result and provenance
├── margin_behavior.md            # paper-facing interpretation table
├── margin_behavior_summary.csv   # aggregate statistics
├── margin_behavior_examples.csv  # paired per-request audit ledger
├── fig_margin_behavior_closure.png
└── fig_margin_behavior_closure.pdf
```

The report includes positive-margin rates, medians, CARE−GKD paired deltas,
10,000-replicate request-ID bootstrap intervals, rescue rates by deterministic
margin-rank quartile, AUROC, and point-biserial association. Bootstrap
resampling is paired at the request-ID level.

## Basic test

```bash
cd /opt/care-workspace/care_opd
/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python \
  -m unittest mechanism_experiments.margin_behavior.test_margin_behavior -v
```

## Claim boundary

This experiment closes a behavioral association loop: a higher frozen
teacher-forced preference can be tested against realized verifier rescue on
the same examples. It does not establish that the margin caused rescue, and a
CARE-versus-GKD contrast compares their complete training packages rather
than isolating the span mask or any single loss coefficient.
