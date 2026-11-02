# TCMEval-PA controlled robustness v1

This follow-up measures both **effectiveness** and **robustness** under one
paired protocol. It is independent of the completed official run and defaults
to a new output root:

```text
/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_controlled_v1
```

## What is controlled

All six Qwen3.5-4B checkpoints see all 328 official IDs under both conditions.
Both conditions use the same label-blind system prompt, Qwen chat template,
`enable_thinking=False`, greedy decoding, and an eight-token output budget.

- `clean`: the canonical options, normalized to the same option-line format as
  the perturbed condition.
- `perturbed`: the option contents are permuted by a fixed SHA256 function of
  the official ID. Every permutation is forced to be non-identity. The displayed
  gold labels and parsed predictions are remapped consistently, and scoring is
  performed in canonical-label space.

`mapping_manifest.json` is materialized and hashed before any model process is
started. A changed mapping, metric schema, preregistration lock, model artifact,
prompt, generation contract, or evaluator changes the run fingerprint and
cannot resume into the same result directory.

## Unified main-table metrics

The exact cross-benchmark field contract is loaded from and hashed at
`public_eval/UNIFIED_METRIC_SCHEMA.json`. TCMEval-specific permutation and
bootstrap choices are frozen separately in `ANALYSIS_SPEC.json`.

- Main effectiveness: `perturbed_task_success_rate` (exact option-set success
  after canonical remapping).
- Main robustness: `conditional_retention` and `regression_rate`.
- Diagnostics: paired state counts (`both_correct`, C→W, W→C, `both_wrong`),
  `robustness_drop_pp`, canonical `prediction_consistency`, and
  `worst_perturbation_tsr`.

Per-model 95% confidence intervals and CARE-vs-Clean-SFT/GKD deltas use a fixed
10,000-replicate percentile bootstrap over normalized question-stem clusters
(314 clusters), rather than treating all 328 IDs as independent. The exact
McNemar test on perturbed task success is retained only as an exploratory
diagnostic.

For CARE versus `clean_sft` (primary) and `gkd_full` (secondary), the frozen
joint preservation margin is -2 percentage points. Both the perturbed-TSR and
conditional-retention CARE-minus-baseline cluster-bootstrap 95% CI lower bounds
must exceed `-0.02`. Passing means non-inferior/preserved, not superior; the two
endpoint decisions and the joint decision are written to
`paired_comparisons.csv` and `suite_summary.json`.

The original official-prompt scores are loaded read-only from the completed
official root, validated by completion and artifact fingerprints, and written
to `official_original_scores.csv`. They are never mixed with the controlled
scores in `suite_summary.csv` and are never used to derive robustness metrics.

## Preregistration gate

The formal evaluator fails closed unless this file exists:

```text
/opt/care-workspace/care_opd/public_eval/PREREGISTRATION_LOCK.json
```

The launcher also accepts `PREREGISTRATION_LOCK=/absolute/path/to/lock.json`.
Its file SHA256 is recorded in every run fingerprint. Create/finalize the lock
only after the protocol, scripts, schema, benchmark choices, models, and planned
analyses are frozen; changing it requires a new output root.

## Formal server run on physical GPUs 4, 5, 6, and 7

```bash
cd /opt/care-workspace/care_opd

# Run this once only after every public evaluator and analysis contract is
# final. The source suite is the already completed six-checkpoint TCMEval suite.
export SOURCE_SUITE=/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/model_suite.json
export PREREGISTRATION_LOCK=/opt/care-workspace/care_opd/public_eval/PREREGISTRATION_LOCK.json
python public_eval/create_preregistration_lock.py \
  --project-root /opt/care-workspace/care_opd \
  --model-suite "$SOURCE_SUITE" \
  --bert-model /opt/care-workspace/models/bert-base-chinese \
  --output "$PREREGISTRATION_LOCK"

nohup env \
  PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python \
  SOURCE_SUITE="$SOURCE_SUITE" \
  BERT_MODEL=/opt/care-workspace/models/bert-base-chinese \
  PREREGISTRATION_LOCK="$PREREGISTRATION_LOCK" \
  PHYSICAL_GPUS=4,5,6,7 \
  NUM_WORKERS=4 \
  DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
  bash public_eval/tcmeval_pa/run_controlled_gpu4_5.sh \
  > /opt/care-workspace/tcmeval_pa_controlled_gpu4_5_6_7.log 2>&1 &

echo $!
tail -f /opt/care-workspace/tcmeval_pa_controlled_gpu4_5_6_7.log
```

Optional path overrides are `OUT_ROOT`, `OFFICIAL_SUITE_ROOT`, `SOURCE_SUITE`,
`OFFICIAL_RESULTS_ROOT`, `DATA_FILE`, `PREREGISTRATION_LOCK`,
`ORIGIN_REFERENCE`, `PHYSICAL_GPUS`, `NUM_WORKERS`, and `DEVICES`. If an origin
reference was used to create the lock, pass the identical `ORIGIN_REFERENCE`
value to every launcher validation.

The launcher validates and copies the existing six-model `model_suite.json`,
but never writes to its source or to the official result root. It runs one model
at a time; by default four spawned processes use logical
`cuda:0,cuda:1,cuda:2,cuda:3` under `CUDA_VISIBLE_DEVICES=4,5,6,7`. The
historical launcher filename is retained for compatibility. Physical and
logical device order plus worker count are bound into each run fingerprint.
Each completed paired item is fsynced as one JSONL row. A rerun removes only a
corrupt terminal line and resumes missing pairs. Final predictions, summaries,
and completion sentinels are atomic.

## Output layout

```text
tcmeval_pa_controlled_v1/
  README.md
  UNIFIED_METRIC_SCHEMA.json
  ANALYSIS_SPEC.json
  mapping_manifest.json
  model_suite.json
  model_suite_source.json
  paired_label_blind/<model>/
    run_manifest.json
    shards/worker_00.jsonl
    shards/worker_01.jsonl
    paired_predictions.jsonl
    summary.json
    COMPLETE.json
  suite_summary.json
  suite_summary.csv
  suite_summary.md
  official_original_scores.csv
  paired_comparisons.csv
  logs/
```

## CPU-only checks

These tests load the checksum-pinned workbook but do not load a model or use a
GPU:

```bash
python public_eval/tcmeval_pa/test_controlled_robustness.py -v
python public_eval/tcmeval_pa/test_protocol.py -v
```
