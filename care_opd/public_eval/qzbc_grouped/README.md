# Identity-bound QZBC grouped evaluation

This directory replaces the old path-only re-analysis with a formal rerun of
the same QZBC model-boundary benchmark. It deliberately reuses
`care_opd.qzbc_rollout`; no alternative Agent, parser, verifier, or task target
is introduced.

## What is bound

Before inference, `prepare` fails closed unless all of the following close:

- the six labels and exact canonical checkpoint paths/fingerprints in the
  frozen `model_suite.json`;
- the static preregistration lock and the evaluator/runtime code inventory;
- the QZBC source contract files and grounding-asset files recorded by the
  seed manifest;
- all/train/dev/test seed paths, rows, and recomputed SHA-256;
- every compiled train/dev/test JSONL, its row count, and recomputed SHA-256;
- every compiled row's `seed_id` membership in its declared split, plus a
  byte-identical deterministic rebuild of all seven test compiler outputs;
- 4,200 unique test request IDs, seed membership, four-variant balance, and
  the request SHA-256;
- pairwise-disjoint seed IDs, exact curated-asset entity keys, and canonical
  request/target hashes.

Each model is fully SHA-256 rehashed before its run. A model is resumable only
when its `COMPLETE.json` has the same run fingerprint and every listed artifact
still has the recorded size and SHA-256. A changed checkpoint, request file,
runtime, evaluator, generation knob, or static lock requires a new output root.

## Scientific scope

The original split is **entity-level in-domain**, not a strict provenance
family split. Exact `(asset_type, asset_id, source_record_id)` entities do not
cross splits, but some `(source_id, source_record_id)` families do. Therefore:

- the full 4,200-row view is retained as `ENTITY_HOLDOUT_ONLY` diagnostic;
- a fixed, result-blind family filter excludes a whole test seed if any of its
  provenance-family keys occurs in train/dev, yielding 975 seeds and 3,900
  requests with zero family overlap;
- that 3,900-row view is a **retrospective contamination-sensitivity view**
  frozen before this identity-bound rerun, not a genuinely preregistered or
  external benchmark;
- the same generator, templates, task families and distribution remain, so
  neither view is evidence of template-level/OOD generalization.

The evaluator audits current data, but no contemporaneous cryptographic receipt
binds those bytes to every historical training run. The output therefore says
`CURRENT_DATA_AUDITED_TRAINING_BYTES_UNVERIFIED`: it proves current bytes and
current split integrity, but cannot retroactively prove that every old
checkpoint used those exact bytes. A strict future claim requires rebuilding a
group-split corpus and retraining all methods from that frozen split.

## Server run (physical GPUs 4, 5, 6, and 7)

First create a new preregistration lock after adding this directory and the
QZBC runtime files to the lock inventory (see the list below). Then run:

```bash
cd /opt/care-workspace/care_opd

EVAL_PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
SOURCE_SUITE=/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/model_suite.json
LOCK=/opt/care-workspace/care_opd/public_eval/PREREGISTRATION_LOCK_QZBC_20260809_4gpu_v1.json

"$EVAL_PY" public_eval/create_preregistration_lock.py \
  --project-root /opt/care-workspace/care_opd \
  --model-suite "$SOURCE_SUITE" \
  --output "$LOCK"

LOG=/opt/care-workspace/care_opd/runs/public_eval/logs/qzbc_grouped_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"

nohup env \
  PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python \
  QZBC_ROOT=/opt/care-workspace/qzbc_source/qzbc-main \
  SOURCE_SUITE="$SOURCE_SUITE" \
  PREREGISTRATION_LOCK="$LOCK" \
  OUT_ROOT=/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1 \
  PHYSICAL_GPUS=4,5,6,7 \
  NUM_WORKERS=4 \
  DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
  bash public_eval/qzbc_grouped/run_existing_gpu4_5.sh \
  >"$LOG" 2>&1 </dev/null &

echo $!
echo "$LOG"
tail -f "$LOG"
```

The historical launcher filename is retained for compatibility. Its defaults
are `PHYSICAL_GPUS=4,5,6,7`, `NUM_WORKERS=4`, and logical devices
`cuda:0,cuda:1,cuda:2,cuda:3`. The four GPUs process disjoint request shards
for one checkpoint, and checkpoints run sequentially. Physical and logical
device order plus worker count are bound into each run fingerprint. Finished
models are validated and skipped on restart.

## Outputs

```text
qzbc_grouped_identity_4gpu_v1/
  model_suite.json
  DATA_AUDIT.json
  PREPARED.json
  formal/<model>/run_manifest.json
  formal/<model>/qzbc_rollouts.jsonl
  formal/<model>/evaluation_summary.json
  formal/<model>/COMPLETE.json
  formal/family_clean_views/<model>.jsonl
  formal/suite_summary.json
  formal/suite_summary.csv
  formal/suite_summary.md
  formal/SUITE_COMPLETE.json
```

`suite_summary.json.main_table_records` is directly consumable as the formal
internal block. `models` contains family-clean metrics; `full_entity_view`
retains the complete 4,200-row diagnostics. Both retain per-variant and failure
type counts, while the main-table record uses the shared ten-field metric
schema.

## Required static-lock additions

The root lock inventory must include:

```text
src/care_opd/qzbc_contract.py
src/care_opd/qzbc_assets.py
src/care_opd/qzbc_dataset.py
src/care_opd/qzbc_rollout.py
src/care_opd/qzbc_seed_builder.py
src/care_opd/rollout.py
src/care_opd/trainer_utils.py
public_eval/qzbc_grouped/ANALYSIS_SPEC.json
public_eval/qzbc_grouped/protocol.py
public_eval/qzbc_grouped/evaluate_qzbc_grouped.py
public_eval/qzbc_grouped/run_existing_gpu4_5.sh
```
