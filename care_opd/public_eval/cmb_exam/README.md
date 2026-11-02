# CMB-Exam public evaluation

This directory evaluates the same six Qwen3.5-4B checkpoints on all 11,200
official CMB-Exam test questions. It is independent of the internal QZBC
recovery verifier: answers are option letters and the primary metric is exact
equality of parsed option sets.

## Fixed protocol

- Public source: CMB repository commit
  `6c8ece46097dae736c6805dd3b831e1a38c08971` (Apache-2.0).
- Official test: 11,200 IDs, six exam types and 28 exam classes (400 per
  class).
- Official question types: 10,010 single-choice and 1,190 multiple-choice.
  The public answer-file metadata defines type; gold cardinality never defines
  the prompt. This matters because 25 official multiple-choice records have a
  one-letter key and 11 question records marked `C型选择题` are marked single
  choice in the official answer file.
- Prompt: official zero-shot answer-only template documented by CMB; one
  Qwen chat turn, `enable_thinking=False`.
- Generation: greedy, bfloat16, `max_new_tokens=8`, fixed batch size eight.
- Primary metric: option-set exact accuracy. Also reported: Wilson 95% CI,
  strict exact/format, parse failure, single/multiple, six exam types, 28 exam
  classes, and 173 subjects.
- Views: `official_full` (11,200) and `overlap_clean` (11,196). The latter
  removes four exact normalized-stem matches dynamically derived from the
  pinned formal overlap report. It does not claim to remove semantic overlap
  or base-model pretraining exposure.
- CARE pairwise analysis: exact McNemar plus fixed-seed paired item and
  normalized-stem-cluster bootstrap intervals. The cluster interval is the
  conservative basis for the preregistered -1 percentage-point preservation
  gate against clean SFT and full GKD.

## Derived controlled robustness track

The same model load also evaluates 2,800 deterministic option-permutation
pairs (`controlled_robustness_v1`): 100 result-blind, overlap-clean items from
each of the 28 classes. Question and option text are preserved while labels,
order, and gold labels are consistently remapped. This is **not an official
CMB track**.

Its fields follow `public_eval/UNIFIED_METRIC_SCHEMA.json`:

- `clean_task_success_rate`
- `perturbed_task_success_rate`
- `conditional_retention`
- `regression_rate`
- `robustness_drop_pp`
- `prediction_consistency`
- `worst_perturbation_tsr`
- `clean_correct_perturbed_wrong`
- `clean_wrong_perturbed_correct`
- `paired_units`

## Formal server run (physical GPUs 4, 5, 6, and 7)

```bash
cd /opt/care-workspace/care_opd

export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
export SOURCE_SUITE=/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/model_suite.json
export PREREGISTRATION_LOCK=/opt/care-workspace/care_opd/public_eval/PREREGISTRATION_LOCK.json
"$PYTHON_BIN" public_eval/create_preregistration_lock.py \
  --project-root /opt/care-workspace/care_opd \
  --model-suite "$SOURCE_SUITE" \
  --bert-model /opt/care-workspace/models/bert-base-chinese \
  --output "$PREREGISTRATION_LOCK"

nohup env \
  PYTHON_BIN="$PYTHON_BIN" \
  PREREGISTRATION_LOCK="$PREREGISTRATION_LOCK" \
  PHYSICAL_GPUS=4,5,6,7 \
  NUM_WORKERS=4 \
  DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
  bash public_eval/cmb_exam/run_existing_gpu4_5.sh \
  > /opt/care-workspace/cmb_exam_gpu4_5_6_7.log 2>&1 &

echo $!
tail -f /opt/care-workspace/cmb_exam_gpu4_5_6_7.log
```

The fixed model order is `frozen`, `clean_sft`, `recovery_sft`,
`recovery_dpo`, `gkd_full`, and `care_opd`. The launcher validates all merged
HF exports, resolves the CARE final pointer, freezes canonical paths and full
content fingerprints, checks that physical GPUs 4, 5, 6, and 7 are idle, and
runs one model at a time. Within a model, four spawned workers use logical
`cuda:0,cuda:1,cuda:2,cuda:3` under `CUDA_VISIBLE_DEVICES=4,5,6,7`. The worker
count, logical devices, and physical mapping are included in the run
fingerprint.

Formal prediction also requires
`public_eval/PREREGISTRATION_LOCK.json`. The launcher fails closed if it is
missing or empty; the evaluator parses it and places its SHA256 under the
exact key `preregistration_lock_sha256` in the run contract, fingerprints,
analysis specification, and suite summary. `PREREGISTRATION_LOCK=/path/file`
may override the default, but every model in one suite must use the same
pre-prediction lock.

Default output:

```text
/opt/care-workspace/care_opd/runs/public_eval/cmb_exam_v1/
  data/
  model_suite.json
  logs/
  official/<model>/
    run_manifest.json
    shards/worker_00.jsonl
    shards/worker_01.jsonl
    shards/worker_02.jsonl
    shards/worker_03.jsonl
    predictions.jsonl                         # 14,000 inference rows
    official_predictions.jsonl                # 11,200 rows
    controlled_robustness_predictions.jsonl   # 2,800 rows
    controlled_robustness_pairs.jsonl         # paired item output
    summary.json
    COMPLETE.json
  official/suite_summary.json
  official/suite_summary.csv
  official/suite_summary.md
```

Shard rows are fsynced after every item. A rerun repairs only a corrupt final
line and resumes missing rows. A model is skipped only after validating every
ID, all fingerprints, all split outputs, and `COMPLETE.json`. If data, code,
prompt, generation, model, robustness selection, or unified metric schema
changes, the evaluator refuses to mix results in the same output root.

## CPU-only checks

```bash
python public_eval/cmb_exam/prepare_data.py \
  --output-dir public_eval/cmb_exam/data \
  --verify-only

python public_eval/cmb_exam/evaluate_cmb_exam.py inspect-data \
  --data-dir public_eval/cmb_exam/data

python public_eval/cmb_exam/test_protocol.py -v
```
