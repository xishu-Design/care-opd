# TCMEval-PA v3 public evaluation

This directory evaluates the six existing Qwen3.5-4B checkpoints on the
official 328-question TCMEval-PA workbook. It is independent of the internal
QZBC recovery benchmark: model outputs are option letters and are **not** sent
through the QZBC `GeneratedResponse` parser or verifier.

## Fixed formal protocol

- Data: Figshare v3, DOI `10.6084/m9.figshare.29651261.v3`.
- Workbook SHA256:
  `905cb28f145ba50f3b1c912265ff6649e0bb38f56adf6cd588d77ca768db92c1`.
- Upstream TCMEval repository commit:
  `12473b50ce94ad2fd22b1e984b826b0a618c4407`.
- 328 unique IDs: 297 single-choice and 31 multiple-choice.
- Generation: greedy, `max_new_tokens=8`, Qwen chat template,
  `enable_thinking=False`, bfloat16.
- Primary metric: exact equality of the parsed option set. Also reported:
  strict exact, strict-format rate, parse-failure rate, single/multiple,
  Category, and Rule_ID slices.
- A question is keyed by its official `ID` and its own option inventory.
  Repeated stems are never cached or deduplicated (328 IDs, 314 unique stems).

There are two deliberately separate prompt protocols:

- `official` (default): recovers the old workbook's single/multiple sheet type
  from gold-answer cardinality and uses the two official system prompts. The
  merged v3 sheet no longer contains a type column. Every manifest therefore
  discloses `type_reconstructed_from_gold_cardinality=true`. This is the main
  protocol for comparison with the official paper's reference table.
- `label_blind`: uses one unified prompt saying that one or more options may be
  correct. No gold-derived type reaches the prompt. This is a sensitivity
  analysis and is not mixed with `official` outputs.

## Server run (physical GPUs 4 and 5)

The launcher fixes the model order to `frozen`, `clean_sft`, `recovery_sft`,
`recovery_dpo`, `gkd_full`, and `care_opd`. It validates every merged HF model,
resolves the CARE final-model pointer, freezes canonical paths and full content
fingerprints in `model_suite.json`, and runs one model at a time. Within each
model two independent processes use logical `cuda:0,cuda:1` under
`CUDA_VISIBLE_DEVICES=4,5`.

```bash
cd /opt/care-workspace/care_opd

nohup env \
  PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python \
  bash public_eval/tcmeval_pa/run_existing_gpu4_5.sh \
  > /opt/care-workspace/tcmeval_pa_gpu4_5.log 2>&1 &

echo $!
tail -f /opt/care-workspace/tcmeval_pa_gpu4_5.log
```

To additionally run the label-blind sensitivity protocol after the official
suite:

```bash
nohup env \
  PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python \
  PROMPT_MODES='official label_blind' \
  bash /opt/care-workspace/care_opd/public_eval/tcmeval_pa/run_existing_gpu4_5.sh \
  > /opt/care-workspace/tcmeval_pa_both_modes_gpu4_5.log 2>&1 &
```

Default output root:

```text
/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/
  data/TCMEval-PA.xlsx  # only used when no bundled workbook is present
  model_suite.json
  logs/
  official/<model>/
    run_manifest.json
    shards/worker_00.jsonl
    shards/worker_01.jsonl
    predictions.jsonl
    summary.json
    COMPLETE.json
  official/suite_summary.json
  official/suite_summary.csv
  official/suite_summary.md
```

The launcher first uses `public_eval/tcmeval_pa/data/TCMEval-PA.xlsx` when it
is included in the delivery bundle and verifies it without network access.
Only when no bundled or explicitly supplied `DATA_FILE` exists does it fetch
the fixed v3 artifact into the output root.

The launcher uses `flock` to prevent two suites from sharing an output root and
refuses to start a model while physical GPU 4 or 5 has a compute process.
The bundled, checksum-pinned workbook is copied into the output root before
evaluation, so the formal run does not require network access.
Shard rows are fsynced after every question. A rerun resumes missing rows; it
skips a model only after validating 328 unique IDs plus the dataset, prompt,
generation, and model fingerprints. `COMPLETE.json` is published atomically
only after merged predictions and the summary are complete.

If a checkpoint changes after `model_suite.json` is frozen, use a new
`OUT_ROOT`; the evaluator intentionally refuses to mix artifacts.

## Data-only and CLI checks

```bash
python public_eval/tcmeval_pa/prepare_data.py \
  --output /opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/data/TCMEval-PA.xlsx

python public_eval/tcmeval_pa/evaluate_tcmeval_pa.py inspect-data \
  --data /opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/data/TCMEval-PA.xlsx
```

Expected inspection counts are 328 questions, 328 IDs, 297/31 by type,
6/86/236 by category, 19 rules, and 314 unique stems.

The CPU-only protocol regression checks can be run without torch or pytest:

```bash
python public_eval/tcmeval_pa/test_protocol.py -v
```

`overlap_audit_summary.json` records the already-completed comparison against
the 9,600 formal QZBC training seeds: no exact, same-item, visible-segment, or
configured 13-character n-gram near match was found. This does not rule out
semantic overlap, shared regulatory knowledge, teacher knowledge, or exposure
during base-model pretraining.
