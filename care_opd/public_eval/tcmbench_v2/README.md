# TCMBench V2 public paired-robustness evaluation

This runner tests whether CARE-OPD transfers recovery/stability beyond the
private QZBC benchmark.  Its primary evidence is not another TCM knowledge
leaderboard score: it compares the same semantic subquestions under the public
`SDT`, `SDT_reverse`, and `SDT_shuffle` sequential-chat histories.

No TCMBench questions are distributed here.  The upstream repository currently
has no explicit `LICENSE`/`COPYING` file, so obtain the data independently and
point `DATA_ROOT` to it.  This folder contains only code, hashes, and item IDs.

## What is primary and what is diagnostic

| Status | Tasks | Use |
|---|---|---|
| Primary paired robustness | `SDT -> SDT_reverse`, `SDT -> SDT_shuffle` | Main generalization claim |
| Diagnostic only | `DDST`, `DDST_hard`, `SDDT`, `SDDT_hard` | Report scores, do not average into the robustness claim |
| Excluded | `herb_predict` | Public prompt/item objective mismatch |
| Optional breadth | All 13 non-herb tasks | Domain knowledge retention, not recovery evidence |

The primary join is `case_index + SHA256(normalized share_content,
subquestion+options)`.  It never joins by list position.  Inspection must verify
77 cases and 264 semantic subquestions per perturbation, with identical gold
answers.  A complete mapping and its hash are saved as `sdt_pair_manifest.json`.

The public snapshot has several machine-verified limitations:

- one of 1,600 FKU rows has a non-A-E gold (`"2"`) and is recorded as
  unscorable rather than silently coerced;
- all 211 clean `DDST` rows omit visible A-E options;
- all 86 `DDST_hard` labels select the final “all questions” position, giving a
  100% positional shortcut; the historical official dispatcher also shadows
  its dedicated hard-task branch;
- 90 clean `SDDT` rows have identical-input matches in `SDDT_hard`, but only
  15/100 match instances retain the same gold label, so these are neither clean
  perturbation pairs nor a valid consistency test;
- `herb_predict`'s prefix asks for 20 herbs, while all 239 embedded item
  questions ask for syndrome and therapy.  Historical code ignores that item
  question and recommends from symptoms instead.

These findings are why only the SDT pair track supports the robustness claim.

## Data snapshot and provenance

- Upstream: <https://github.com/ywjawmw/TCMBench>
- question-data commit: `7e2d2326ff79da186c6fccaafa374cf1f77ac968`
- historical pipeline used for protocol audit:
  `f0138f30eff2ab8f5abcf07637fb3aeea8515c8a`
- all 14 JSON checksums and row counts are pinned in `protocol.py`;
- the answer-only results are **not claimed as published-official numbers**.

Example independent checkout on the server:

```bash
cd /opt/care-workspace
git clone https://github.com/ywjawmw/TCMBench.git TCMBench
git -C /opt/care-workspace/TCMBench checkout 7e2d2326ff79da186c6fccaafa374cf1f77ac968
```

If the current branch no longer contains data, use a detached worktree without
changing the main checkout:

```bash
git -C /opt/care-workspace/TCMBench worktree add \
  /opt/care-workspace/TCMBench_data_7e2d232 \
  7e2d2326ff79da186c6fccaafa374cf1f77ac968
export TCMBENCH_REPO=/opt/care-workspace/TCMBench_data_7e2d232
```

## Formal six-checkpoint run on physical GPUs 4, 5, 6, and 7

The default `robustness` track runs the valid SDT triplet plus all requested
DDST/SDDT diagnostics.  It uses four independent model replicas, one per GPU,
and evaluates the six frozen checkpoints sequentially.

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

export TCMBENCH_REPO=/opt/care-workspace/TCMBench_data_7e2d232
export OUT_ROOT=/opt/care-workspace/care_opd/runs/public_eval/tcmbench_v2
export PROTOCOL=answer_only
export TRACK=robustness
export PHYSICAL_GPUS=4,5,6,7
export NUM_WORKERS=4
export DEVICES=cuda:0,cuda:1,cuda:2,cuda:3

nohup bash public_eval/tcmbench_v2/run_existing_gpu4_5.sh \
  > /opt/care-workspace/tcmbench_v2_gpu4_5_6_7.log 2>&1 &

echo $! > /opt/care-workspace/tcmbench_v2_gpu4_5_6_7.pid
tail -f /opt/care-workspace/tcmbench_v2_gpu4_5_6_7.log
```

Rerunning the identical command resumes at completed episode boundaries.  It
refuses a changed model path/fingerprint, data hash, prompt protocol, task set,
worker layout, or overlap manifest.  A partial final JSONL line is repaired;
non-terminal corruption is fatal.
The run fingerprint includes the four logical devices, worker count, and the
physical mapping recorded from `CUDA_VISIBLE_DEVICES`.

To run only the scientifically valid and faster pair track:

```bash
TRACK=primary_sdt \
OUT_ROOT=/opt/care-workspace/care_opd/runs/public_eval/tcmbench_v2_primary \
nohup bash public_eval/tcmbench_v2/run_existing_gpu4_5.sh \
  > /opt/care-workspace/tcmbench_v2_primary_gpu4_5_6_7.log 2>&1 &
```

To cover all 13 non-herb tasks as an additional knowledge-retention appendix:

```bash
TRACK=all13 \
OUT_ROOT=/opt/care-workspace/care_opd/runs/public_eval/tcmbench_v2_all13 \
nohup bash public_eval/tcmbench_v2/run_existing_gpu4_5.sh \
  > /opt/care-workspace/tcmbench_v2_all13_gpu4_5_6_7.log 2>&1 &
```

## Prompt protocols

`answer_only` is the pre-registered primary protocol.  It preserves official
questions, option labels, sequential turn order, and exact accuracy, but asks
for `<eoa>A</eoa>` and uses deterministic decoding.  It is economical and has
reliable strict parsing; it must be labelled “adapted answer-only TCMBench V2
paired robustness”, not “official TCMBench score”.

`reconstructed_official_cot` retains the public task-specific prompt and
multi-turn history.  It is still not directly comparable to published numbers
because the original API/model versions are different and known historical
dispatch defects are not reproduced.  It is much slower:

```bash
PROTOCOL=reconstructed_official_cot MAX_NEW_TOKENS=384 \
OUT_ROOT=/opt/care-workspace/care_opd/runs/public_eval/tcmbench_v2_cot \
nohup bash public_eval/tcmbench_v2/run_existing_gpu4_5.sh \
  > /opt/care-workspace/tcmbench_v2_cot_gpu4_5_6_7.log 2>&1 &
```

Do not mix the two protocol variants in one table cell.

## Metrics and result organization

Every prediction is scored as option exact 0/1.  The primary summary exposes:

- `clean_task_success_rate` and its perturbation macro;
- `perturbed_task_success_rate` and its perturbation macro;
- `conditional_retention = P(perturbed correct | clean correct)`;
- `regression_rate = 1 - conditional_retention` plus the two clean/perturbed
  correctness flip counts;
- `robustness_drop_pp = 100 * (clean TSR - perturbed TSR)` and a separately
  named `macro_drop_pp`;
- `prediction_consistency` after canonical semantic mapping;
- `worst_perturbation_tsr`;
- `paired_units`, the number of semantic clean/perturbed pairs;
- reverse and shuffle breakdowns;
- question-level main results and case-all-correct secondary results.

`perturbed_task_success_rate` is the main public effectiveness metric;
`conditional_retention`/`regression_rate` are the main robustness metrics.
Drop and prediction consistency are supporting diagnostics.
The run manifest records the exact SHA-256 of
`public_eval/UNIFIED_METRIC_SCHEMA.json`; changing that frozen contract changes
the run fingerprint and requires a new output root.

Task-level official-data accuracy, task macro/micro, strict-format rate, parse
failures, Wilson intervals, paired flip counts, and exact McNemar p-values are
also retained.  Do not combine this benchmark numerically with QZBC or
TCMEval-PA into one opaque overall score.

The suite additionally preregisters `care_opd_vs_clean_sft` and
`care_opd_vs_gkd_full`: perturbed success is compared over all semantic pairs,
while the paired retention test uses the subset both models solved in clean
form.  Full per-model conditional retention is still shown separately.

Each view is reported twice:

- `official_full`: every item in the pinned public snapshot;
- `overlap_clean`: removes the 39 item IDs in `overlap_exclusions.json`.

The robustness suite contains four exclusions (one SDDT and three SDDT_hard),
while the valid SDT pair track has zero detected overlap, so its two views must
be identical.  The bundled exclusion file was derived from the existing 9,600
row training overlap audit and contains no question text.

Main artifacts:

```text
OUT_ROOT/
  model_suite.json
  inspection/data_inspection.json
  inspection/sdt_pair_manifest.json
  answer_only/suite_summary.json
  answer_only/suite_summary.csv
  answer_only/<model>/run_manifest.json
  answer_only/<model>/episodes.jsonl
  answer_only/<model>/predictions.jsonl
  answer_only/<model>/paired_predictions.jsonl
  answer_only/<model>/COMPLETE.json
  answer_only/<model>/shards/
  logs/
```

The `frozen` label is inherited from the frozen six-model suite and is intended
to be the raw initial Qwen3.5-4B checkpoint.  Publication still requires
checking its recorded path and full weight/tokenizer fingerprint; the label
alone is not identity proof.  OPD's isolated contribution is CARE-OPD versus
the shared clean-SFT warm start, not CARE-OPD versus frozen alone.

## CPU verification

```bash
cd /opt/care-workspace/care_opd
export TCMBENCH_DATA_ROOT=/opt/care-workspace/TCMBench_data_7e2d232/data
export PYTHONPATH=/opt/care-workspace/care_opd/public_eval/tcmbench_v2:/opt/care-workspace/care_opd/src

"$PYTHON_BIN" public_eval/tcmbench_v2/test_protocol.py -v
"$PYTHON_BIN" public_eval/tcmbench_v2/evaluate_tcmbench_v2.py inspect-data \
  --data-root "$TCMBENCH_DATA_ROOT" \
  --track robustness
```

This is a synthetic/public benchmark evaluation, not clinical advice.
