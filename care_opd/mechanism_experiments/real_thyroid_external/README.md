# Real thyroid external-test preparation

This package converts the supplied thyroid workbook into **report-level private
records** for a locked external extraction test.  It does not create patient
linkage, cross-modal lesion matches, or new clinical labels.

## Safety defaults

- The default command is aggregate-only audit and never emits cell values.
- Private export requires an explicit flag, acknowledgement, and a 32-byte-or-
  longer HMAC secret supplied through an environment variable or a private key
  file.  The secret is never printed; a generated key file must remain inside
  the ignored private run directory.
- Private output files are named `*.private.jsonl`; the output directory gets a
  deny-all `.gitignore`.
- Source text remains `model_input_allowed=false` until a human marks the
  workbook `reviewed_private_use`.
- `patient_group_hmac`, `encounter_group_hmac`, and `lesion_group_hmac` stay
  null because the workbook contains no valid linkage keys.

## Report grouping contract

- `超声`: a non-empty `检查报告单-检查结论（原始）` begins a report; following blank-source
  rows are continuation nodule rows.  The locked workbook must yield 23 report
  groups.  Nodule IDs are reassigned locally as `N01`, `N02`, ... and are not
  global lesion IDs.
- `病理`: each non-empty `检查报告单-检查结论（原始）` row is one report.  The locked workbook must
  yield 49 reports.  Primary, frozen, and supplemental diagnoses stay separate.

The statistical unit is the report group until an authorized patient HMAC is
provided from outside this workbook.

## Four-state evaluation contract

Builder v5 exposes two named tracks rather than scoring hidden structured
labels against a raw report that may not contain them verbatim:

- `structured_evidence_faithfulness` supplies the reviewed raw report and the
  provided structured clinical fields as two independently citable evidence
  sources. On the frozen input it covers 65 families / 586 fields.
- `raw_strict_extraction_challenge` exposes only the reviewed raw report and
  admits only uniquely span-localized fields. On the frozen input it covers 54
  families / 163 fields.

Each eligible report is materialized as a closed, result-blind family with
`clean`, `bad_history`, `source_outage`, and `evidence_conflict` rows. Only
clean and bad-history rows receive explicit `required_output_slots` with the
actual heading and required source ID. Outage removes every source and clears
the structured context. Conflict keeps the visible evidence but marks it
non-citable and clears both `allowed_extraction_schema` and
`required_output_slots`, so the request neither advertises fields nor asks the
model to extract from evidence it is forbidden to cite. Both boundary
states require a source-free response with no medical-field section or
citation, plus explicit uncertainty, warning, next action, and disclaimer. A
generic evidence-boundary section is optional; if emitted, it must be
source-free, non-clinical, and must not use a locked-field or packet-container
heading. The complete identity, source, transformation, claim, and privacy
contract is frozen in
[`FOUR_STATE_CONTRACT.md`](FOUR_STATE_CONTRACT.md).

Variant names, control labels, and sentinels remain in private verifier
metadata only; they are not rendered into model-visible
messages. In particular, the conflict sentinel is not a retrieved-evidence
item. State recovery must follow from the visible evidence conditions.

For either named track, the primary public endpoints are clean TSR, each named
perturbation TSR, conditional retention/regression, and joint four-state TSR.
All confidence intervals resample whole report families. The pooled
bad-history/outage/conflict TSR combines different target semantics and is
therefore **descriptive only**; it is not a primary superiority endpoint.

The three-axis extraction diagnostics are restricted to `clean+bad_history`.
Outage and conflict are assessed with their boundary contract and the joint
four-state endpoint, never with extraction/content recall. Report the
structured result as **structured-evidence faithfulness TSR** and the raw
result as **raw-report strict extraction challenge TSR**; neither is clinical
diagnostic accuracy.

For structured clean and bad-history rows, success is an exact rendering
contract: one and only one section for every required slot, the exact locked
heading/key, the exact singleton structured source ID, and the exact literal
content. Missing, duplicate, extra, malformed, mis-sourced, or value-mismatched
sections fail. The model-visible
`constraints.conditional_output_contract.with_citable_slots` branch states the
auxiliary target literally as `title=""`, `summary=""`, `warnings=[]`,
`next_actions=[]`, and `disclaimer=""`; any substantive auxiliary output fails.
Boundary rows select `without_citable_slots`. Packet-scaffolding
headings, controlled sentinels, and affirmative reuse of the bad-history answer
also fail. Auxiliary prose cannot compensate for an incorrect slot.

The earlier v4 three-family smoke was a result-blind protocol calibration, not
an effectiveness gate. `SMOKE=PASS` meant only that the selected track,
complete quartet, generation path, verifier, and output writer executed without
runtime errors. Its TSR/F1 values are not paper results. Because v5 changes
model-visible extraction instructions and the conflict/boundary contract, no
v4 smoke or formal directory may be resumed or relabeled as v5.

Build the two tracks into separate locked directories:

```powershell
python mechanism_experiments/real_thyroid_external/build_locked_benchmark.py `
  --canonical "runs\private_real_thyroid_external\locked_v1\canonical_records.private.jsonl" `
  --clinical-gold "runs\private_real_thyroid_external\direct_clinical_gold_v1\clinical_gold.private.jsonl" `
  --task-mode structured_evidence_faithfulness `
  --output-dir "runs\private_real_thyroid_external\benchmark_v5_structured" `
  --acknowledge-private-clinical-data

python mechanism_experiments/real_thyroid_external/build_locked_benchmark.py `
  --canonical "runs\private_real_thyroid_external\locked_v1\canonical_records.private.jsonl" `
  --clinical-gold "runs\private_real_thyroid_external\direct_clinical_gold_v1\clinical_gold.private.jsonl" `
  --task-mode raw_strict_extraction_challenge `
  --output-dir "runs\private_real_thyroid_external\benchmark_v5_raw" `
  --acknowledge-private-clinical-data
```

The expected frozen counts are 65 families / 260 rows / 586 fields for the
structured main track and 54 families / 216 rows / 163 fields for the raw
challenge track. Keep their benchmark and evaluation directories separate.

Minimal authorized-server single-model evaluation paths are:

```bash
cd /opt/care-workspace/care_opd
PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
QZBC=/opt/care-workspace/qzbc_source/qzbc-main
MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model

BENCH=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/benchmark_v5_structured
OUT=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/eval_v5_structured_care
"$PY" mechanism_experiments/real_thyroid_external/thyroid_external_eval.py \
  --seeds "$BENCH/locked_external_qzbc.private.jsonl" \
  --field-gold "$BENCH/field_gold.private.jsonl" \
  --qzbc-root "$QZBC" --model "care_opd=$MODEL" \
  --reference-model care_opd --output-dir "$OUT" --device cuda:0 \
  --generation-seed 42 --bootstrap-replicates 10000 \
  --bootstrap-seed 20260819 --acknowledge-private-clinical-data --overwrite

# Run only after the structured job finishes on the same GPU.
BENCH=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/benchmark_v5_raw
OUT=/opt/care-workspace/care_opd/runs/private_real_thyroid_external/eval_v5_raw_care
"$PY" mechanism_experiments/real_thyroid_external/thyroid_external_eval.py \
  --seeds "$BENCH/locked_external_qzbc.private.jsonl" \
  --field-gold "$BENCH/field_gold.private.jsonl" \
  --qzbc-root "$QZBC" --model "care_opd=$MODEL" \
  --reference-model care_opd --output-dir "$OUT" --device cuda:0 \
  --generation-seed 42 --bootstrap-replicates 10000 \
  --bootstrap-seed 20260819 --acknowledge-private-clinical-data --overwrite
```

The reserved formal six-model output roots are
`eval_direct_gold_structured_six_models_v5` and
`eval_direct_gold_raw_six_models_v5`. Do not use either name for a single-model
run; the full guarded server commands are in `RUNBOOK_ZH.md`.

Do not reuse or resume the v4 direct-gold server bundle. Regenerate its
benchmark payloads, evaluator/verifier copies, run scripts, runbook, bundle
manifest, and checksum index as listed in `RUNBOOK_ZH.md`; private payloads are
materialized only inside the authorized private environment and are not copied
by this documentation update.

## 1. Aggregate-only audit (recommended first)

From `care_opd/`:

```powershell
python mechanism_experiments/real_thyroid_external/prepare_external_test.py `
  "D:\private\甲状腺结构化数据.xlsx" `
  --output-dir "runs\private_real_thyroid_external\audit"
```

The only output is `audit_summary.json`, containing counts and hashes but no
cell values or absolute source path.

## 2. Private canonical records and Agent queue

Use an approved local/private agent environment only.  Do not put the secret on
the command line because shell history may retain it.

```powershell
$env:CARE_OPD_THYROID_HMAC_KEY = "replace-with-a-random-secret-of-at-least-32-bytes"

python mechanism_experiments/real_thyroid_external/prepare_external_test.py `
  "D:\private\甲状腺结构化数据.xlsx" `
  --output-dir "runs\private_real_thyroid_external\locked_v1" `
  --private-export `
  --acknowledge-private-clinical-data `
  --deidentification-status unreviewed `
  --authorization-reference "LOCAL-APPROVAL-REFERENCE"

Remove-Item Env:CARE_OPD_THYROID_HMAC_KEY
```

For the first locked build, the script can create a stable private key file
without exposing its value:

```powershell
python mechanism_experiments/real_thyroid_external/prepare_external_test.py `
  "D:\private\甲状腺结构化数据.xlsx" `
  --output-dir "runs\private_real_thyroid_external\locked_v1" `
  --private-export --acknowledge-private-clinical-data `
  --hmac-key-file "runs\private_real_thyroid_external\locked_v1\.thyroid_hmac_key.private" `
  --generate-hmac-key-file
```

Reuse the same key file for every rebuild; never regenerate it after report IDs
have been frozen.

Outputs:

- `canonical_records.private.jsonl`: 72 report-level candidate records.
- `agent_work_queue.private.jsonl`: report text, field templates, and Excel
  candidates for the local runner.  The runner constructs a blind-extractor
  view that excludes `candidate_fields`, and a separate auditor view that
  includes them.
- `private_manifest.json`: counts and file hashes only.

The private canonical and queue contracts are frozen in
`canonical_record.schema.json` and `agent_queue.schema.json`.

The Agent may extract only what has exact source support.  It must not infer
patient linkage, ultrasound-pathology correspondence, a final diagnosis,
negative findings from blank cells, or treatment recommendations.  Agent
annotations remain provisional until clinician review/adjudication.

The queue implements the contract in `agent_queue.schema.json`.  Every
row has `case_id`, `report_group_hmac`, `modality`, `source_text_private`,
`source_record_sha256`, and `candidate_fields`.  Each visible candidate field
has the runner-facing keys `entity_id`, `field_path`, and `raw_value` plus its
annotation requirements.  The local runner, rather than the queue file,
enforces candidate blinding for the independent extractor.  A primary label
requires the sheet candidate, an exact or pre-approved-alias span from the
candidate auditor, and agreement from the independent blind extractor;
disagreements require clinician adjudication.

## 3. Source-grounded multi-Agent supplementation

The runner uses a blind evidence locator, a spreadsheet-candidate auditor, and
an adjudicator.  It accepts a field into the strict layer only when the
spreadsheet candidate, independent extraction, exact source quote, entity, and
adjudicator agree.  Everything else is written to a human-review queue.

```powershell
python mechanism_experiments/real_thyroid_external/run_local_agent_annotation.py `
  --queue "runs\private_real_thyroid_external\locked_v1\agent_work_queue.private.jsonl" `
  --output-dir "runs\private_real_thyroid_external\locked_v1\agent_annotations" `
  --base-url "http://127.0.0.1:11434" `
  --locator-model "<APPROVED_LOCAL_MODEL_A>" `
  --auditor-model "<APPROVED_LOCAL_MODEL_B>" `
  --adjudicator-model "<APPROVED_LOCAL_MODEL_C>" `
  --acknowledge-private-clinical-data `
  --seed 20260818
```

Use different approved local models for locator and auditor when available. If
one model serves all three roles, the summary records
`same_model_all_roles=true`; describe the result as single-model multi-agent
silver annotation, not independent annotator consensus.

Outputs:

- `exact_verified_fields.private.jsonl`: strict source-verified layer suitable
  for the primary extraction endpoint.
- `human_review_queue.private.jsonl`: all unresolved, ambiguous, missing, or
  agent-disagreement fields.
- `agent_annotation_summary.json`: aggregate counts only; contains no report
  text or field values.
- `case_results.private/`: private role outputs and hashes for audit/resume.

Non-loopback endpoints are blocked by default.  If an institution-approved
private GPU service is used, the operator must pass `--allow-nonloopback`; this
flag is an acknowledgement, not a privacy or authorization check.

After an authorized human de-identification review, rerun with
`--deidentification-status reviewed_private_use`; that flag records the review
state but does not perform de-identification itself.

## 4. Validation

The end-to-end preparation, strict benchmark build, local-checkpoint
evaluation, private/public output separation, and report-family-cluster bootstrap
commands are documented in [RUNBOOK_ZH.md](RUNBOOK_ZH.md).

The tests build a fully synthetic XLSX fixture and contain no real reports:

```powershell
python -m unittest discover `
  -s mechanism_experiments/real_thyroid_external/tests `
  -p "test_*.py" -v
```

## Publication boundary

Only aggregate metrics, schemas, code, and fully synthetic examples may leave
the private environment.  Never upload the original workbook, private JSONL,
agent outputs, per-case model outputs, or the HMAC secret to GitHub/Hugging Face.
