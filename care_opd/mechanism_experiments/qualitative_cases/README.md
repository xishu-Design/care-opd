# Result-blind qualitative case selector

This offline utility produces a six-slot case panel without manually browsing
responses and choosing favorable examples. Selection depends only on fixed
categories, verifier outcomes for two explicitly diagnostic slots, CARE full
recovery margins, and a fixed seeded SHA-256 tie-break.

## Six fixed slots

1. `bad_history`: closest to the category median CARE full margin.
2. `source_outage`: closest to the category median CARE full margin.
3. `evidence_conflict`: closest to the category median CARE full margin.
4. Clean-SFT `invented_source_id`: closest to its median CARE full margin.
5. CARE still fails: closest to the median margin among remaining CARE failures.
6. CARE regression: Clean-SFT passes but CARE fails, selected by seeded hash.

IDs are unique across slots in the listed priority order. If a category has no
eligible unused item, its slot is emitted as `MISSING`; no replacement category
or invented example is allowed. Slots 5 and 6 are deliberately
outcome-conditioned diagnostics. The aggregate quantitative evaluation—not
this panel—supports effectiveness claims.

The margin assay is defined only on Clean-SFT failures, so slot 6 cannot use a
margin without changing the cohort. It therefore uses the frozen seeded-hash
rule. Response text is never a selection feature.

## Input contract

- Three complete, paired formal `family_clean_views` archives for Clean-SFT,
  GKD-full, and CARE-OPD.
- The result-blind Clean-SFT failure bank used by the mechanism audit.
- `recovery_margin.jsonl`, containing one record for every
  `(selected ID, clean_sft/gkd_full/care_opd)` pair.
- Every failure-bank ID must remain a formal Clean-SFT failure, and every
  failure-bank/margin ID must exist in all formal rollout archives.

The JSON output preserves full input file hashes, request IDs and ID hashes.
Long text is truncated only in the report preview; a SHA-256 of the unmodified
text is retained. The preview applies conservative regex redaction for email,
mainland mobile numbers, national ID patterns, and long digit sequences. It is
not a general clinical de-identification guarantee; inputs are expected to be
the project's controlled synthetic benchmark records.

## Run

```bash
cd /opt/care-workspace/care_opd

export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
export MECHANISM_ROOT=/opt/care-workspace/care_opd/runs/mechanism_eval/qzbc_care_opd_v1
export FORMAL_ROOT=/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal
export OUTPUT_DIR="$MECHANISM_ROOT/qualitative_cases"

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  PYTHON_BIN="$PYTHON_BIN" \
  MECHANISM_ROOT="$MECHANISM_ROOT" \
  FORMAL_ROOT="$FORMAL_ROOT" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  SELECTION_SEED=20260813 \
  MAX_PREVIEW_CHARACTERS=1200 \
  bash /opt/care-workspace/care_opd/mechanism_experiments/qualitative_cases/run_qualitative_cases.sh \
  >"$LOG" 2>&1 </dev/null &

echo $! > "$OUTPUT_DIR/run.pid"
echo "$LOG"
tail -f "$LOG"
```

Outputs:

```text
qualitative_cases/
├── qualitative_cases.json  # full audit metadata and redacted previews
└── qualitative_cases.md    # presentation-ready six-slot panel
```

## Basic test

```bash
cd /opt/care-workspace/care_opd
/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python \
  -m unittest mechanism_experiments.qualitative_cases.test_qualitative_cases -v
```
