# CARE-OPD mechanism experiments

This directory contains a small, post-training evidence package for the
mechanism claim behind CARE-OPD.  It deliberately reuses frozen checkpoints
and frozen QZBC outputs: no script here trains a model or executes the QZBC
Agent graph.

## What is tested

| Experiment | Question | Cost | Main output | Claim boundary |
|---|---|---|---|---|
| M1. Matched rescue | Among requests failed by the shared Clean-SFT start, how many are rescued by each checkpoint, and how many Clean-SFT successes regress? | Existing rollouts only | `failure_decomposition/` | Direct evidence for recovery behavior; not an isolated loss ablation |
| M2. Localized supervision | How much of a rejected response is selected by the verifier-derived repair fields? | Tokenizer only | `localization/` | Establishes supervision locality/sparsity, not its causal accuracy gain |
| M3. Recovery preference | Does the final checkpoint assign higher conditional likelihood to the verified oracle than to its rejected response? | At most 200 teacher-forced forward examples by default | `recovery_margin/` | Evidence that the preference is in model parameters; CARE-vs-GKD identifies the complete CARE objective package, not one coefficient |
| M4. Round dynamics | Which failures disappear, persist, or newly appear over the three on-policy rounds? | Existing training archives only | `round_dynamics/` | Descriptive curriculum evidence; changing checkpoints/seeds prevent a standalone causal claim |

The tested model boundary is:

`chat messages + Evidence Packet -> GeneratedResponse JSON`

The model does not choose LightRAG, Graphiti or Mem0 actions in these tests.
Agent routing, rule post-processing and the Safety Guard are outside this
mechanism package.

## Inputs expected on the server

Defaults match the completed formal run:

```text
/opt/care-workspace/care_opd/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl
/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal/
/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/model_suite.json
/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/round_01..round_03/
/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/warm_start/merged
/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/gkd_full/final_model
/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model
```

M1 uses the common 3,900-row family-clean subset embedded in the 4,200 formal
requests.  Its strata are frozen from Clean-SFT failures, so CARE results do
not determine which cases enter a failure slice.

## 1. Run the CPU/tokenizer diagnostics

These steps build the fixed Clean-SFT failure bank, compute rescue/regression,
audit repair-field token coverage, and summarize the three saved rounds.

```bash
cd /opt/care-workspace/care_opd

export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
export OUT_ROOT=/opt/care-workspace/care_opd/runs/mechanism_eval/qzbc_care_opd_v1

mkdir -p "$OUT_ROOT/logs"
CPU_LOG="$OUT_ROOT/logs/cpu_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  PYTHON_BIN="$PYTHON_BIN" \
  OUT_ROOT="$OUT_ROOT" \
  bash /opt/care-workspace/care_opd/mechanism_eval/run_cpu_mechanisms.sh \
  >"$CPU_LOG" 2>&1 </dev/null &

echo $! > "$OUT_ROOT/cpu.pid"
echo "$CPU_LOG"
```

Monitor it with:

```bash
tail -f "$CPU_LOG"
```

## 2. Run the bounded GPU forward diagnostic

This performs no decoding and no optimization.  It selects at most 200
failure-bank examples by a fixed result-blind hash, then loads Clean-SFT,
GKD-full and CARE-OPD sequentially on one GPU.  There is intentionally no GPU
occupancy refusal in the launcher.

```bash
cd /opt/care-workspace/care_opd

export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
export OUT_ROOT=/opt/care-workspace/care_opd/runs/mechanism_eval/qzbc_care_opd_v1

mkdir -p "$OUT_ROOT/logs"
MARGIN_LOG="$OUT_ROOT/logs/margin_$(date +%Y%m%d_%H%M%S)_gpu4.log"

nohup env \
  PYTHON_BIN="$PYTHON_BIN" \
  OUT_ROOT="$OUT_ROOT" \
  PHYSICAL_GPU=4 \
  SOURCE_SUITE=/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/model_suite.json \
  MAX_EXAMPLES=200 \
  SELECTION_SEED=20260813 \
  bash /opt/care-workspace/care_opd/mechanism_eval/run_margin_gpu.sh \
  >"$MARGIN_LOG" 2>&1 </dev/null &

echo $! > "$OUT_ROOT/margin.pid"
echo "$MARGIN_LOG"
```

Monitor it with:

```bash
tail -f "$MARGIN_LOG"
```

## Outputs

```text
runs/mechanism_eval/qzbc_care_opd_v1/
├── heldout_failure_bank.jsonl
├── heldout_failure_bank.jsonl.manifest.json
├── failure_decomposition/
│   ├── failure_decomposition.json
│   ├── failure_decomposition.csv
│   └── failure_decomposition.md
├── localization/
│   ├── localization_analysis.json
│   ├── localization_analysis.csv
│   └── localization_analysis.md
├── round_dynamics/
│   ├── round_dynamics.json
│   ├── round_dynamics.csv
│   └── round_dynamics.md
└── recovery_margin/
    ├── recovery_margin.jsonl
    ├── run_manifest.json
    ├── recovery_margin_summary.json
    ├── recovery_margin_summary.csv
    └── recovery_margin_summary.md
```

Use [RESULT_TABLES.md](RESULT_TABLES.md) to transfer values into the report.
Do not fill a value from a console estimate when the corresponding JSON/CSV is
absent.

## Reading the evidence

- Strong M1 evidence is high CARE rescue with low regression, particularly on
  `bad_history`, `source_outage`, and `evidence_conflict`.
- Strong M2 evidence is low selected-token coverage and a large
  full-response/selected-token multiple, with a low full-response fallback
  rate.  This says CARE concentrates supervision; it does not prove the
  concentration alone caused the final gain.
- Strong M3 evidence is a larger positive verified-oracle versus rejected
  margin for CARE than Clean-SFT, ideally also larger than GKD in selected
  fields without an equally large outside-field shift.
- M4 may show a shrinking failure set, but inspect format and semantic failures
  separately.  A total reduction driven only by JSON formatting must not be
  described as broad semantic recovery.

The strongest defensible conclusion is about the joint CARE-OPD mechanism:
on-policy failed states + verifier-localized fields + risk/trust/retention
constraints.  Without newly trained `w/o span`, `w/o trust`, or `w/o
retention` checkpoints, these diagnostics do not identify the independent
causal contribution of a single component.

## Basic server check

```bash
cd /opt/care-workspace/care_opd
export PYTHONPATH=/opt/care-workspace/care_opd/src:/opt/care-workspace/care_opd
export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python

"$PYTHON_BIN" -m py_compile mechanism_eval/*.py
"$PYTHON_BIN" -m pytest \
  mechanism_eval/test_analyze_failure_decomposition.py \
  mechanism_eval/test_analyze_round_dynamics.py \
  mechanism_eval/test_recovery_margin.py \
  tests/test_analyze_localization.py \
  -q
```
