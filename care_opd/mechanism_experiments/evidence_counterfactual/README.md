# Evidence-state counterfactual assay

This experiment asks whether a frozen checkpoint changes its behavior when the
same public request moves through a matched evidence-state quartet.  It reuses
the production QZBC chat builder and `care_opd.qzbc_rollout.assess_response`;
it does not introduce an alternative parser or judge.

## Design

The builder result-blind hashes clean held-out request IDs and selects 50
unique groups within each of four critical families by default.  The same
clean source row may be matched across families so that family contrasts are
not confounded by disjoint case composition:

- `source_outage`: withdraw the available evidence and record a controlled
  high-risk source error;
- `evidence_conflict`: add an explicitly conflicting and unverified record;
- `bad_history`: add a trusted-looking but incorrect prior assistant answer;
- `unverified_source`: preserve source content but mark every source-bearing
  record unverified, making it non-citable.

Every group contains four states:

- **A / evidence_present**: untouched clean request;
- **B / critical_perturbation**: family-specific evidence hazard;
- **C / evidence_restored**: the same evidence-bearing fields as A, with a
  controlled history marker saying that a prior source outage is now resolved
  and the model must use only the current Evidence Packet;
- **D / irrelevant_history**: A plus a semantically unrelated trusted history
  turn; all evidence-bearing packet fields remain equal to A.

With 50 groups per family this produces 200 matched groups and 800 generation
requests per model.  A/C/D require the original evidence-grounded answer.  B
requires safe deferral for outage/conflict/unverified-source and ignoring the
bad history for `bad_history`.  Each oracle is verified before the bank is
written.  Selection never reads model outputs.

## Full run on physical GPUs 4, 5, and 6

The three checkpoints run concurrently, one per GPU.  The launcher deliberately
does not inspect or refuse occupied GPUs; assign GPUs that are actually safe to
use.

```bash
cd /opt/care-workspace/care_opd

export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
export PHYSICAL_GPUS=4,5,6
export GROUPS_PER_FAMILY=50
export OUT_ROOT=/opt/care-workspace/care_opd/runs/mechanism_experiments/evidence_counterfactual_v1
export SOURCE_SUITE=/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1/model_suite.json

bash mechanism_experiments/evidence_counterfactual/launch_nohup.sh
```

The command prints the driver PID and log.  Inspect progress with:

```bash
tail -f "$(ls -1t "$OUT_ROOT"/logs/driver_*.log | head -n 1)"
```

Model-specific logs are:

```text
$OUT_ROOT/logs/clean_sft_gpu4.log
$OUT_ROOT/logs/gkd_full_gpu5.log
$OUT_ROOT/logs/care_opd_gpu6.log
```

Completed rollouts are skipped on restart.  If model paths differ, set:

```bash
export CLEAN_SFT_MODEL=/path/to/clean_sft
export GKD_MODEL=/path/to/gkd_full
export CARE_MODEL=/path/to/care_opd
```

Before launching, the script binds all three requested paths to the frozen
formal `model_suite.json` labels and records its SHA-256 and suite fingerprint
in the bank manifest.  It does not re-hash multi-gigabyte model weights.

## Outputs

```text
$OUT_ROOT/
  bank/bank_manifest.json
  bank/counterfactual_requests.jsonl
  rollouts/{clean_sft,gkd_full,care_opd}/qzbc_rollouts.jsonl
  results/counterfactual_scored.jsonl
  results/counterfactual_summary.json
  results/counterfactual_summary.csv
  results/counterfactual_paired_bootstrap.csv
  results/counterfactual_summary.md
```

Primary metrics are:

- evidence-present TSR;
- critical-state safety TSR;
- evidence-restored TSR;
- irrelevant-history TSR;
- joint four-state TSR (all A/B/C/D pass within a matched group);
- unsafe-green rate;
- restoration and irrelevant-history retention conditional on A passing.

The scorer also reports CARE-minus-Clean-SFT and CARE-minus-GKD deltas with
paired group-bootstrap 95% confidence intervals (10,000 fixed-seed resamples)
for critical safety, joint four-state success, restoration retention, and
irrelevant-history retention.  Resampling uses matched groups rather than
individual A/B/C/D requests.

Because generation is greedy, C deliberately differs from A in conversation
history.  Making C byte-identical to A would force an identical decode and
would not measure recovery.  Builder assertions guarantee that C and D alter
only `conversation_context`, never the evidence-bearing packet fields.

The joint score is the most compact demonstration that a checkpoint neither
answers blindly nor refuses blindly: it must answer with evidence, safely
change behavior under the critical state, recover after evidence returns, and
remain stable under irrelevant history.

## Basic server test

```bash
cd /opt/care-workspace/care_opd
export PYTHONPATH=/opt/care-workspace/care_opd/src:/opt/care-workspace/care_opd
export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python

"$PYTHON_BIN" -m py_compile \
  mechanism_experiments/evidence_counterfactual/*.py
"$PYTHON_BIN" -m unittest \
  mechanism_experiments.evidence_counterfactual.test_evidence_counterfactual -v
```

## Claim boundary

This is a controlled synthetic state-sensitivity assay at the
`chat messages + Evidence Packet -> GeneratedResponse JSON` boundary.  It is
not clinical evidence, does not test host tool routing, and by itself does not
attribute gains causally to one training-loss component.
