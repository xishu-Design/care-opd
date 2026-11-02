# CARE-OPD formal effectiveness and robustness evaluation

This directory produces one auditable result package for the six frozen model
checkpoints.  The primary table contains:

1. an identity-bound rerun of the internal QZBC grouped clean/fault pairs;
2. TCMEval-PA official effectiveness and label-blind controlled robustness;
3. TCMBench V2 audited SDT→reverse/shuffle robustness (additional public set 1);
4. MTCMB provenance-clean Clean9, with its SE-B paired track in the unified
   robustness table (additional public set 2).

CMB-Exam is a broad Chinese medical knowledge-retention appendix. It is not
required for the two-public-benchmark claim, but the four-GPU command below
sets `RUN_CMB=1` and evaluates it as requested. No cross-benchmark average is
used for ranking.

## Unified main endpoints

For every valid clean/perturbed pair, task success is binary.  The primary
effectiveness endpoint is perturbed task-success rate (TSR); the primary
robustness endpoint is conditional retention among clean-correct items.
Clean TSR, regression, drop, canonical prediction consistency, worst-family
TSR and the two flip counts are reported as context/diagnostics.  A parse
failure is always inconsistent, including when both outputs fail parsing.

Official scores and project-derived perturbation scores remain separate.  In
particular, MTCMB-Full is diagnostic because 1,062 unique ED-A/ED-B records
occur in formal training provenance.  Clean9 removes both complete task
families and the remote-judge SE-A task; it does not pretend that the full set
is contamination-free.

## One-time server preparation

Use the existing isolated baseline environment; this does not modify the
training environment.

```bash
cd /opt/care-workspace

git clone https://github.com/ywjawmw/TCMBench.git TCMBench
git -C /opt/care-workspace/TCMBench checkout 7e2d2326ff79da186c6fccaafa374cf1f77ac968

git clone https://github.com/Wayyuanyuan/MTCMB.git MTCMB
git -C /opt/care-workspace/MTCMB checkout faffd813c67fe012cef74277a23270d5dd6db9b8

export EVAL_PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
"$EVAL_PY" -m pip install -r \
  /opt/care-workspace/care_opd/public_eval/mtcmb_clean/requirements-eval.txt

# MTCMB scoring is network-free during the formal run. Download this once if
# the directory is not already present.
"$EVAL_PY" -c 'from huggingface_hub import snapshot_download; snapshot_download("google-bert/bert-base-chinese", local_dir="/opt/care-workspace/models/bert-base-chinese")'
```

If the TCMBench current checkout no longer exposes its data directory, create
a detached worktree at the same pinned commit and set `TCMBENCH_REPO` to it.
Do not redistribute TCMBench data in the competition code package without
clarifying the upstream data license.

## Complete formal run on physical GPUs 4, 5, 6 and 7

The launcher creates/validates one static protocol/model lock before any new
prediction, reruns the internal QZBC set with checkpoint and split receipts,
runs the three required public tracks sequentially, and finally builds the main table. Each
model uses four spawned workers on logical `cuda:0,cuda:1,cuda:2,cuda:3`
under physical GPUs 4, 5, 6 and 7. Completed models resume only after full ID, contract and artifact
hash validation.  Before creating the lock it also rehashes all six checkpoints,
checks that CARE and baseline branches share the same warm start, and validates
the saved TCMEval run contracts.  A locally consistent but externally unattested
`frozen` origin is reported as `NOT_VERIFIED`, without blocking the relative
CARE-versus-baseline experiment.

The old 4,200-row QZBC rollouts do not embed checkpoint fingerprints or the
formal split manifest, so they are kept only as an optional legacy diagnostic.
The formal main table uses the new rerun. Its 3,900-request family-clean view is
an in-domain retrospective sensitivity analysis, not an external/OOD benchmark.

Use a new versioned lock/output root if any protocol byte changed after an
earlier attempt; never overwrite an old formal lock.

```bash
cd /opt/care-workspace/care_opd

mkdir -p /opt/care-workspace/care_opd/runs/public_eval/logs
LOG=/opt/care-workspace/care_opd/runs/public_eval/logs/generalization_$(date +%Y%m%d_%H%M%S)_gpu4_5_6_7.log

nohup env \
  PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python \
  SOURCE_SUITE=/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/model_suite.json \
  PREREGISTRATION_LOCK=/opt/care-workspace/care_opd/public_eval/PREREGISTRATION_LOCK_20260809_4gpu_v1.json \
  TCMBENCH_REPO=/opt/care-workspace/TCMBench \
  MTCMB_REPO=/opt/care-workspace/MTCMB \
  BERT_MODEL=/opt/care-workspace/models/bert-base-chinese \
  PHYSICAL_GPUS=4,5,6,7 \
  NUM_WORKERS=4 \
  DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
  QZBC_GROUPED_OUT=/opt/care-workspace/care_opd/runs/public_eval/qzbc_grouped_identity_4gpu_v1 \
  TCMEVAL_CONTROLLED_OUT=/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_controlled_4gpu_v1 \
  TCMBENCH_OUT=/opt/care-workspace/care_opd/runs/public_eval/tcmbench_v2_4gpu_v1 \
  MTCMB_OUT=/opt/care-workspace/care_opd/runs/public_eval/mtcmb_clean_4gpu_v1 \
  CMB_OUT=/opt/care-workspace/care_opd/runs/public_eval/cmb_exam_4gpu_v1 \
  MAIN_TABLE_OUT=/opt/care-workspace/care_opd/runs/public_eval/generalization_main_table_4gpu_v1 \
  RUN_CMB=1 \
  bash public_eval/run_generalization_gpu4_5.sh \
  > "$LOG" 2>&1 </dev/null &

echo $! > /opt/care-workspace/care_opd/runs/public_eval/generalization_suite.pid
printf '%s\n' "$LOG" > /opt/care-workspace/care_opd/runs/public_eval/LATEST_GENERALIZATION_LOG.txt
echo "PID=$!"
echo "LOG=$LOG"
tail -f "$LOG"
```

Set `RUN_CMB=0` only when the optional 11,200-item CMB appendix should be
skipped. The command above includes it.

## Final outputs

```text
/opt/care-workspace/care_opd/runs/public_eval/
  qzbc_grouped_identity_4gpu_v1/formal/suite_summary.{json,csv,md}
  internal_qzbc_legacy_v1/suite_summary.{json,csv,md}  # optional diagnostic
  tcmeval_pa_controlled_4gpu_v1/suite_summary.{json,csv}
  tcmbench_v2_4gpu_v1/answer_only/suite_summary.{json,csv}
  mtcmb_clean_4gpu_v1/official/suite_summary.{json,csv,md}
  cmb_exam_4gpu_v1/official/suite_summary.{json,csv,md}
  generalization_main_table_4gpu_v1/
    main_table.md
    main_table_long.csv
    main_table_effectiveness.csv
    main_table_robustness.csv
    noninferiority_gates.csv
    noninferiority_gates.json
    main_table.json
```

`frozen` is the intended untrained HF checkpoint entry, but the label does not
prove upstream identity.  A matching locally supplied immutable-revision file
is reported only as `REFERENCE_MATCHED`; use the lineage audit and independent
provenance before calling it the raw initial base.  OPD method attribution is
primarily CARE-OPD versus the identical clean-SFT warm start.
