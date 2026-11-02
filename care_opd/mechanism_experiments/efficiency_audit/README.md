# Deployment efficiency audit

This experiment measures `clean_sft`, `gkd_full`, and `care_opd` sequentially
on one GPU. It uses the same 200 result-blind QZBC family-clean requests (50 per
variant), request order, no-thinking chat template, greedy decoding, bfloat16,
batch size one, and 1,200-token cap. Five fixed requests warm up each model;
warm-up observations are excluded.

The measured inference path loads one local Hugging Face checkpoint and calls
`model.generate` directly. It does not instantiate an Agent loop, verifier,
teacher, retriever, or tool. Thus it tests the paper claim that CARE-OPD changes
training but adds no deployment component.

## Run on physical GPU 4

```bash
cd /opt/care-workspace/care_opd

LOG=/opt/care-workspace/care_opd/runs/mechanism_eval/logs/efficiency_$(date +%Y%m%d_%H%M%S)_gpu4.log
mkdir -p "$(dirname "$LOG")"

nohup env PHYSICAL_GPU=4 \
  bash mechanism_experiments/efficiency_audit/run_gpu.sh \
  >"$LOG" 2>&1 </dev/null &

echo $!
echo "$LOG"
tail -f "$LOG"
```

The launcher intentionally does not inspect or reject other GPU PIDs. Set
`PHYSICAL_GPU`, `MODEL_SUITE`, `DATA_AUDIT`, `REQUESTS`, or `OUTPUT_DIR` to
override a path. Completed per-model results are validated and skipped on
restart; append `--rerun` to measure them again under the same contract.

## Outputs

```text
efficiency_audit/
  sample_manifest.json
  clean_sft/{per_request.jsonl,metrics.json,COMPLETE.json}
  gkd_full/{per_request.jsonl,metrics.json,COMPLETE.json}
  care_opd/{per_request.jsonl,metrics.json,COMPLETE.json}
  efficiency_metrics.json
  efficiency_metrics.csv
  efficiency_audit.md
  COMPLETE.json
```

The report puts latency next to output length and generated-token throughput,
because a model that generates longer responses can have greater wall-clock
latency without a slower token-generation kernel. Peak allocated memory includes
resident model parameters; peak reserved memory describes the CUDA allocator.

## Basic CPU test

```bash
python -m unittest mechanism_experiments/efficiency_audit/test_efficiency_audit.py
```
