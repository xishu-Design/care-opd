# CARE-OPD additional mechanism experiment package

This package turns the remaining mechanism questions into executable assays.
None of the launchers starts expensive work unless the command explicitly
requests it.  The recommended evidence order is:

1. **Causal ablation**: six equal-example/equal-update one-round arms isolate
   localisation, retention, risk weighting, and trust masking; a separate
   fresh-vs-stale arm tests the persistent-failure cohort.
2. **Evidence counterfactual**: 200 matched evidence groups, each with four
   states, test whether a model answers with evidence, defers safely when the
   critical evidence state changes, recovers when evidence returns, and ignores
   unrelated history.
3. **Margin-to-behavior closure**: no training/inference; connects the existing
   teacher-forced recovery margin to actual verifier rescue on the same IDs.
4. **Deployment efficiency**: 200 result-blind requests on one GPU establish
   latency, throughput, memory, and the absence of extra deployment modules.
5. **Qualitative cases**: six reproducibly selected examples for the paper or
   defense; response text never controls which favorable example is selected.

## Entrypoints

Each experiment has its complete server command and input contract in its own
README:

- `causal_ablation/README.md`
- `evidence_counterfactual/README.md`
- `margin_behavior/README.md`
- `efficiency_audit/README.md`
- `qualitative_cases/README.md`

Run the zero-cost analyses first:

```bash
cd /opt/care-workspace/care_opd

bash mechanism_experiments/margin_behavior/run_margin_behavior.sh
bash mechanism_experiments/qualitative_cases/run_qualitative_cases.sh
```

Then run the four-state evidence assay and the efficiency audit.  The causal
ablation README deliberately creates a frozen plan before `--execute` is used.

## Publication figures

After the corresponding summaries exist:

```bash
cd /opt/care-workspace/care_opd

PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
FIG=/opt/care-workspace/care_opd/runs/mechanism_experiments/figures

# One-time preflight. If it fails, install the small optional plotting bundle:
# "$PY" -m pip install -r mechanism_experiments/requirements-plot.txt
"$PY" -c 'import matplotlib, numpy; print("plotting dependencies: PASS")'

"$PY" mechanism_experiments/plot_new_experiments.py \
  --causal-summary /opt/care-workspace/care_opd/runs/mechanism_ablation/eval/summary/summary.json \
  --counterfactual-summary /opt/care-workspace/care_opd/runs/mechanism_experiments/evidence_counterfactual_v1/results/counterfactual_summary.json \
  --efficiency-summary /opt/care-workspace/care_opd/runs/mechanism_eval/qzbc_care_opd_v1/efficiency_audit/efficiency_metrics.json \
  --output-dir "$FIG"
```

The plotter emits both vector PDF and 300-DPI PNG.  It refuses missing or
incomplete summaries rather than drawing placeholder bars.

## Claim boundaries

- Equal budget means equal rows and optimizer updates, not equal selected-token
  mass.  The character-matched random view is excluded from formal training.
- Fresh-vs-stale exact-ID matching conditions on failures present in both
  banks; it is not an all-request on-policy effect.
- Margin-to-behavior is an association, not a loss-component intervention.
- Evidence counterfactuals test synthetic QZBC evidence-state sensitivity, not
  clinical validity or Agent tool routing.
- Efficiency is a descriptive single-GPU deployment audit, not accuracy.

## Basic development check

```bash
cd /opt/care-workspace/care_opd
export PYTHONPATH=/opt/care-workspace/care_opd/src:/opt/care-workspace/care_opd
PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python

"$PY" -m py_compile mechanism_experiments/*.py mechanism_experiments/*/*.py
"$PY" -m unittest discover -s mechanism_experiments -p 'test_*.py' -v
```
