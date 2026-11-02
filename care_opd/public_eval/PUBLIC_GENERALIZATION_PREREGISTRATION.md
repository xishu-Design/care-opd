# Public generalization preregistration

Created before inspecting any TCMBench V2, MTCMB-Clean, optional CMB-Exam, or TCMEval controlled-perturbation predictions. TCMEval-PA official-clean results and the internal QZBC results had already been observed and are disclosed as retrospective inputs.

The later identity-bound QZBC rerun is therefore an artifact-attribution and
reproducibility repair, not a newly blind benchmark. Its 3,900-request
source-family-clean view is selected without model outputs from frozen
train/dev/test provenance keys, but is still labelled a retrospective
contamination-sensitivity analysis because the original 4,200-row scores were
already known. The original split is exact-entity-disjoint, not strictly
source-record-family-disjoint, and both views share the training generator,
templates, task families and distribution.

## Fixed model suite

The suite is frozen once and reused without changing checkpoint paths:

1. `frozen`: the suite entry intended to represent unmodified Qwen3.5-4B; raw-upstream identity is a separate lineage claim and is not inferred from this label;
2. `clean_sft`: the warm-start policy before recovery optimization;
3. `recovery_sft`;
4. `recovery_dpo`;
5. `gkd_full`;
6. `care_opd`.

All models use the same tokenizer/chat-template convention, `enable_thinking=False`, greedy decoding, the same precision, and the same item prompt within each benchmark. Model, dataset, prompt, generation and evaluator fingerprints are written before generation.

## Unified effectiveness and robustness endpoints

Every evaluated unit is reduced to an exact binary `task_success`. The public main table uses the same endpoints for every benchmark:

- `clean_task_success_rate`;
- `perturbed_task_success_rate`;
- `conditional_retention = P(perturbed_success | clean_success)`;
- `robustness_drop_pp = 100 * (clean_TSR - perturbed_TSR)`;
- `prediction_consistency` after mapping perturbed labels/order back to the canonical answer space;
- `worst_perturbation_tsr` when a benchmark has multiple perturbations.

For every paired track, **primary effectiveness** is perturbed TSR and **primary robustness** is conditional retention (equivalently, regression among clean-correct items is `1 - retention`). Clean TSR is the capability anchor. Robustness drop, prediction consistency and worst-perturbation TSR are diagnostic endpoints and cannot replace a failed primary endpoint. Every result also reports the counts `clean-correct -> perturbed-wrong` and `clean-wrong -> perturbed-correct`, because consistency alone can reward a model that is wrong twice.

Official scores and project-derived robustness scores are never mixed under one label or numeric column. Deterministic option/order/format transformations are named `controlled_robustness_v1`; the transform manifest, seed, mappings and hash are frozen before model predictions are inspected. A derived clean/perturbed pair map for public subtasks is likewise labeled Derived even when both source files are official.

No cross-benchmark average is a primary result or generalization gate: benchmark difficulty and perturbation severity are not commensurate. An equal-weight descriptive macro may appear only in an appendix with every component shown; it is not used for ranking, significance, or the generalization claim.

## Existing benchmark: TCMEval-PA robustness follow-up

The 328 clean `official` predictions were already observed, so their ranking is not retrospectively preregistered. Before inspecting any perturbed predictions, we fix a `controlled_robustness_v1` follow-up using deterministic option permutation with gold-label remapping. Both members of each robustness pair use the same label-blind prompt; the clean label-blind member is rerun rather than paired with the earlier official prompt. Its two primary endpoints are perturbed TSR and conditional retention. Robustness drop and canonical prediction consistency are diagnostic. The original official score remains a separate effectiveness result and is never paired with the new prompt condition.

For the new controlled track, the primary comparison is `care_opd - clean_sft` and `care_opd - gkd_full` is secondary.  The pre-prediction joint preservation margin is `-2.0` percentage points for both perturbed TSR and conditional retention.  Using the fixed 10,000-replicate normalized-stem-cluster bootstrap (seed 20260808), both 95% CI lower bounds must exceed `-0.02`; passing means preserved/non-inferior under this controlled perturbation, not superior.

## Optional appendix: CMB-Exam

Purpose: optional broad Chinese medical knowledge-retention check outside the QZBC output contract. It is not one of the two additional primary public benchmarks because its multiple-choice format is less aligned with the recovery claim and overlaps in role with TCMEval-PA.

- Dataset: official CMB-Exam test, 11,200 questions across 6 major and 28 minor categories.
- Primary metric: option-set exact accuracy on all 11,200 official items.
- Contamination-sensitive companion view: the same metric after removing every item flagged as an exact normalized stem match by the precomputed, result-blind QZBC overlap audit.
- Secondary metrics: single/multiple exact; major- and minor-category micro accuracy; strict-format rate.
- Controlled robustness track: a fixed category-stratified, overlap-clean subset with deterministic option permutation and label remapping. Clean predictions are reused; perturbed predictions are mapped to canonical answer content before exact scoring and consistency analysis.
- Primary comparison for method attribution: `care_opd - clean_sft` on the same items.
- Preservation gate: the lower bound of the normalized-stem-cluster paired bootstrap 95% confidence interval for `care_opd - clean_sft` must exceed `-1.0` percentage point on the exact-overlap-excluded set (2,000 replicates, seed 20260808). This margin is fixed before predictions and represents the largest tolerated absolute broad-knowledge loss.
- Superiority is claimed only if a predeclared paired test rejects equality; passing the preservation gate alone is described as non-inferior/preserved, not superior.

Both official-full and exact-overlap-excluded results are retained. The latter is not called contamination-free: exact normalization cannot rule out paraphrases, common source banks, or knowledge-level overlap.

## Additional primary benchmark 1: TCMBench V2 robustness track

Purpose: test whether the learned recovery behavior transfers to public clinical-reasoning perturbations.

- Primary units: public clean/perturbed pairs that can be joined by benchmark-provided identifiers and verified content, especially `SDT -> SDT_shuffle` and `SDT -> SDT_reverse`.
- Primary effectiveness metric: exact answer accuracy on perturbed items (`perturbed_task_success_rate`).
- Primary robustness metric: the fraction of clean-correct items that remain correct (`conditional_retention`); the corresponding clean-correct to perturbed-wrong regression count is mandatory.
- Diagnostic metrics: clean-to-perturbed accuracy drop, canonical prediction consistency, clean-wrong to perturbed-correct count, task-level micro/macro accuracy, strict-format rate, and hard-task slices (`DDST_hard`, `SDDT_hard`).
- Primary comparison: `care_opd` versus `clean_sft`; `care_opd` versus `gkd_full` is the strongest distillation control.
- A pair is included only when the evaluator can verify an unambiguous public mapping. Unpaired hard items remain in task accuracy but not in paired-retention statistics.
- The official per-subtask accuracies and the project-audited pair map are reported separately. File names such as `SDT_reverse` or `SDT_shuffle` do not by themselves prove a one-to-one, label-preserving pair.
- Joint preservation gate: before predictions, the lock fixes a `-2.0` percentage-point non-inferiority margin for both CARE-minus-clean-SFT perturbed TSR and conditional retention. Both paired confidence-interval lower bounds must exceed their margin; drop or consistency cannot rescue a failure. The margin is an explicit maximum tolerated absolute loss, not an estimate selected from results.

If the evaluator uses a shorter answer-only prompt instead of the repository's analysis-generating prompt, results are named `answer_only_v1` and are not compared numerically with the repository's published CoT scores.

## Additional primary benchmark 2: MTCMB provenance-clean Clean9

Purpose: test multi-task TCM transfer after removing the public task families directly consumed by formal QZBC training.

- MTCMB-Full contains 7,100 records, but 1,062 unique records from `TCM-ED-A` and `TCM-ED-B` occur in the formal training provenance. Full-set results are therefore diagnostic, not independent-generalization evidence.
- The primary filtered view removes both complete ED task families and the 50-item `TCM-SE-A` task whose official protocol requires a remote subjective GLM judge. The fixed Clean9 view contains 1,050 records across FT, TCMeEE, CHGD, LitData, MSDD, Diagnosis, PR, FRD and SE-B.
- Official task metrics remain task-specific (accuracy/set metrics/ROUGE/BLEU/BERTScore as applicable). They are reported separately; no fabricated sample-weighted cross-task “overall score” is used.
- Only `TCM-SE-B` provides a strict, locally verifiable 50-item option-permutation pair track inside Clean9. Its primary effectiveness endpoint is perturbed TSR and its primary robustness endpoint is conditional retention. The small sample size is disclosed and confidence intervals are mandatory.
- The primary comparison is `care_opd - clean_sft` (`care_opd - gkd_full` secondary).  Before prediction, the joint margin is fixed at `-5.0` percentage points for both perturbed TSR and conditional retention.  A paired item bootstrap with 10,000 replicates and seed 20260809 supplies the difference intervals; both lower bounds must exceed `-0.05`.  Because n=50 is small, failure to pass is reported as uncertainty rather than rescued by a favorable point estimate.
- The broader item-level overlap-clean view is retained as a diagnostic, but because ED-A/ED-B family exposure remains possible, it is not called provenance-clean.

## Overlap policy

- `MTCMB-Full` and the ED-A/ED-B task families are excluded from independent-generalization claims. Only the explicitly named provenance-clean Clean9 view is used as supplementary public transfer evidence.
- CMB-Exam and TCMBench are reported in both full and exact-overlap-excluded forms; the latter is not described as contamination-free.
- Exact/same-item exclusions are frozen before predictions are examined.
- Semantic knowledge overlap with shared medical textbooks is expected and is not called data leakage; direct item/provenance overlap is disclosed separately.

## Statistical and reporting policy

- No benchmark macro is used as a primary endpoint, method rank, or generalization gate. Any appendix-only descriptive macro shows every component and never mixes QZBC recovery or unsafe-green into public knowledge/robustness.
- All paired comparisons use item-level predictions. Repeated stems/pairs are clustered where applicable.
- Report numerator, denominator, percentage and 95% confidence interval together.
- Paired tracks keep exactly the two declared primary constructs (perturbed TSR and conditional retention); diagnostic slices go in a separate table.
- A public benchmark can support retention or task transfer, but only a recovery-shaped public perturbation benchmark can support recovery generalization.
- Final end-to-end claims additionally require a fixed QZBC Agent A/B in which only the base model changes.

## Immutable preregistration lock

Before any new controlled/public prediction is generated, create `public_eval/PREREGISTRATION_LOCK.json` from the already frozen six-model `model_suite.json`. This repository implements a static protocol/model lock: it hashes every listed evaluator, schema, data/exclusion manifest, model-loading helper and the exact formal launcher, and embeds the six model fingerprints. Each run then binds the dynamic dataset, prompt, generation, transform/pair-map, runtime-library and model identities in its own run fingerprint. The lock SHA-256 is copied into every run manifest. Later changes require a new versioned lock and output root; the old lock is never overwritten. A local lock is not, by itself, an independent timestamp service, so “externally auditable preregistration” additionally requires publishing its hash to a non-rewritable external record before predictions.
