# CARE-OPD mechanism result templates

Fill these tables only from the scripts in this directory.  Do not infer a
causal component contribution from an untrained ablation.

## M1. Matched rescue and regression

| Slice | Model | Clean-SFT failures (n) | Rescued (n) | Rescue rate | Clean-SFT passes (n) | Regressed (n) | Regression rate |
|---|---|---:|---:|---:|---:|---:|---:|
| all perturbed | GKD-full | TBD | TBD | TBD | TBD | TBD | TBD |
| all perturbed | CARE-OPD | TBD | TBD | TBD | TBD | TBD | TBD |
| bad_history | CARE-OPD | TBD | TBD | TBD | TBD | TBD | TBD |
| source_outage | CARE-OPD | TBD | TBD | TBD | TBD | TBD | TBD |
| evidence_conflict | CARE-OPD | TBD | TBD | TBD | TBD | TBD | TBD |

## M2. Localized supervision audit

| Failure slice | Examples | Selected tokens / response tokens | Full-response fallback | Median pollution multiple | P90 pollution multiple |
|---|---:|---:|---:|---:|---:|
| all | TBD | TBD | TBD | TBD | TBD |
| invalid format | TBD | TBD | TBD | TBD | TBD |
| semantic failure | TBD | TBD | TBD | TBD | TBD |

`pollution multiple = full-response token count / localized selected-token count`.
This measures where supervision is placed, not the performance effect of that
choice.

## M3. Parameterized recovery preference

| Model | Examples | Full oracle-vs-rejected margin | Selected-field margin | Non-selected margin | Localization gap |
|---|---:|---:|---:|---:|---:|
| Clean-SFT | TBD | TBD | TBD | TBD | TBD |
| GKD-full | TBD | TBD | TBD | TBD | TBD |
| CARE-OPD | TBD | TBD | TBD | TBD | TBD |

All margins are length-normalized mean log-probability differences.  A positive
value means the checkpoint assigns greater conditional probability to the
host-verified oracle than to its rejected response on the same public prompt.

## M4. Online-round dynamics (descriptive)

| Round | Passed / 12,000 | Failure rate | Format failures | Semantic failures | Persistent from prior | Newly failed | Closed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | TBD | TBD | TBD | TBD | N/A | N/A | N/A |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Round-to-round changes are descriptive because the sampling seed changes with
the current checkpoint.  They do not isolate on-policy sampling causally.
