# GuardTCM and CARE-OPD

Source code for **GuardTCM**, a contract-governed medical agent, and
**CARE-OPD**, responsibility-aware on-policy distillation.

GuardTCM separates runtime authorization from response generation. Its controller
governs evidence access, tools, role-scoped memory, and response release for
physician, patient, and student workflows. CARE-OPD turns typed verification
failures into field-aligned token supervision during post-training.

## Repository layout

| Directory | Contents |
| --- | --- |
| `care_opd/src/care_opd/` | Contracts, data compilation, rollout, Repair/Protect/Oracle losses, training, export and evaluation |
| `care_opd/configs/` | Training and baseline configurations |
| `care_opd/scripts/` | Data-building, training, evaluation and deployment launchers |
| `care_opd/tests/` | Regression tests |
| `care_opd/public_eval/` | Public benchmark and grouped controlled-evaluation implementations |
| `care_opd/mechanism_eval/` | Localization, recovery, failure and round-dynamics analyses |
| `care_opd/mechanism_experiments/` | Ablations, token-matched controls, TrOPD backfill, external and visual-evidence studies |
| `care_opd/competition_evidence/` | Execution, retrieval, memory and role-workflow evaluation utilities |
| `qzbc_source/qzbc-main/` | GuardTCM backend, frontend, migrations, service integration and launchers |
| `care_opd_multimedia_boundary_v1/` | Separate multimedia evidence-boundary evaluation |
| `care_opd_native_layout_probe_v1/` | Native-layout evidence probing and scoring |
| `paper/figures/` | Paper figure-generation source, including historical revisions |
| `release_tools/` | Existing model-packaging and public-upload checking utilities |

## Training code

Use a Linux CUDA environment for model training. Python 3.12 is recommended when
using both the training package and the agent backend. Consult the component
dependency files before choosing PyTorch and CUDA builds.

```bash
git clone https://github.com/xishu-Design/care-opd.git
cd care-opd
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e './care_opd[dev]'
cd care_opd
export QZBC_ROOT="$(cd ../qzbc_source/qzbc-main && pwd)"
export STUDENT_MODEL=Qwen/Qwen3.5-4B
export TEACHER_MODEL=Qwen/Qwen3.5-9B
```

After supplying the required approved knowledge assets, the main entry points are:

```bash
bash scripts/01_build_formal_data.sh
bash scripts/03_train_care_opd.sh
```

`scripts/run_all_formal.sh` coordinates the full pipeline. Baseline training is in
`scripts/02_train_baselines.sh`. Training and evaluation settings, rollout
parallelism, dataset roots and output locations are configurable through the
existing configuration files and environment variables. Historical server-command
documents use generic example roots; replace them with your own installation
paths rather than copying them unchanged.

### Learning objectives

- **Repair** distills teacher distributions on trust-filtered responsibility tokens.
- **Protect** anchors the complement of the **pre-trust** responsibility region,
  together with clean controls, to the frozen round-start reference.
- **Oracle** supervises selected fields of independently constructed,
  verifier-validated targets.

Student responses are resampled between rounds. Deployment uses the merged
student checkpoint, without the training-time teacher or reference models.

## GuardTCM application

The agent is under `qzbc_source/qzbc-main/`. Read its component README files for
backend, frontend, LightRAG, memory and graph-service setup. The backend requires
Python 3.12 or later; the frontend uses React, TypeScript and Vite.

Use the included `.env.example` files as configuration templates, generate unique
secrets locally, and bind development services to loopback. Fixed demo accounts
and development defaults are for isolated demonstrations, not production access.
Clinical use is outside the scope of this research release.

## Data, weights and reproducibility

This is a **source-only release**. It does not publish patient records, private
evaluation questions or gold labels, third-party knowledge corpora, generated
responses, reviewer materials, runtime databases, credentials, or model weights.
The historical formal compiled dataset and server checkpoints are not included.
Dataset builders and scoring code are provided; reproducing data-dependent
experiments requires the corresponding authorized inputs and model checkpoints.
Some regression tests also require those local assets.

Paper plotting scripts may require separately supplied aggregate tables. The
original per-request harness for some fixed-budget tool, retrieval and memory
results has not been located in this local snapshot; plotting source alone is
not an end-to-end reproduction of those experiments.

No result files or manuscript text have been added to imply a new experiment.
The visual-evidence studies are separate evaluations, not a claim that the
deployed student directly diagnoses raw medical images.

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for scope, checks and exclusions, and
[CODE_AND_DATA_LICENSES.md](CODE_AND_DATA_LICENSES.md) for licensing boundaries.

## Source integrity

`SOURCE_MANIFEST.json` records the released file paths, byte counts and SHA-256
hashes. Source files retain their component layout. Public-copy normalization
removes personal machine roots and replaces example environment secrets; it does
not change the Repair/Protect/Oracle objectives, model settings or evaluation
denominators.
