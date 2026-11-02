# GPU4/5 coordinator: local validation

Validation date: 2026-09-16. This record is a software check, not a GPU benchmark result.

## Checked

- All 46 CPU-only tests in `test_*.py` passed: 34 existing visual adapter tests and 12 GPU4/5 coordinator tests.
- Python compilation passed for `gpu45.py`, `evaluate.py`, and `build_bundle.py`.
- Bash syntax check passed for `run_visual_transfer_gpu45.sh`.
- The 4/5 schedule covers VLM-direct, Clean-SFT, GKD-Full, TROPD-Full, and CARE-OPD exactly once.
- A single parent lock covers environment inspection, manifest binding, observation generation, parallel answer workers, and scoring.
- Each CUDA worker is spawned fresh and selects one physical GPU before importing/initializing torch.
- Both devices' software/hardware metadata, code hashes and model identities are bound to the run manifest.
- All text methods read the same completed observation JSONL. Workers write disjoint method directories.
- Empty, truncated and invalid first outputs remain frozen on resume. No score-triggered regeneration was added.
- A failed observation stage blocks answer generation. A failed answer lane does not kill the other lane; both are joined and final scoring is blocked.
- No GPU occupancy check, reset, process termination, or original backfill modification was added.

## Deployment

Use a new V3 config/run root. Do not mix the new coordinator with an old single-GPU run manifest.
The V3 archive is built with `--visual-only`; it must contain only
`mechanism_experiments/visual_evidence_transfer/` source files, no model/data/results and no `tropd_backfill/` entries.

## Not verified locally

- Actual A800 GPU inference, GPU memory headroom, local server model paths, downloaded SLAKE assets and server dependency compatibility.
- No experiment accuracy, clinical performance or publication outcome is established by these unit tests.

Server launch and result paths are in `SERVER_COMMANDS_GPU45.md`.
