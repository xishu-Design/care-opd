# Source release notes

## Scope

This snapshot collects the locally available GuardTCM and CARE-OPD implementation,
training and baseline launchers, TrOPD evaluation backfill, mechanism analyses,
public/external evaluation code, agent frontend/backend, and separate visual
evidence evaluation utilities. It does not certify that a remote server's source
tree is identical to the local snapshot.

## Public-copy changes

- Normalize line endings to UTF-8/LF and remove BOMs.
- Replace personal absolute machine roots with generic example roots while
  retaining the original directory relationships.
- Replace example environment-file secrets with explicit local setup placeholders.
- Exclude private collaboration attestation documents and unrelated literature
  search notes.
- Keep original research folders unchanged.

## Excluded material

Actual `.env` files, private keys, raw or processed medical datasets, evaluation
JSON/JSONL payloads, private labels and reviewer records, source corpora, generated
model responses, databases and graph/vector stores, checkpoints and model weights,
logs, archives, installed dependencies, and draft or final submission documents
are not part of this source-only release.

## Validation scope

Python syntax is checked for the released Python sources. The release is scanned
for common API-token/private-key formats and known personal machine-root strings.
These checks do not constitute a security audit or validate clinical content.
The CPU-only TrOPD protocol test file passes all seven tests. Across the wider
16-test protocol/launcher suite, nine checks pass and seven cannot execute because
the available Windows Bash runtime fails to start. These environment-blocked
launcher checks are not reported as passing. Full CUDA training, frontend builds,
and data-dependent end-to-end evaluation have not been rerun for this release.
