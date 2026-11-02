from __future__ import annotations

import hashlib


def stable_record_id(
    prefix: str,
    *,
    source_id: str,
    commit_sha: str,
    source_file: str,
    source_record_id: str,
) -> str:
    """Return a deterministic ID tied to one immutable source record."""
    raw = "\x1f".join((source_id, commit_sha, source_file, source_record_id))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20].upper()
    return f"{prefix}_{digest}"
