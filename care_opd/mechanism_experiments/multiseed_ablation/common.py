"""Shared strict I/O helpers for the multi-seed experiment package."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


TRAIN_SUITE_SCHEMA = "care_opd.multiseed_ablation_plan.v1"
EVAL_SUITE_SCHEMA = "care_opd.multiseed_ablation_eval_plan.v1"
AGGREGATE_SCHEMA = "care_opd.multiseed_ablation_aggregate.v1"
QUEUE_STATUS_SCHEMA = "care_opd.multiseed_ablation_queue_status.v1"
CONTRASTS_SCHEMA = "care_opd.multiseed_ablation_contrasts.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_object(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError(
            f"{path}: expected schema_version={schema!r}, got {payload.get('schema_version')!r}"
        )
    return payload


def atomic_text(path: Path, content: str, *, freeze_existing: bool = False) -> None:
    """Atomically replace a text file, or freeze it once first written.

    ``freeze_existing`` is used for plans: an identical rerun is idempotent,
    while a changed plan must use a new output root.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if freeze_existing and path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        raise FileExistsError(f"refusing to overwrite frozen plan with different content: {path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    )
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: dict[str, Any], *, freeze_existing: bool = False) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        freeze_existing=freeze_existing,
    )


def parse_seeds(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",")]
        if not pieces or any(not piece for piece in pieces):
            raise ValueError("--seeds must be a comma-separated list of integers")
        try:
            seeds = [int(piece) for piece in pieces]
        except ValueError as exc:
            raise ValueError("--seeds must contain integers only") from exc
    else:
        seeds = [int(seed) for seed in raw]
    if len(seeds) < 2:
        raise ValueError("at least two distinct training seeds are required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("training seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    return seeds


def require_keys(payload: dict[str, Any], keys: Iterable[str], *, where: str) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise ValueError(f"{where}: missing required keys {missing}")


def verify_path_binding(binding: Any, *, where: str) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{where}: file binding must be an object")
    require_keys(binding, ("path", "sha256"), where=where)
    if not isinstance(binding["path"], str) or not isinstance(binding["sha256"], str):
        raise ValueError(f"{where}: path and sha256 must be strings")
    path = Path(binding["path"])
    actual = sha256(path)
    if actual != binding["sha256"]:
        raise ValueError(f"{where}: SHA-256 mismatch for {path}: expected {binding['sha256']}, got {actual}")
    return path


def verify_file_binding(binding: Any, *, where: str, schema: str | None = None) -> tuple[Path, dict[str, Any]]:
    path = verify_path_binding(binding, where=where)
    return path, load_json_object(path, schema=schema)
