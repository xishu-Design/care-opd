"""Shared fail-closed validation for the public evaluation protocol lock."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "care-opd-public-preregistration-lock-v1"
MODEL_ORDER = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((str(root), str(candidate))) == str(root)
    except ValueError:
        return False


def validate_static_lock(
    path: Path,
    *,
    project_root: Path,
    required_paths: Iterable[str] = (),
    timing_key: str | None = None,
    expected_model_order: Sequence[str] = MODEL_ORDER,
) -> dict[str, Any]:
    """Validate the complete static inventory and return its immutable identity.

    ``project_root`` is the ``care_opd`` directory.  Every locked relative path
    is resolved beneath it and rehashed; an unrelated non-empty JSON file is
    never accepted as a preregistration lock.
    """

    resolved_lock = path.resolve(strict=True)
    root = project_root.resolve(strict=True)
    payload = json.loads(resolved_lock.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"preregistration lock must be an object: {resolved_lock}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected preregistration schema: {resolved_lock}")
    embedded_fingerprint = str(payload.get("contract_fingerprint") or "")
    contract = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "contract_fingerprint"}
    }
    if not embedded_fingerprint or canonical_hash(contract) != embedded_fingerprint:
        raise ValueError("preregistration contract fingerprint does not close")
    if tuple(payload.get("fixed_model_order") or ()) != tuple(expected_model_order):
        raise ValueError("preregistration model order changed")
    model_suite = payload.get("model_suite")
    if not isinstance(model_suite, Mapping):
        raise ValueError("preregistration model suite is absent")
    models = model_suite.get("models")
    if not isinstance(models, list):
        raise ValueError("preregistration model suite entries are absent")
    suite_labels = tuple(str(item.get("label") or "") for item in models)
    if suite_labels != tuple(expected_model_order):
        raise ValueError("preregistered model suite order changed")
    semantic = []
    for item in models:
        fingerprint = str(item.get("model_fingerprint") or "")
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise ValueError("invalid preregistered model fingerprint")
        semantic.append((str(item["label"]), fingerprint))
    suite_fingerprint = canonical_hash(semantic)
    if model_suite.get("suite_fingerprint") != suite_fingerprint:
        raise ValueError("preregistered model suite fingerprint does not close")
    if not str(payload.get("created_at_utc") or ""):
        raise ValueError("preregistration creation timestamp is absent")

    origin_reference = payload.get("origin_reference")
    if isinstance(origin_reference, Mapping) and origin_reference.get(
        "status"
    ) == "REFERENCE_MATCHED":
        origin_path = Path(str(origin_reference.get("path") or ""))
        if not origin_path.is_file():
            raise FileNotFoundError(f"origin reference disappeared: {origin_path}")
        if sha256_file(origin_path) != origin_reference.get("sha256"):
            raise ValueError("origin reference content changed after locking")

    auxiliary_models = payload.get("auxiliary_models")
    if isinstance(auxiliary_models, Mapping):
        for name, raw_asset in auxiliary_models.items():
            if not isinstance(raw_asset, Mapping) or raw_asset.get("status") != "LOCKED":
                continue
            canonical_path = str(raw_asset.get("canonical_path") or "")
            asset_root = Path(canonical_path)
            inventory_rows = raw_asset.get("inventory")
            if not asset_root.is_dir() or not isinstance(inventory_rows, list):
                raise ValueError(f"invalid locked auxiliary model: {name}")
            resolved_asset_root = asset_root.resolve(strict=True)
            if str(resolved_asset_root) != canonical_path:
                raise ValueError(f"auxiliary model path is not canonical: {name}")
            observed_inventory = []
            for row in inventory_rows:
                if not isinstance(row, Mapping):
                    raise ValueError(f"invalid auxiliary inventory row: {name}")
                relative = str(row.get("name") or "")
                target = (resolved_asset_root / relative).resolve()
                if not _inside(resolved_asset_root, target) or not target.is_file():
                    raise FileNotFoundError(
                        f"locked auxiliary artifact is missing: {name}/{relative}"
                    )
                observed = {
                    "name": relative,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
                if observed != dict(row):
                    raise ValueError(
                        f"locked auxiliary artifact changed: {name}/{relative}"
                    )
                observed_inventory.append(observed)
            if canonical_hash(observed_inventory) != raw_asset.get("artifact_fingerprint"):
                raise ValueError(f"auxiliary model fingerprint changed: {name}")
    if timing_key:
        disclosure = payload.get("timing_disclosure")
        if not isinstance(disclosure, Mapping) or disclosure.get(timing_key) is not False:
            raise ValueError(
                f"timing disclosure {timing_key!r} is absent or predictions were already observed"
            )

    inventory = payload.get("locked_files")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("preregistration locked_files inventory is empty")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in inventory:
        if not isinstance(item, Mapping):
            raise ValueError("invalid locked_files row")
        relative = str(item.get("path") or "")
        if not relative or relative in indexed:
            raise ValueError(f"invalid or duplicate locked path: {relative!r}")
        indexed[relative] = item
    missing = set(required_paths) - set(indexed)
    if missing:
        raise ValueError(f"required files are absent from lock: {sorted(missing)}")
    for relative, item in indexed.items():
        target = (root / relative).resolve()
        if not _inside(root, target):
            raise ValueError(f"locked path escapes project root: {relative}")
        if not target.is_file():
            raise FileNotFoundError(f"preregistered file is missing: {target}")
        if int(item.get("bytes", -1)) != target.stat().st_size:
            raise ValueError(f"preregistered file size changed: {relative}")
        expected_hash = str(item.get("sha256") or "")
        if expected_hash != sha256_file(target):
            raise ValueError(f"preregistered file hash changed: {relative}")
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(resolved_lock),
        "sha256": sha256_file(resolved_lock),
        "contract_fingerprint": embedded_fingerprint,
        "created_at_utc": str(payload["created_at_utc"]),
        "fixed_model_order": list(expected_model_order),
        "model_suite_fingerprint": suite_fingerprint,
        "origin_reference_status": str(
            (payload.get("origin_reference") or {}).get("status") or "NOT_VERIFIED"
        ),
        "auxiliary_models": {
            str(name): {
                "status": str(asset.get("status") or ""),
                "canonical_path": str(asset.get("canonical_path") or ""),
                "artifact_fingerprint": str(
                    asset.get("artifact_fingerprint") or ""
                ),
            }
            for name, asset in (payload.get("auxiliary_models") or {}).items()
            if isinstance(asset, Mapping)
        },
        "locked_files": len(indexed),
        "locked_files_fingerprint": canonical_hash(payload["locked_files"]),
    }
