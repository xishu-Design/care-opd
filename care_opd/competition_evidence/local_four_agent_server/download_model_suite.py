#!/usr/bin/env python3
"""Resolve immutable revisions and download the three local blind-review baselines.

The selection lock is written before any model is downloaded.  A resumed run
reuses that lock even if a repository's ``main`` branch later moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "care_opd.competition.local_model_suite.v1"
LOCK_SCHEMA = "care_opd.competition.local_model_suite.selection_lock.v1"
DOWNLOAD_SCHEMA = "care_opd.competition.local_model_suite.download_manifest.v1"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def normalize_license(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def validate_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")
    models = spec.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("model suite must contain exactly three baselines")
    if not all(isinstance(row, dict) for row in models):
        raise ValueError("every model entry must be an object")
    rows = [dict(row) for row in models]
    required = {
        "system_id",
        "repo_id",
        "requested_revision",
        "local_dir_name",
        "served_model_name",
        "license_expected",
    }
    for row in rows:
        missing = sorted(key for key in required if not str(row.get(key) or "").strip())
        if missing:
            raise ValueError(f"model entry missing {missing}: {row.get('system_id')}")
    for key in ("system_id", "repo_id", "local_dir_name", "served_model_name"):
        values = [str(row[key]) for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {key} in model suite")
    return rows


def observed_license(info: Any) -> str:
    card = getattr(info, "card_data", None)
    if hasattr(card, "to_dict"):
        card = card.to_dict()
    if isinstance(card, dict):
        return normalize_license(card.get("license"))
    return ""


def resolve_lock(
    spec: dict[str, Any], rows: list[dict[str, Any]], lock_path: Path, token: str | None
) -> dict[str, Any]:
    spec_sha = canonical_hash(spec)
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock.get("schema_version") != LOCK_SCHEMA:
            raise ValueError(f"unexpected existing lock schema: {lock_path}")
        if lock.get("model_suite_sha256") != spec_sha:
            raise ValueError(
                "existing selection lock belongs to a different model suite; use a new root"
            )
        return lock

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - server dependency check
        raise RuntimeError(
            "huggingface_hub is required; install it in the evaluation environment"
        ) from exc

    api = HfApi(token=token)
    resolved: list[dict[str, Any]] = []
    for row in rows:
        info = api.model_info(
            str(row["repo_id"]), revision=str(row["requested_revision"])
        )
        sha = str(getattr(info, "sha", "") or "")
        if len(sha) != 40:
            raise RuntimeError(f"could not resolve immutable commit for {row['repo_id']}")
        seen_license = observed_license(info)
        expected_license = normalize_license(row["license_expected"])
        if seen_license and seen_license != expected_license:
            raise RuntimeError(
                f"license mismatch for {row['repo_id']}: "
                f"expected={expected_license}, observed={seen_license}"
            )
        resolved.append(
            {
                **row,
                "resolved_revision": sha,
                "license_observed": seen_license or "not_exposed_by_hub_api",
            }
        )
    lock = {
        "schema_version": LOCK_SCHEMA,
        "status": "FROZEN_BEFORE_ANY_BLIND_EVAL_GENERATION",
        "model_suite_sha256": spec_sha,
        "selection_scope": spec.get("selection_scope"),
        "comparison_boundary": spec.get("comparison_boundary"),
        "models": resolved,
    }
    write_json_atomic(lock_path, lock)
    return lock


def directory_audit(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    critical_names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    critical = {}
    for name in critical_names:
        candidate = path / name
        if candidate.is_file():
            critical[name] = file_hash(candidate)
    weight_files = [item for item in files if item.suffix == ".safetensors"]
    if not (path / "config.json").is_file() or not weight_files:
        raise RuntimeError(f"download is incomplete: {path}")
    return {
        "file_count": len(files),
        "weight_file_count": len(weight_files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "critical_file_sha256": critical,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = args.spec.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    spec = read_json(spec_path)
    rows = validate_spec(spec)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        output_root.chmod(0o700)
    except OSError:
        pass
    token = os.environ.get(args.token_env, "").strip() or None
    lock_path = output_root / "suite_selection_lock.private.json"
    lock = resolve_lock(spec, rows, lock_path, token)
    if args.resolve_only:
        return {
            "status": "SELECTION_LOCKED_NO_DOWNLOAD",
            "selection_lock": str(lock_path),
            "model_count": len(lock["models"]),
        }

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - server dependency check
        raise RuntimeError("huggingface_hub is required") from exc

    completed: list[dict[str, Any]] = []
    manifest_path = output_root / "suite_download_manifest.private.json"
    for row in lock["models"]:
        target = output_root / str(row["local_dir_name"])
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(row["repo_id"]),
            revision=str(row["resolved_revision"]),
            local_dir=target,
            token=token,
            max_workers=args.max_workers,
        )
        completed.append(
            {
                "system_id": row["system_id"],
                "repo_id": row["repo_id"],
                "resolved_revision": row["resolved_revision"],
                "local_dir": str(target),
                **directory_audit(target),
            }
        )
        write_json_atomic(
            manifest_path,
            {
                "schema_version": DOWNLOAD_SCHEMA,
                "status": "IN_PROGRESS",
                "selection_lock_sha256": file_hash(lock_path),
                "completed": completed,
            },
        )
    result = {
        "schema_version": DOWNLOAD_SCHEMA,
        "status": "COMPLETE",
        "selection_lock_sha256": file_hash(lock_path),
        "model_count": len(completed),
        "completed": completed,
    }
    write_json_atomic(manifest_path, result)
    return {**result, "manifest": str(manifest_path)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--spec", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--token-env", default="HF_TOKEN")
    value.add_argument("--max-workers", type=int, default=4)
    value.add_argument("--resolve-only", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
