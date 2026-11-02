from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.data_pipeline.validators.license_validator import DEFAULT_RAW_DIR, DEFAULT_REPORTS_DIR
from app.data_pipeline.validators.source_registry_validator import DEFAULT_REGISTRY_DIR, SourceRegistryValidator


Runner = Callable[..., subprocess.CompletedProcess[str]]


def build_download_report(
    *,
    registry_dir: Path | str = DEFAULT_REGISTRY_DIR,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    output_path: Path | str = DEFAULT_REPORTS_DIR / "download_report.json",
    source_id: str | None = None,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    started_at = _now()
    validator = SourceRegistryValidator(registry_dir)
    registry_report = validator.validate()
    if not registry_report.valid:
        raise ValueError(f"Source registry is invalid: {registry_report.to_dict()}")

    raw_root = Path(raw_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    queue = validator.download_queue()
    if source_id:
        queue = [source for source in queue if source.get("source_id") == source_id]
        if not queue:
            raise ValueError(f"Source {source_id} is not approved for download")

    items = [_download_source(source, raw_root, force=force, runner=runner) for source in queue]
    report = {
        "run_id": f"download_{_compact_timestamp()}",
        "started_at": started_at,
        "ended_at": _now(),
        "total_sources": len(items),
        "downloaded": sum(1 for item in items if item["download_status"] == "success"),
        "failed": sum(1 for item in items if item["download_status"] == "failed"),
        "skipped": sum(1 for item in items if item["download_status"] == "skipped"),
        "items": items,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download approved GuardTCM open-source data sources.")
    parser.add_argument("--force", action="store_true", help="Remove an existing raw/{source_id} directory before downloading.")
    parser.add_argument("--source", help="Download a single approved source_id.")
    args = parser.parse_args(argv)
    report = build_download_report(source_id=args.source, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _download_source(source: dict[str, Any], raw_root: Path, *, force: bool, runner: Runner) -> dict[str, Any]:
    source_id = str(source["source_id"])
    local_path = raw_root / source_id
    repository_url = str(source.get("repository_url") or source.get("download_url") or "")
    base_item = {
        "source_id": source_id,
        "name": source.get("name", ""),
        "repository_url": repository_url,
        "source_version": str(source.get("source_version") or source.get("version") or ""),
        "local_path": str(local_path),
        "download_status": "failed",
        "commit_hash": "",
        "branch": "",
        "downloaded_at": "",
        "error": None,
    }

    if local_path.exists():
        if not force:
            commit_hash = _git_output(["git", "-C", str(local_path), "rev-parse", "HEAD"], runner)
            branch = _git_output(["git", "-C", str(local_path), "rev-parse", "--abbrev-ref", "HEAD"], runner)
            status = "skipped" if commit_hash else "failed"
            return {
                **base_item,
                "download_status": status,
                "commit_hash": commit_hash,
                "branch": branch,
                "downloaded_at": _now(),
                "error": "local_path_exists" if commit_hash else "existing_path_is_not_a_git_repository",
            }
        shutil.rmtree(local_path)

    if not _is_git_source(source):
        return {**base_item, "download_status": "failed", "error": "unsupported_source_type"}

    try:
        clone_result = runner(
            ["git", "clone", "--depth", "1", repository_url, str(local_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {**base_item, "download_status": "failed", "downloaded_at": _now(), "error": str(exc)}

    if clone_result.returncode != 0:
        error = clone_result.stderr.strip() or clone_result.stdout.strip() or f"git clone failed with exit code {clone_result.returncode}"
        return {**base_item, "download_status": "failed", "downloaded_at": _now(), "error": error}

    commit_hash = _git_output(["git", "-C", str(local_path), "rev-parse", "HEAD"], runner)
    branch = _git_output(["git", "-C", str(local_path), "rev-parse", "--abbrev-ref", "HEAD"], runner)
    return {
        **base_item,
        "download_status": "success",
        "commit_hash": commit_hash,
        "branch": branch,
        "downloaded_at": _now(),
        "error": None,
    }


def _git_output(cmd: list[str], runner: Runner) -> str:
    result = runner(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _is_git_source(source: dict[str, Any]) -> bool:
    source_type = str(source.get("source_type", ""))
    repository_url = str(source.get("repository_url") or source.get("download_url") or "")
    return "github" in source_type or "huggingface" in source_type or repository_url.endswith(".git") or "github.com" in repository_url or "huggingface.co/datasets/" in repository_url


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _compact_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
