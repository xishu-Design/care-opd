#!/usr/bin/env python3
"""Build the minimal PRIVATE bundle for the local four-system server run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


EVIDENCE_FILES = (
    "execute_three_role_eval.py",
    "three_role_eval_common.py",
    "validate_local_four_agent_run.py",
    "prepare_agent_preference_review.py",
    "summarize_agent_preference_review.py",
)
LOCAL_FILES = (
    "model_suite.json",
    "download_model_suite.py",
    "run_four_agent_local_server.sh",
    "RUNBOOK_ZH.md",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    ".conda",
    ".redteam_venv",
    "node_modules",
}
EXCLUDED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pyc", ".log"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_backend(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        copy_file(path, target / relative)


def run(args: argparse.Namespace) -> dict[str, object]:
    care_root = args.care_root.expanduser().resolve()
    qzbc_root = args.qzbc_root.expanduser().resolve()
    tasks = args.tasks.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="four_agent_bundle_") as temporary:
        staging = Path(temporary) / "four_agent_expert_blind_bundle"
        evidence_target = staging / "care_opd" / "competition_evidence"
        local_target = evidence_target / "local_four_agent_server"
        evidence_source = care_root / "competition_evidence"
        local_source = evidence_source / "local_four_agent_server"
        for name in EVIDENCE_FILES:
            copy_file(evidence_source / name, evidence_target / name)
        for name in LOCAL_FILES:
            copy_file(local_source / name, local_target / name)

        task_target = (
            staging
            / "care_opd"
            / "runs"
            / "private_real_thyroid_external"
            / "three_role_eval"
            / "paired_fact_n12"
            / "three_role_tasks.private.jsonl"
        )
        copy_file(tasks, task_target)
        copy_backend(qzbc_root / "backend", staging / "qzbc_private_eval" / "backend")
        copy_file(qzbc_root / "requirements.txt", staging / "qzbc_private_eval" / "requirements.txt")
        if (qzbc_root / "environment.yml").is_file():
            copy_file(
                qzbc_root / "environment.yml",
                staging / "qzbc_private_eval" / "environment.yml",
            )

        included = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = {
            "schema_version": "care_opd.competition.local_server_private_bundle.v1",
            "status": "PRIVATE_SERVER_INPUT_ONLY",
            "privacy_warning": (
                "Contains de-identified structured real medical task payloads. "
                "Do not publish, upload to a model hub, or include in the competition package."
            ),
            "file_count": len(included),
            "files": [
                {
                    "path": path.relative_to(staging.parent).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in included
            ],
        }
        manifest_path = staging / "BUNDLE_MANIFEST.private.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(staging, arcname=staging.name, recursive=True)

    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    return {
        "status": "COMPLETE_PRIVATE_BUNDLE",
        "archive": str(output),
        "archive_sha256": sha256(output),
        "sidecar": str(sidecar),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--care-root", type=Path, required=True)
    value.add_argument("--qzbc-root", type=Path, required=True)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
