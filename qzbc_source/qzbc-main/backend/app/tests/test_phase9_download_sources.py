from __future__ import annotations

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

from app.data_pipeline.scripts.download_sources import build_download_report
from app.data_pipeline.validators.source_registry_validator import DATA_PIPELINE_DIR


def test_download_sources_downloads_only_approved_source(tmp_path):
    registry_dir = _write_registry(
        tmp_path,
        registry_body="""
        sources:
          - source_id: SRC_APPROVED
            name: Approved source
            homepage: https://github.com/example/approved
            download_url: https://github.com/example/approved.git
            repository_url: https://github.com/example/approved
            source_type: github_dataset
            license: MIT
            license_url: https://github.com/example/approved
            allowed_use: import_allowed
            data_format: csv_txt
            expected_content:
              - prescriptions
            status: approved
            notes: test
            citation: test
          - source_id: SRC_PENDING
            name: Pending source
            homepage: https://github.com/example/pending
            download_url: https://github.com/example/pending.git
            repository_url: https://github.com/example/pending
            source_type: github_dataset
            license: TO_BE_VERIFIED
            license_url: ""
            allowed_use: pending_review
            data_format: csv_txt
            expected_content:
              - prescriptions
            status: candidate
            notes: test
            citation: test
        """,
        approved_body="approved_sources:\n  - SRC_APPROVED\n",
    )
    raw_dir = tmp_path / "raw"
    report_path = tmp_path / "reports" / "download_report.json"
    commands: list[list[str]] = []

    def fake_runner(cmd, **kwargs):
        del kwargs
        commands.append(cmd)
        if cmd[:3] == ["git", "clone", "--depth"]:
            Path(cmd[-1]).mkdir(parents=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[-1] == "HEAD":
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        if cmd[-1] == "--abbrev-ref":
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    report = build_download_report(registry_dir=registry_dir, raw_dir=raw_dir, output_path=report_path, runner=fake_runner)

    assert report["downloaded"] == 1
    assert report["failed"] == 0
    assert report["items"][0]["source_id"] == "SRC_APPROVED"
    assert report["items"][0]["commit_hash"] == "abc123"
    assert (raw_dir / "SRC_APPROVED").exists()
    assert not (raw_dir / "SRC_PENDING").exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["items"][0]["source_id"] == "SRC_APPROVED"
    assert commands[0][:3] == ["git", "clone", "--depth"]


def test_download_sources_rejects_pending_source_in_approved_list(tmp_path):
    registry_dir = _write_registry(
        tmp_path,
        registry_body="""
        sources:
          - source_id: SRC_PENDING
            name: Pending source
            homepage: https://github.com/example/pending
            download_url: https://github.com/example/pending.git
            repository_url: https://github.com/example/pending
            source_type: github_dataset
            license: TO_BE_VERIFIED
            license_url: ""
            allowed_use: pending_review
            data_format: csv_txt
            expected_content:
              - prescriptions
            status: candidate
            notes: test
            citation: test
        """,
        approved_body="approved_sources:\n  - SRC_PENDING\n",
    )

    with pytest.raises(ValueError):
        build_download_report(registry_dir=registry_dir, raw_dir=tmp_path / "raw", output_path=tmp_path / "download_report.json")

    assert not (tmp_path / "raw" / "SRC_PENDING").exists()


def test_download_sources_rejects_rejected_source_even_if_approved(tmp_path):
    registry_dir = _write_registry(
        tmp_path,
        registry_body="""
        sources:
          - source_id: SRC_REJECTED
            name: Rejected source
            homepage: https://github.com/example/rejected
            download_url: https://github.com/example/rejected.git
            repository_url: https://github.com/example/rejected
            source_type: github_dataset
            license: MIT
            license_url: https://github.com/example/rejected
            allowed_use: import_allowed
            data_format: csv_txt
            expected_content:
              - prescriptions
            status: approved
            notes: test
            citation: test
        """,
        approved_body="approved_sources:\n  - SRC_REJECTED\n",
        rejected_body="rejected_sources:\n  - SRC_REJECTED\n",
    )

    with pytest.raises(ValueError):
        build_download_report(registry_dir=registry_dir, raw_dir=tmp_path / "raw", output_path=tmp_path / "download_report.json")


def test_download_sources_rejects_approved_source_missing_from_registry(tmp_path):
    registry_dir = _write_registry(
        tmp_path,
        registry_body="sources: []",
        approved_body="approved_sources:\n  - SRC_MISSING\n",
    )

    with pytest.raises(ValueError):
        build_download_report(registry_dir=registry_dir, raw_dir=tmp_path / "raw", output_path=tmp_path / "download_report.json")


def test_download_sources_writes_failed_item_when_git_clone_fails(tmp_path):
    registry_dir = _write_registry(
        tmp_path,
        registry_body="""
        sources:
          - source_id: SRC_APPROVED
            name: Approved source
            homepage: https://github.com/example/approved
            download_url: https://github.com/example/approved.git
            repository_url: https://github.com/example/approved
            source_type: github_dataset
            license: MIT
            license_url: https://github.com/example/approved
            allowed_use: import_allowed
            data_format: csv_txt
            expected_content:
              - prescriptions
            status: approved
            notes: test
            citation: test
        """,
        approved_body="approved_sources:\n  - SRC_APPROVED\n",
    )

    def failing_runner(cmd, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="network unavailable")

    report = build_download_report(registry_dir=registry_dir, raw_dir=tmp_path / "raw", output_path=tmp_path / "download_report.json", runner=failing_runner)

    assert report["downloaded"] == 0
    assert report["failed"] == 1
    assert report["items"][0]["download_status"] == "failed"
    assert report["items"][0]["error"] == "network unavailable"


def test_raw_directory_is_gitignored():
    raw_gitignore = DATA_PIPELINE_DIR / "raw" / ".gitignore"

    assert raw_gitignore.exists()
    assert "*" in raw_gitignore.read_text(encoding="utf-8")


def _write_registry(tmp_path, *, registry_body: str, approved_body: str, rejected_body: str = "rejected_sources: []") -> Path:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "data_sources_registry.yaml").write_text(dedent(registry_body).strip() + "\n", encoding="utf-8")
    (registry_dir / "approved_sources.yaml").write_text(dedent(approved_body).strip() + "\n", encoding="utf-8")
    (registry_dir / "rejected_sources.yaml").write_text(dedent(rejected_body).strip() + "\n", encoding="utf-8")
    return registry_dir
