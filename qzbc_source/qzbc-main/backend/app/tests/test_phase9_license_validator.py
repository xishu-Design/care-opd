from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from app.data_pipeline.validators.license_validator import LicenseValidator, generate_license_report


MIT_TEXT = "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy."


def test_license_validator_detects_mit_as_processable(tmp_path):
    raw_dir = tmp_path / "raw"
    source_path = raw_dir / "SRC_MIT"
    source_path.mkdir(parents=True)
    (source_path / "LICENSE").write_text(MIT_TEXT, encoding="utf-8")
    result = LicenseValidator(raw_dir).check_source({"source_id": "SRC_MIT", "license": "MIT"})

    assert result.detected_license == "MIT"
    assert result.license_file_found is True
    assert result.can_process is True
    assert result.can_import is True
    assert result.status == "approved"


def test_license_validator_keeps_to_be_verified_pending_even_with_license_file(tmp_path):
    raw_dir = tmp_path / "raw"
    source_path = raw_dir / "SRC_PENDING"
    source_path.mkdir(parents=True)
    (source_path / "LICENSE").write_text(MIT_TEXT, encoding="utf-8")
    result = LicenseValidator(raw_dir).check_source({"source_id": "SRC_PENDING", "license": "TO_BE_VERIFIED"})

    assert result.detected_license == "MIT"
    assert result.can_process is False
    assert result.can_import is False
    assert result.status == "pending_review"


def test_license_validator_marks_missing_license_file_as_pending(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "SRC_NO_LICENSE").mkdir(parents=True)
    result = LicenseValidator(raw_dir).check_source({"source_id": "SRC_NO_LICENSE", "license": "MIT"})

    assert result.detected_license == "NO_LICENSE_FOUND"
    assert result.license_file_found is False
    assert result.can_process is False
    assert result.status == "pending_review"


def test_generate_license_report_writes_report_json(tmp_path):
    registry_dir = _write_registry(tmp_path, declared_license="MIT")
    raw_dir = tmp_path / "raw"
    source_path = raw_dir / "SRC_MIT"
    source_path.mkdir(parents=True)
    (source_path / "LICENSE").write_text(MIT_TEXT, encoding="utf-8")
    download_report = {
        "run_id": "download_test",
        "items": [
            {
                "source_id": "SRC_MIT",
                "download_status": "success",
                "local_path": str(source_path),
            }
        ],
    }
    download_report_path = tmp_path / "download_report.json"
    output_path = tmp_path / "license_report.json"
    download_report_path.write_text(json.dumps(download_report), encoding="utf-8")

    report = generate_license_report(
        download_report_path=download_report_path,
        output_path=output_path,
        registry_dir=registry_dir,
        raw_dir=raw_dir,
    )

    assert report["approved_for_processing"] == 1
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["items"][0]["can_process"] is True


def test_license_report_warns_when_declared_and_detected_license_differ(tmp_path):
    registry_dir = _write_registry(tmp_path, declared_license="Apache-2.0")
    raw_dir = tmp_path / "raw"
    source_path = raw_dir / "SRC_MIT"
    source_path.mkdir(parents=True)
    (source_path / "LICENSE").write_text(MIT_TEXT, encoding="utf-8")
    download_report_path = tmp_path / "download_report.json"
    download_report_path.write_text(
        json.dumps({"run_id": "download_test", "items": [{"source_id": "SRC_MIT", "download_status": "success", "local_path": str(source_path)}]}),
        encoding="utf-8",
    )

    report = generate_license_report(
        download_report_path=download_report_path,
        output_path=tmp_path / "license_report.json",
        registry_dir=registry_dir,
        raw_dir=raw_dir,
    )

    item = report["items"][0]
    assert item["detected_license"] == "MIT"
    assert item["can_process"] is True
    assert item["warnings"]


def _write_registry(tmp_path, *, declared_license: str) -> Path:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "data_sources_registry.yaml").write_text(
        dedent(
            f"""
            sources:
              - source_id: SRC_MIT
                name: MIT source
                homepage: https://github.com/example/source
                download_url: https://github.com/example/source.git
                repository_url: https://github.com/example/source
                source_type: github_dataset
                license: {declared_license}
                license_url: https://github.com/example/source
                allowed_use: import_allowed
                data_format: csv_txt
                expected_content:
                  - prescriptions
                status: approved
                notes: test
                citation: test
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (registry_dir / "approved_sources.yaml").write_text("approved_sources:\n  - SRC_MIT\n", encoding="utf-8")
    (registry_dir / "rejected_sources.yaml").write_text("rejected_sources: []\n", encoding="utf-8")
    return registry_dir
