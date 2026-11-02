import json
from pathlib import Path
from textwrap import dedent

from app.data_pipeline.scripts.process_approved_sources import process_approved_sources


def test_process_approved_sources_processes_only_can_process_true(tmp_path):
    registry_dir = _registry(tmp_path)
    raw_dir = _raw_fixture(tmp_path)
    reports_dir = tmp_path / "reports"
    processed_dir = tmp_path / "processed"
    download_report_path = reports_dir / "download_report.json"
    license_report_path = reports_dir / "license_report.json"
    reports_dir.mkdir()
    download_report_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "source_id": "SRC_TCM_DATA_FOR_ML",
                        "download_status": "success",
                        "local_path": str(raw_dir / "SRC_TCM_DATA_FOR_ML"),
                        "commit_hash": "abc123",
                        "downloaded_at": "2026-07-07T00:00:00+00:00",
                    },
                    {"source_id": "SRC_PENDING", "download_status": "success", "local_path": str(raw_dir / "SRC_PENDING")},
                ]
            }
        ),
        encoding="utf-8",
    )
    license_report_path.write_text(
        json.dumps(
            {
                "items": [
                    {"source_id": "SRC_TCM_DATA_FOR_ML", "declared_license": "MIT", "detected_license": "MIT", "can_process": True},
                    {"source_id": "SRC_PENDING", "declared_license": "TO_BE_VERIFIED", "detected_license": "NO_LICENSE_FOUND", "can_process": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = process_approved_sources(
        license_report_path=license_report_path,
        download_report_path=download_report_path,
        registry_dir=registry_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert report["processed_sources"] == 1
    assert report["failed_sources"] == 0
    herbs = json.loads((processed_dir / "herb_candidates.json").read_text(encoding="utf-8"))
    formulas = json.loads((processed_dir / "formula_candidates.json").read_text(encoding="utf-8"))
    catalog = json.loads((processed_dir / "source_catalog.json").read_text(encoding="utf-8"))
    assert herbs["items"]
    assert formulas["items"]
    assert catalog["sources"][0]["source_id"] == "SRC_TCM_DATA_FOR_ML"
    assert catalog["sources"][0]["commit_hash"] == "abc123"
    assert (reports_dir / "processing_report.json").exists()
    assert not (tmp_path / "content").exists()


def test_process_approved_sources_with_no_processable_sources_writes_empty_outputs(tmp_path):
    registry_dir = _registry(tmp_path)
    raw_dir = tmp_path / "raw"
    reports_dir = tmp_path / "reports"
    processed_dir = tmp_path / "processed"
    reports_dir.mkdir()
    (reports_dir / "download_report.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    (reports_dir / "license_report.json").write_text(json.dumps({"items": [{"source_id": "SRC_PENDING", "can_process": False}]}), encoding="utf-8")

    report = process_approved_sources(
        license_report_path=reports_dir / "license_report.json",
        download_report_path=reports_dir / "download_report.json",
        registry_dir=registry_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
    )

    assert report["total_sources"] == 0
    assert json.loads((processed_dir / "herb_candidates.json").read_text(encoding="utf-8")) == {"items": []}


def _registry(tmp_path) -> Path:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "data_sources_registry.yaml").write_text(
        dedent(
            """
            sources:
              - source_id: SRC_TCM_DATA_FOR_ML
                name: TCMDataForML
                homepage: https://github.com/GuoxiangZu/TCMDataForML
                download_url: https://github.com/GuoxiangZu/TCMDataForML.git
                repository_url: https://github.com/GuoxiangZu/TCMDataForML
                source_type: github_dataset
                license: MIT
                license_url: https://github.com/GuoxiangZu/TCMDataForML
                allowed_use: import_allowed
                data_format: csv_txt
                expected_content:
                  - prescriptions
                  - herb_index
                status: approved
                notes: test
                citation: test
              - source_id: SRC_PENDING
                name: Pending
                homepage: https://example.test
                download_url: ""
                repository_url: ""
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
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (registry_dir / "approved_sources.yaml").write_text("approved_sources:\n  - SRC_TCM_DATA_FOR_ML\n", encoding="utf-8")
    (registry_dir / "rejected_sources.yaml").write_text("rejected_sources: []\n", encoding="utf-8")
    return registry_dir


def _raw_fixture(tmp_path) -> Path:
    raw_dir = tmp_path / "raw"
    source_dir = raw_dir / "SRC_TCM_DATA_FOR_ML"
    source_dir.mkdir(parents=True)
    (source_dir / "IndexName.csv").write_text("name\n黄芪\n白术\n", encoding="utf-8")
    (source_dir / "TCM.csv").write_text("formula,herbs\n示例方一,\"黄芪,白术\"\n", encoding="utf-8")
    return raw_dir
