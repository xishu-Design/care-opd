from __future__ import annotations

from textwrap import dedent

from app.data_pipeline.validators.source_registry_validator import SourceRegistryValidator, validate_default_registry


def test_default_source_registry_validates_and_builds_download_queue():
    validator = SourceRegistryValidator()
    sources = validator.load_sources()
    report = validate_default_registry()
    source_ids = {source["source_id"] for source in sources}

    assert report.valid, report.to_dict()
    assert "SRC_TCM_DATA_FOR_ML" in source_ids
    assert "SRC_TCM_MKG_ZENODO" in source_ids
    assert "SRC_TCM_LIB_SEARCH" in source_ids
    assert "SRC_HERB_DATABASE" in source_ids
    assert "SRC_SYMMAP_DATABASE" in source_ids
    assert "SRC_TCMD_QA" in source_ids
    assert report.download_queue == [
        "SRC_TCM_DATA_FOR_ML",
        "SRC_MTCMB",
        "SRC_DRYAD_TCM_NETWORK",
        "SRC_SHENNONG_TCM_DATASET",
    ]
    assert report.approved_count == 4
    assert report.pending_count >= 1

    for source in sources:
        assert source["source_id"]
        assert source["homepage"]
        assert source["license"]
        assert source["status"]
    for source in sources:
        if source["source_id"] not in {
            "SRC_TCM_DATA_FOR_ML",
            "SRC_MTCMB",
            "SRC_DRYAD_TCM_NETWORK",
            "SRC_SHENNONG_TCM_DATASET",
        }:
            assert not validator.can_download(source)


def test_license_status_check_marks_approved_and_pending_sources():
    validator = SourceRegistryValidator()
    by_id = {source["source_id"]: source for source in validator.load_sources()}

    assert validator.license_status(by_id["SRC_TCM_DATA_FOR_ML"]) == "approved"
    assert validator.license_status(by_id["SRC_TCM_MKG_ZENODO"]) == "pending_review"
    assert validator.license_status(by_id["SRC_TCM_LIB_SEARCH"]) == "pending_review"


def test_registry_validator_detects_missing_source_id(tmp_path):
    _write_registry(
        tmp_path,
        registry_body="""
        sources:
          - name: Missing ID
            homepage: https://example.test
            download_url: ""
            repository_url: ""
            source_type: test
            license: MIT
            license_url: ""
            allowed_use: import_allowed
            data_format: csv
            expected_content: []
            status: approved
            notes: test
            citation: test
        """,
        approved_body="approved_sources: []",
    )

    report = SourceRegistryValidator(tmp_path).validate()

    assert not report.valid
    assert "MISSING_SOURCE_ID" in {issue.code for issue in report.errors}


def test_registry_validator_blocks_approved_source_without_approved_license(tmp_path):
    _write_registry(
        tmp_path,
        registry_body="""
        sources:
          - source_id: SRC_PENDING_BUT_APPROVED
            name: Pending source
            homepage: https://example.test
            download_url: https://example.test/data.git
            repository_url: https://example.test/data
            source_type: github_dataset
            license: TO_BE_VERIFIED
            license_url: ""
            allowed_use: import_allowed
            data_format: csv
            expected_content:
              - herb_index
            status: approved
            notes: test
            citation: test
        """,
        approved_body="approved_sources:\n  - SRC_PENDING_BUT_APPROVED\n",
    )

    report = SourceRegistryValidator(tmp_path).validate()

    assert not report.valid
    assert report.download_queue == []
    assert "APPROVED_SOURCE_LICENSE_NOT_APPROVED" in {issue.code for issue in report.errors}


def test_registry_validator_rejects_approved_and_rejected_overlap(tmp_path):
    _write_registry(
        tmp_path,
        registry_body="""
        sources:
          - source_id: SRC_OVERLAP
            name: Overlap source
            homepage: https://example.test
            download_url: https://example.test/data.git
            repository_url: https://example.test/data
            source_type: github_dataset
            license: MIT
            license_url: ""
            allowed_use: import_allowed
            data_format: csv
            expected_content:
              - herb_index
            status: approved
            notes: test
            citation: test
        """,
        approved_body="approved_sources:\n  - SRC_OVERLAP\n",
        rejected_body="rejected_sources:\n  - SRC_OVERLAP\n",
    )

    report = SourceRegistryValidator(tmp_path).validate()

    assert not report.valid
    assert report.download_queue == []
    assert "SOURCE_APPROVED_AND_REJECTED" in {issue.code for issue in report.errors}


def _write_registry(tmp_path, *, registry_body: str, approved_body: str, rejected_body: str = "rejected_sources: []") -> None:
    (tmp_path / "data_sources_registry.yaml").write_text(dedent(registry_body).strip() + "\n", encoding="utf-8")
    (tmp_path / "approved_sources.yaml").write_text(dedent(approved_body).strip() + "\n", encoding="utf-8")
    (tmp_path / "rejected_sources.yaml").write_text(dedent(rejected_body).strip() + "\n", encoding="utf-8")
