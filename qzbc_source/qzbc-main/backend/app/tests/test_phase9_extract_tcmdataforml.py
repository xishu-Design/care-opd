from pathlib import Path

from app.data_pipeline.extractors.tcmdataforml_extractor import TCMDataForMLExtractor
from app.data_pipeline.scripts.inspect_raw_source import inspect_raw_source


def test_tcmdataforml_extractor_reads_index_and_tcm_csv(tmp_path):
    raw_dir = _raw_fixture(tmp_path)
    extractor = _extractor(raw_dir)

    result = extractor.extract()

    assert result["stats"]["herb_candidates"] >= 2
    assert result["stats"]["formula_candidates"] == 2
    assert result["stats"]["prescription_candidates"] == 2
    assert result["stats"]["learning_case_candidates"] == 2
    herb = result["herb_candidates"][0]
    formula = result["formula_candidates"][0]
    assert herb["source_id"] == "SRC_TCM_DATA_FOR_ML"
    assert herb["source_record_id"]
    assert herb["license"] == "MIT"
    assert "functions" in herb["fields_missing"]
    assert all(not item["name"].isdigit() for item in result["herb_candidates"])
    assert formula["components"]
    assert formula["source_record_id"] == "row_1"
    assert formula["visibility_suggestion"] == ["doctor", "student"]


def test_tcmdataforml_extractor_missing_files_generates_pending_items(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "SRC_TCM_DATA_FOR_ML").mkdir(parents=True)
    result = _extractor(raw_dir).extract()

    reasons = {item["reason"] for item in result["pending_items"]}

    assert "index_name_file_missing" in reasons
    assert "tcm_csv_missing" in reasons
    assert result["stats"]["herb_candidates"] == 0


def test_tcmdataforml_extractor_missing_raw_dir_generates_rejected_item(tmp_path):
    result = _extractor(tmp_path / "raw").extract()

    assert result["rejected_items"][0]["reason"] == "raw_source_missing"


def test_inspect_raw_source_lists_csv_and_text_files(tmp_path):
    raw_dir = _raw_fixture(tmp_path)
    report = inspect_raw_source("SRC_TCM_DATA_FOR_ML", raw_dir)

    paths = {item["path"] for item in report["files"]}

    assert "IndexName.csv" in paths
    assert "TCM.csv" in paths
    assert report["status"] == "ok"


def _raw_fixture(tmp_path) -> Path:
    raw_dir = tmp_path / "raw"
    source_dir = raw_dir / "SRC_TCM_DATA_FOR_ML"
    source_dir.mkdir(parents=True)
    (source_dir / "IndexName.csv").write_text("index,name\n0,黄芪\n1,白术\n2,黄 芪\n", encoding="utf-8")
    (source_dir / "TCM.csv").write_text("formula,herbs\n示例方一,\"黄芪,白术\"\n,\"茯苓;甘草\"\n", encoding="utf-8")
    return raw_dir


def _extractor(raw_dir: Path) -> TCMDataForMLExtractor:
    return TCMDataForMLExtractor(
        raw_dir,
        {"source_id": "SRC_TCM_DATA_FOR_ML", "name": "TCMDataForML", "repository_url": "https://github.com/GuoxiangZu/TCMDataForML", "license": "MIT"},
        {"source_id": "SRC_TCM_DATA_FOR_ML", "declared_license": "MIT", "detected_license": "MIT", "can_process": True},
        {"source_id": "SRC_TCM_DATA_FOR_ML", "commit_hash": "abc123", "downloaded_at": "2026-07-07T00:00:00+00:00"},
    )
