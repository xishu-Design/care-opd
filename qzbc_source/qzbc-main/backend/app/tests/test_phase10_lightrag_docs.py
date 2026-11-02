import json

from app.content.services.content_import_service import ContentImportService
from app.db.models import LightRAGDocManifest
from app.lightrag_integration.document_builder import (
    build_lightrag_docs,
    build_lightrag_docs_from_db,
    load_lightrag_docs,
    persist_lightrag_doc_manifest,
)
from app.lightrag_integration.index_builder import build_lightrag_index
from app.tests.test_phase9_content_import import _content_fixture


def test_lightrag_document_builder_exports_traceable_docs(tmp_path):
    content_dir = _content_fixture(tmp_path)
    docs_path = tmp_path / "input_docs" / "guardtcm_lightrag_docs.jsonl"
    report_path = tmp_path / "reports" / "doc_report.json"

    report = build_lightrag_docs(content_data_dir=content_dir, output_path=docs_path, report_path=report_path)
    docs = load_lightrag_docs(docs_path)

    assert report["status"] == "success"
    assert report["docs_written"] == 2
    assert docs_path.exists()
    assert {doc.entry_type for doc in docs} == {"herb", "formula"}
    herb = next(doc for doc in docs if doc.entry_type == "herb")
    assert herb.metadata["source_id"] == "SRC_TCM_DATA_FOR_ML"
    assert herb.metadata["license"] == "MIT"
    assert herb.metadata["visibility"] == ["doctor", "student"]
    assert "未推断" not in herb.text
    assert "不作为患者自行用药依据" in herb.text


def test_lightrag_document_builder_skips_missing_source_id(tmp_path):
    content_dir = _content_fixture(tmp_path)
    herbs_path = content_dir / "herbs.json"
    herbs = json.loads(herbs_path.read_text(encoding="utf-8"))
    herbs["items"][0].pop("source_ids")
    herbs_path.write_text(json.dumps(herbs, ensure_ascii=False), encoding="utf-8")

    report = build_lightrag_docs(
        content_data_dir=content_dir,
        output_path=tmp_path / "input_docs" / "guardtcm_lightrag_docs.jsonl",
        report_path=tmp_path / "reports" / "doc_report.json",
    )

    assert report["docs_written"] == 1
    assert report["skipped"] == 1
    assert report["errors"][0]["code"] == "MISSING_SOURCE_ID"


def test_lightrag_index_builder_writes_report_and_working_dir(tmp_path):
    content_dir = _content_fixture(tmp_path)
    docs_path = tmp_path / "input_docs" / "guardtcm_lightrag_docs.jsonl"
    build_lightrag_docs(content_data_dir=content_dir, output_path=docs_path, report_path=tmp_path / "reports" / "doc_report.json")

    report = build_lightrag_index(
        docs_path=docs_path,
        working_dir=tmp_path / "working_dir",
        report_path=tmp_path / "reports" / "index_report.json",
        rebuild=True,
    )

    assert report["status"] == "success"
    assert report["docs_indexed"] == 2
    assert (tmp_path / "working_dir" / "guardtcm_lightrag_index.jsonl").exists()


def test_lightrag_document_builder_can_export_database_knowledge_entries(db_session, tmp_path):
    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()
    docs_path = tmp_path / "db_docs" / "guardtcm_lightrag_docs.jsonl"

    report = build_lightrag_docs_from_db(db_session, output_path=docs_path, report_path=tmp_path / "reports" / "db_doc_report.json")
    docs = load_lightrag_docs(docs_path)

    assert report["source"] == "database"
    assert report["docs_written"] == 2
    assert {doc.entry_type for doc in docs} == {"herb", "formula"}
    assert all(doc.metadata["source_id"] == "SRC_TCM_DATA_FOR_ML" for doc in docs)


def test_doc_ids_are_stable_and_manifest_is_traceable(db_session, tmp_path):
    content_dir = _content_fixture(tmp_path)
    docs_path = tmp_path / "docs.jsonl"

    build_lightrag_docs(content_data_dir=content_dir, output_path=docs_path, report_path=tmp_path / "first.json")
    first_ids = [doc.doc_id for doc in load_lightrag_docs(docs_path)]
    build_lightrag_docs(content_data_dir=content_dir, output_path=docs_path, report_path=tmp_path / "second.json")
    second_ids = [doc.doc_id for doc in load_lightrag_docs(docs_path)]

    assert first_ids == second_ids
    assert persist_lightrag_doc_manifest(db_session, snapshot_id="SNAP_TEST", docs_path=docs_path) == len(first_ids)
    rows = db_session.query(LightRAGDocManifest).all()
    assert rows
    assert all(row.source_ids_json and row.content_hash and row.entry_ids_json for row in rows)
