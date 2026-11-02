from __future__ import annotations

import json
from pathlib import Path

from app.data_pipeline.extractors.mtcmb_extractor import MTCMBExtractor
from app.data_pipeline.stable_ids import stable_record_id
from app.tests.conftest import auth_header, login, register


def _extractor(raw_dir: Path) -> MTCMBExtractor:
    return MTCMBExtractor(
        raw_dir,
        {
            "source_id": "SRC_MTCMB",
            "license": "Apache-2.0",
            "clinical_evidence_allowed": False,
            "doctor_evidence_allowed": False,
            "patient_visible": False,
            "student_visible": True,
        },
        {"detected_license": "Apache-2.0", "can_process": True},
        {"commit_hash": "faffd813"},
    )


def test_mtcmb_only_approves_safe_student_evaluation_items(tmp_path):
    data_dir = tmp_path / "raw" / "SRC_MTCMB" / "data"
    data_dir.mkdir(parents=True)
    records = [
        {"id": 1, "question": "中医基础理论中的五行是什么？", "answer": "木火土金水", "source": "教学题库"},
        {"id": 2, "question": "制川乌常用剂量是多少？", "answer": "1.5～3g", "source": "教学题库"},
    ]
    (data_dir / "TCM-ED-A.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")

    result = _extractor(tmp_path / "raw").extract()

    assert len(result["approved_training_items"]) == 1
    approved = result["approved_training_items"][0]
    assert approved["student_visible"] is True
    assert approved["patient_visible"] is False
    assert approved["doctor_evidence_allowed"] is False
    assert any("dosage_output_detected" in item["reason"] for item in result["rejected_items"])


def test_mtcmb_quarantines_patient_case_datasets(tmp_path):
    data_dir = tmp_path / "raw" / "SRC_MTCMB" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "TCM-PR.jsonl").write_text(
        json.dumps({"id": 1, "disease_case": {"主诉": "失眠"}, "answer": ["酸枣仁"], "source": "评测"}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = _extractor(tmp_path / "raw").extract()

    assert result["approved_training_items"] == []
    assert result["rejected_items"][0]["reason"] == "clinical_or_patient_case_dataset"


def test_stable_record_id_is_repeatable_and_commit_bound():
    values = {"source_id": "SRC_TEST", "commit_sha": "abc", "source_file": "data.jsonl", "source_record_id": "row_1"}

    first = stable_record_id("HERB", **values)
    second = stable_record_id("HERB", **values)
    changed = stable_record_id("HERB", **{**values, "commit_sha": "def"})

    assert first == second
    assert first != changed
    assert first.startswith("HERB_")


def test_patient_cannot_access_graph_api(client):
    register(client, "graph_patient", "patient")
    token = login(client, "graph_patient", "patient")

    response = client.get("/api/lightrag/graph?label=黄芪", headers=auth_header(token))

    assert response.status_code == 403


def test_student_graph_fallback_never_returns_doctor_internal(client):
    register(client, "graph_student", "student")
    token = login(client, "graph_student", "student")

    response = client.get("/api/lightrag/graph?label=黄芪", headers=auth_header(token))

    assert response.status_code == 200
    payload = response.json()["data"]
    assert all("doctor_internal" not in node["visibility"] for node in payload["nodes"])


def test_graph_endpoint_returns_local_result_when_lightrag_is_disabled(client):
    register(client, "graph_doctor_variant", "doctor")
    token = login(client, "graph_doctor_variant", "doctor")

    response = client.get("/api/lightrag/graph?label=麝虫", headers=auth_header(token))

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["meta"]["lightrag_status"] == "disabled"
    assert payload["meta"]["query_correction"] == "麝香"
    assert any(node["label"] == "麝香" for node in payload["nodes"])
