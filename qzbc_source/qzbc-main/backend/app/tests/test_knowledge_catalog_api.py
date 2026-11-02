from sqlalchemy.orm import Session

from app.db.models import KnowledgeEntry
from app.tests.conftest import auth_header, login, register


def test_catalog_lists_role_visible_entries_with_pagination_and_search(client, db_engine):
    _seed_entries(db_engine)
    register(client, "catalog_doctor", "doctor")
    token = login(client, "catalog_doctor", "doctor")

    response = client.get(
        "/api/knowledge/entries",
        headers=auth_header(token),
        params={"entry_type": "herb", "q": "黄芪", "page": 1, "page_size": 1},
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["total"] == 1
    assert data["items"][0]["title"] == "黄芪"
    assert data["items"][0]["license"] == "MIT"
    assert data["items"][0]["source_records"] == [{"source_id": "SRC_TCM_DATA_FOR_ML", "source_record_id": "row-1", "source_file": "herbs.csv"}]
    assert data["items"][0]["chpa_efficacies"] == ["补气"]
    assert data["items"][0]["chpa_symptoms"] == ["乏力"]
    assert {"补气", "乏力"} <= set(data["items"][0]["functions"] + data["items"][0]["symptoms"])
    assert data["items"][0]["incompatible_herbs"] == []


def test_herb_catalog_exposes_traditional_incompatibilities_instead_of_ui_visibility(client, db_engine):
    _seed_entries(db_engine)
    register(client, "catalog_pair_doctor", "doctor")
    token = login(client, "catalog_pair_doctor", "doctor")

    response = client.get(
        "/api/knowledge/entries",
        headers=auth_header(token),
        params={"entry_type": "herb", "q": "甘草"},
    )

    assert response.status_code == 200, response.text
    relations = response.json()["data"]["items"][0]["incompatible_herbs"]
    assert {(item["herb_name"], item["category"]) for item in relations} == {
        ("甘遂", "十八反"),
        ("大戟", "十八反"),
        ("芫花", "十八反"),
        ("海藻", "十八反"),
    }


def test_catalog_hides_student_only_mtcmb_from_doctor_and_exposes_to_student(client, db_engine):
    _seed_entries(db_engine)
    register(client, "catalog_doctor_visibility", "doctor")
    register(client, "catalog_student_visibility", "student")
    doctor_token = login(client, "catalog_doctor_visibility", "doctor")
    student_token = login(client, "catalog_student_visibility", "student")

    doctor = client.get("/api/knowledge/entries", headers=auth_header(doctor_token), params={"q": "MTCMB"})
    student = client.get("/api/knowledge/entries", headers=auth_header(student_token), params={"q": "MTCMB"})
    missing = client.get("/api/knowledge/entries/SRC_MTCMB/CASE_MTCMB_1", headers=auth_header(doctor_token))
    detail = client.get("/api/knowledge/entries/SRC_MTCMB/CASE_MTCMB_1", headers=auth_header(student_token))

    assert doctor.json()["data"]["total"] == 0
    assert student.json()["data"]["total"] == 1
    assert missing.status_code == 404
    assert detail.status_code == 200
    assert "patient_name" not in str(detail.json()["data"])


def test_patient_cannot_access_catalog_endpoints(client, db_engine):
    _seed_entries(db_engine)
    register(client, "catalog_patient", "patient")
    token = login(client, "catalog_patient", "patient")

    listing = client.get("/api/knowledge/entries", headers=auth_header(token))
    detail = client.get("/api/knowledge/entries/SRC_TCM_DATA_FOR_ML/HERB_1", headers=auth_header(token))

    assert listing.status_code == 403
    assert detail.status_code == 403


def test_catalog_filters_unknown_question_mark_herb(client, db_engine):
    _seed_entries(db_engine)
    session = Session(db_engine)
    try:
        session.add(
            KnowledgeEntry(
                entry_id="HERB_UNKNOWN_QUESTION_MARK",
                source_id="SRC_DRAGON_TCM",
                entry_type="herb",
                title="?",
                content="无法识别的源记录。",
                keywords_json=["?"],
                source_title="DragonTCM",
                source_type="open_dataset",
                metadata_json={
                    "display_name": "未知",
                    "aliases": ["?"],
                    "visibility": ["doctor", "student"],
                    "source_catalog": {"doctor_evidence_allowed": True, "student_visible": True},
                },
            )
        )
        session.commit()
    finally:
        session.close()
    register(client, "catalog_unknown_doctor", "doctor")
    token = login(client, "catalog_unknown_doctor", "doctor")

    response = client.get(
        "/api/knowledge/entries",
        headers=auth_header(token),
        params={"entry_type": "herb", "q": "未知"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["total"] == 0


def _seed_entries(db_engine) -> None:
    session = Session(db_engine)
    try:
        session.add_all(
            [
                KnowledgeEntry(
                    entry_id="HERB_1",
                    source_id="SRC_TCM_DATA_FOR_ML",
                    entry_type="herb",
                    title="黄芪",
                    content="来自已批准开源数据集的药材名称记录。",
                    keywords_json=["黄芪", "药材"],
                    source_title="TCMDataForML",
                    source_type="open_dataset",
                    metadata_json={
                        "visibility": ["doctor", "student"],
                        "aliases": ["黄耆"],
                        "license": "MIT",
                        "commit_hash": "abc123",
                        "source_records": [{"source_id": "SRC_TCM_DATA_FOR_ML", "source_record_id": "row-1", "source_file": "herbs.csv"}],
                        "chpa_efficacies": ["补气"],
                        "chpa_symptoms": ["乏力"],
                        "chpa_relation_note": "CHPA 统计关联，待医学复核。",
                        "source_catalog": {"doctor_evidence_allowed": True, "student_visible": True},
                    },
                ),
                KnowledgeEntry(
                    entry_id="HERB_GANCAO",
                    source_id="SRC_TCM_DATA_FOR_ML",
                    entry_type="herb",
                    title="甘草",
                    content="甘草药材目录记录。",
                    keywords_json=["甘草", "药材"],
                    source_title="TCMDataForML",
                    source_type="open_dataset",
                    metadata_json={
                        "visibility": ["doctor", "student"],
                        "license": "MIT",
                        "source_catalog": {"doctor_evidence_allowed": True, "student_visible": True},
                    },
                ),
                KnowledgeEntry(
                    entry_id="CASE_MTCMB_1",
                    source_id="SRC_MTCMB",
                    entry_type="case",
                    title="MTCMB 教学案例 1",
                    content="已脱敏教学案例。",
                    keywords_json=["MTCMB", "教学"],
                    source_title="MTCMB",
                    source_type="open_dataset",
                    metadata_json={
                        "visibility": "student_learning",
                        "license": "Apache-2.0",
                        "patient_name": "Real Patient",
                        "source_catalog": {"doctor_evidence_allowed": False, "student_visible": True},
                    },
                ),
            ]
        )
        session.commit()
    finally:
        session.close()
