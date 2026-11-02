import json

from app.core.config import get_settings
from app.db.models import KnowledgeEntry, User
from app.knowledge.ingestion import get_knowledge_entry, list_knowledge_entries
from app.knowledge.internal_knowledge import import_shennong_jsonl
from app.lightrag_integration.local_graph import local_graph
from app.services.patient_service import create_or_bind_patient
from app.services.prescription_suggestion_service import suggest_prescription_candidates
from app.schemas.patient import PatientCreateRequest


def test_shennong_import_creates_internal_documents_and_catalog_without_polluting_deterministic_graph(db_session, tmp_path):
    path = tmp_path / "shennong.jsonl"
    path.write_text(json.dumps({"query": "黄芪有哪些功效？", "response": "黄芪具有补气升阳、固表止汗功效，适用于气虚。"}, ensure_ascii=False) + "\n", encoding="utf-8")
    _seed_herb(db_session, "黄芪", functions=["补气升阳"])

    report = import_shennong_jsonl(db_session, path=path, revision="rev-test", allow_internal=True)
    db_session.commit()
    catalog = list_knowledge_entries(db_session, role="doctor", entry_type="herb", query="黄芪")
    detail = get_knowledge_entry(db_session, role="doctor", source_id="SRC_TEST", entry_id="HERB_黄芪")
    graph = local_graph(label="黄芪", role="doctor", max_depth=2, max_nodes=50, db=db_session)

    assert report["documents"] == 1
    assert report["claims"] >= 1
    assert catalog["items"][0]["functions"]
    assert detail and detail["metadata"]["claims"]
    # The visual graph is built from versioned catalog JSON, not from database
    # claims extracted from a ShenNong Q&A response. Static catalog efficacy
    # nodes are allowed; dynamic internal-claim nodes are not.
    assert all(node.type not in {"symptom", "syndrome", "case"} for node in graph.nodes)
    assert all("补气升阳、固表止汗功效，适用于气虚" not in node.label for node in graph.nodes)
    assert all(edge.source != "HERB_黄芪" and edge.target != "HERB_黄芪" for edge in graph.edges)


def test_doctor_suggestions_are_candidate_only_without_dose(db_session, tmp_path):
    path = tmp_path / "shennong.jsonl"
    path.write_text(json.dumps({"query": "黄芪有哪些功效？", "response": "黄芪具有补气升阳功效。"}, ensure_ascii=False) + "\n", encoding="utf-8")
    _seed_herb(db_session, "黄芪", functions=["补气升阳"])
    import_shennong_jsonl(db_session, path=path, revision="rev-test", allow_internal=True)
    doctor = User(username="suggestion_doctor", hashed_password="test", role="doctor", display_name="Doctor")
    db_session.add(doctor)
    db_session.flush()
    patient = create_or_bind_patient(db_session, doctor, PatientCreateRequest(username="suggestion_patient", password="password123", display_name="Patient", current_symptoms="气虚"))

    result = suggest_prescription_candidates(db_session, doctor, patient_id=patient.id, query="气虚")

    assert result["requires_human_confirm"] is True
    assert result["suggestions"]
    assert result["suggestions"][0]["no_dose_or_usage"] is True
    assert "dose" not in result["suggestions"][0]
    assert all("SRC_SHENNONG_TCM_DATASET" not in item["source_ids"] for item in result["suggestions"])


def test_doctor_suggestions_include_catalog_only_chinese_efficacy(db_session):
    db_session.add(
        KnowledgeEntry(
            entry_id="HERB_明目测试药",
            source_id="SRC_TEST",
            entry_type="herb",
            title="明目测试药",
            content="目录药材",
            keywords_json=["明目测试药"],
            source_title="Test",
            source_type="fixture",
            metadata_json={"visibility": ["doctor"], "display_functions": ["明目"]},
        )
    )
    doctor = User(username="catalog_suggestion_doctor", hashed_password="test", role="doctor", display_name="Doctor")
    db_session.add(doctor)
    db_session.flush()
    patient = create_or_bind_patient(db_session, doctor, PatientCreateRequest(username="catalog_suggestion_patient", password="password123", display_name="Patient"))

    result = suggest_prescription_candidates(db_session, doctor, patient_id=patient.id, query="患者眼睛疲劳")

    assert any(item["herb_name"] == "明目测试药" and "明目" in item["functions"] for item in result["suggestions"])


def test_doctor_suggestions_exclude_patient_allergy_herb(db_session):
    _seed_herb(db_session, "丹参", functions=["活血"])
    _seed_herb(db_session, "川芎", functions=["活血"])
    doctor = User(username="allergy_suggestion_doctor", hashed_password="test", role="doctor", display_name="Doctor")
    db_session.add(doctor)
    db_session.flush()
    patient = create_or_bind_patient(
        db_session,
        doctor,
        PatientCreateRequest(
            username="allergy_suggestion_patient",
            password="password123",
            display_name="Patient",
            allergy_history="患者对丹参过敏",
        ),
    )

    result = suggest_prescription_candidates(db_session, doctor, patient_id=patient.id, query="活血")

    names = {item["herb_name"] for item in result["suggestions"]}
    assert "丹参" not in names
    assert "川芎" in names
    assert result["excluded_allergy_herbs"] == ["丹参"]
    assert result["allergy_history"] == "患者对丹参过敏"


def test_doctor_suggestions_use_low_weight_chpa_statistical_relations(db_session):
    db_session.add(
        KnowledgeEntry(
            entry_id="HERB_CHPA_TEST",
            source_id="SRC_DRYAD_TCM_NETWORK",
            entry_type="herb",
            title="桑叶",
            content="CHPA 统计关联测试",
            keywords_json=["桑叶", "眼疲劳"],
            source_title="CHPA",
            source_type="fixture",
            metadata_json={
                "visibility": ["doctor", "student"],
                "chpa_efficacies": ["明目"],
                "chpa_symptoms": ["眼疲劳"],
                "relation_source_ids": ["SRC_DRYAD_TCM_NETWORK"],
                "chpa_translation_review_status": "doctor_reviewed",
            },
        )
    )
    doctor = User(username="chpa_suggestion_doctor", hashed_password="test", role="doctor", display_name="Doctor")
    db_session.add(doctor)
    db_session.flush()
    patient = create_or_bind_patient(db_session, doctor, PatientCreateRequest(username="chpa_suggestion_patient", password="password123", display_name="Patient"))

    result = suggest_prescription_candidates(db_session, doctor, patient_id=patient.id, query="眼疲劳")

    candidate = next(item for item in result["suggestions"] if item["herb_name"] == "桑叶")
    assert "眼疲劳" in candidate["symptoms"]
    assert "chpa_statistical_association_pending_review" in candidate["risk_labels"]
    assert any("CHPA统计症状关联" in reason for reason in candidate["match_reasons"])


def test_local_graph_projects_chpa_relations_with_separate_edge_types(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "herbs.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "entry_id": "HERB_CHPA_GRAPH",
                        "name": "桑叶",
                        "display_name": "桑叶",
                        "source_ids": ["SRC_DRYAD_TCM_NETWORK"],
                        "visibility": ["doctor", "student"],
                        "chpa_efficacies": ["明目"],
                        "chpa_symptoms": ["眼疲劳"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for file_name in ("formulas.json", "sources.json", "relationships.json"):
        (content_dir / file_name).write_text('{"items":[]}', encoding="utf-8")

    graph = local_graph(label="桑叶", role="doctor", max_depth=2, max_nodes=50, content_dir=content_dir)

    assert {node.label for node in graph.nodes} >= {"桑叶", "明目", "眼疲劳"}
    assert {edge.label for edge in graph.edges} >= {"HAS_CHPA_EFFICACY", "ASSOCIATED_WITH_CHPA_SYMPTOM"}


def test_local_graph_corrects_known_catalog_input_variant():
    graph = local_graph(label="麝虫", role="doctor", max_depth=2, max_nodes=50)

    # “麝虫” is a common input variant, while the approved catalog entity is
    # “麝香”.  The correction is explicit in metadata so callers never have to
    # silently substitute a medical entity in the UI.
    assert graph.meta["query_correction"] == "麝香"
    assert any(node.label == "麝香" for node in graph.nodes)


def test_local_graph_excludes_internal_claim_relations(db_session, tmp_path):
    path = tmp_path / "shennong.jsonl"
    rows = [
        {"query": "黄芪有什么功效？", "response": "黄芪具有补气功效。"},
        {"query": "黄芪有哪些传统功效？", "response": "黄芪常见功效为补气。"},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    _seed_herb(db_session, "黄芪")
    import_shennong_jsonl(db_session, path=path, revision="rev-test", allow_internal=True)

    graph = local_graph(label="黄芪", role="doctor", max_depth=2, max_nodes=50, db=db_session)
    assert all(edge.source != "HERB_黄芪" and edge.target != "HERB_黄芪" for edge in graph.edges)


def _seed_herb(db_session, name: str, functions: list[str] | None = None) -> None:
    db_session.add(KnowledgeEntry(entry_id=f"HERB_{name}", source_id="SRC_TEST", entry_type="herb", title=name, content=name, keywords_json=[name], source_title="Test", source_type="fixture", metadata_json={"visibility": ["doctor", "student"], "license": "MIT", "display_functions": functions or []}))
    db_session.flush()
