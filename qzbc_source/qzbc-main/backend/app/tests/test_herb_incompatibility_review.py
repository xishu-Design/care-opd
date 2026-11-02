import json

from app.db.models import Prescription, PrescriptionItem, User
from app.lightrag_integration.local_graph import local_graph
from app.rules.rule_engine import run_structured_prescription_review
from app.rules.rule_loader import load_rules
from app.services.herb_incompatibility_service import (
    expanded_incompatibility_pairs,
    find_herb_incompatibilities,
    incompatibilities_for_herb,
    load_herb_incompatibilities,
)


def test_source_bound_incompatibility_dataset_has_verified_groups_and_pairs():
    payload = load_herb_incompatibilities()
    categories = [item["category"] for item in payload["constraints"]]

    assert payload["source"]["source_id"] == "SRC_DB4403_T142_2021"
    assert payload["source"]["sha256"] == "454ec4a03d1d2e506db7e50288485790ceb0e6be6be9913a31b0ac96e8bf8c72"
    assert categories.count("十八反") == 3
    assert categories.count("十九畏") == 9
    assert len(expanded_incompatibility_pairs()) == 75
    assert "Dryad" not in json.dumps(payload, ensure_ascii=False)


def test_exact_matching_finds_eighteen_antagonisms_and_nineteen_incompatibilities():
    eighteen = find_herb_incompatibilities(["甘草", "甘遂", "人参"])
    nineteen = find_herb_incompatibilities(["巴豆", "牵牛子"])
    alias = find_herb_incompatibilities(["硫磺", "朴硝"])

    assert [(item["herb_a"], item["herb_b"], item["category"]) for item in eighteen] == [("甘草", "甘遂", "十八反")]
    assert [(item["herb_a"], item["herb_b"], item["category"]) for item in nineteen] == [("巴豆", "牵牛子", "十九畏")]
    assert [(item["herb_a"], item["herb_b"], item["category"]) for item in alias] == [("硫磺", "朴硝", "十九畏")]
    assert find_herb_incompatibilities(["人参", "丹参"]) == []


def test_catalog_lookup_lists_only_source_bound_incompatible_partner_herbs():
    gancao = incompatibilities_for_herb("甘草")

    assert {(item["herb_name"], item["category"]) for item in gancao} == {
        ("大戟", "十八反"),
        ("甘遂", "十八反"),
        ("海藻", "十八反"),
        ("芫花", "十八反"),
    }
    assert incompatibilities_for_herb("黄芪") == []


def test_structured_review_triggers_red_source_bound_pair_finding(db_session):
    doctor = User(username="pair_doctor", hashed_password="test", role="doctor", display_name="Doctor")
    patient = User(username="pair_patient", hashed_password="test", role="patient", display_name="Patient")
    db_session.add_all([doctor, patient])
    db_session.flush()
    prescription = Prescription(
        doctor_id=doctor.id,
        patient_id=patient.id,
        title="传统配伍禁忌测试",
        usage_method="每日一剂，水煎服",
        items=[
            PrescriptionItem(herb_name="附子", dose="3", unit="g"),
            PrescriptionItem(herb_name="半夏", dose="6", unit="g"),
        ],
    )
    db_session.add(prescription)
    db_session.flush()
    load_rules(db_session)

    result = run_structured_prescription_review(db_session, prescription)

    finding = next(item for item in result["rule_findings"] if item["rule_id"] == "RULE_TRADITIONAL_HERB_INCOMPATIBILITY")
    assert result["risk_level"] == "red"
    assert finding["source_id"] == "SRC_DB4403_T142_2021"
    assert finding["matched_pairs"][0]["herb_a"] == "附子"
    assert finding["matched_pairs"][0]["herb_b"] == "半夏"
    assert finding["matched_pairs"][0]["category"] == "十八反"
    assert result["herb_incompatibility_matches"] == finding["matched_pairs"]
    assert "附子—半夏（十八反）" in result["review_summary"]


def test_local_graph_projects_traditional_incompatibility_in_chinese_ui_relation(tmp_path):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    (content_dir / "herbs.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "entry_id": "HERB_TEST_GANCAO",
                        "name": "甘草",
                        "source_ids": ["SRC_DB4403_T142_2021"],
                        "visibility": ["doctor", "student"],
                    },
                    {
                        "entry_id": "HERB_TEST_GANSUI",
                        "name": "甘遂",
                        "source_ids": ["SRC_DB4403_T142_2021"],
                        "visibility": ["doctor", "student"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source_payload = load_herb_incompatibilities()
    source_payload["constraints"] = [source_payload["constraints"][0]]
    (content_dir / "herb_incompatibilities.json").write_text(json.dumps(source_payload, ensure_ascii=False), encoding="utf-8")
    for file_name in ("formulas.json", "sources.json", "relationships.json"):
        (content_dir / file_name).write_text('{"items":[]}', encoding="utf-8")

    graph = local_graph(label="甘草", role="doctor", max_depth=1, max_nodes=50, content_dir=content_dir)
    labels = {node.id: node.label for node in graph.nodes}
    conflict_edges = [edge for edge in graph.edges if edge.label == "TRADITIONAL_INCOMPATIBILITY_18"]

    assert any({labels[edge.source], labels[edge.target]} == {"甘草", "甘遂"} for edge in conflict_edges)
    assert all(edge.source_ids == ["SRC_DB4403_T142_2021"] for edge in conflict_edges)
