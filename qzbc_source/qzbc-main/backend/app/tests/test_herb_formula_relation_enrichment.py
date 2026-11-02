import json

from app.data_pipeline.scripts.enrich_herb_formula_relations import enrich_herb_formula_relations
from app.services.prescription_suggestion_service import _catalog_match_score


def test_formula_relations_are_projected_as_lower_weight_indirect_evidence(tmp_path):
    (tmp_path / "herbs.json").write_text(
        json.dumps({"items": [{"entry_id": "HERB_1", "name": "白术", "display_name": "白术", "keywords": []}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "formulas.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "entry_id": "FORMULA_1",
                        "name": "四君子汤",
                        "display_name": "四君子汤",
                        "source_ids": ["SRC_TCM_MKG"],
                        "syndromes": ["脾胃气虚证"],
                        "symptoms": ["食少便溏"],
                        "display_functions": ["益气健脾"],
                        "components": [{"normalized_name": "白术"}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = enrich_herb_formula_relations(tmp_path)
    herb = json.loads((tmp_path / "herbs.json").read_text(encoding="utf-8"))["items"][0]
    assert report["herbs_with_formula_relations"] == 1
    assert herb["indirect_syndromes"] == ["脾胃气虚证"]
    assert herb["indirect_symptoms"] == ["食少便溏"]
    assert herb["related_formulas"][0]["formula_name"] == "四君子汤"
    assert "不表示该药材可单独治疗" in herb["formula_relation_note"]

    indirect_score, indirect_reasons = _catalog_match_score(
        "脾胃气虚",
        "白术",
        [],
        [],
        [],
        indirect_syndromes=herb["indirect_syndromes"],
    )
    direct_score, _ = _catalog_match_score("脾胃气虚", "白术", [], [], [], syndromes=["脾胃气虚证"])
    assert indirect_score > 0
    assert indirect_score < direct_score
    assert indirect_reasons == ["经方剂关联证候：脾胃气虚证"]
