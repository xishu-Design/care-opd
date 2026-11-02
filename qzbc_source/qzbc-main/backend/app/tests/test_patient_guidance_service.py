from app.services.patient_guidance_service import (
    food_interactions_for_herbs,
    herb_incompatibilities_for_prescription,
)


def test_food_interaction_catalog_matches_exact_prescribed_herb():
    matches = food_interactions_for_herbs(["人参", "茯苓"])

    assert len(matches) == 1
    assert matches[0]["herb_name"] == "人参"
    assert matches[0]["food_items"] == ["萝卜"]
    assert matches[0]["relation_label"] == "传统服药食忌参考"
    assert matches[0]["source_id"] == "SRC_NMPA_DIET_EDU_2020"


def test_food_interaction_catalog_does_not_infer_unlisted_foods():
    assert food_interactions_for_herbs(["川贝母", "款冬花", "百合", "麦冬", "梨子"]) == []


def test_herb_incompatibility_list_reuses_review_standard():
    matches = herb_incompatibilities_for_prescription(["甘草"])

    assert matches
    assert all(item["source_id"] == "SRC_DB4403_T142_2021" for item in matches)
    assert {item["relation_label"] for item in matches} <= {"十八反（传统配伍禁忌）", "十九畏（传统配伍禁忌）"}
