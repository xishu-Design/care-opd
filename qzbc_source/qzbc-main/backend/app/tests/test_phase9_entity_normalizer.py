from app.data_pipeline.normalizers.tcm_entity_normalizer import normalize_entity_name, unique_normalized_names
from app.data_pipeline.normalizers.text_normalizer import normalize_text, split_candidate_text


def test_text_normalizer_trims_full_width_and_control_chars():
    assert normalize_text("  黄芪　\x00 ") == "黄芪"
    assert normalize_entity_name(" 黄 芪 ") == "黄芪"


def test_split_candidate_text_filters_empty_values():
    assert split_candidate_text("黄芪， 白术;;  ") == ["黄芪", "白术"]
    assert split_candidate_text("") == []


def test_unique_normalized_names_keeps_original_and_does_not_merge_different_names():
    items = unique_normalized_names([" 黄芪 ", "黄 芪", "白术"])

    assert items == [
        {"original_name": "黄芪", "normalized_name": "黄芪", "name": "黄芪"},
        {"original_name": "白术", "normalized_name": "白术", "name": "白术"},
    ]

