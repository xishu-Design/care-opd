from app.data_pipeline.validators.duplicate_validator import merge_by_key


def test_merge_by_key_merges_sources_records_and_keywords():
    items = [
        {
            "normalized_name": "黄芪",
            "source_ids": ["SRC_A"],
            "source_records": [{"source_id": "SRC_A", "source_record_id": "row_1"}],
            "keywords": ["黄芪"],
        },
        {
            "normalized_name": "黄芪",
            "source_ids": ["SRC_A", "SRC_B"],
            "source_records": [{"source_id": "SRC_B", "source_record_id": "row_2"}],
            "keywords": ["补充词"],
        },
    ]

    merged = merge_by_key(items, "normalized_name")

    assert len(merged) == 1
    assert merged[0]["source_ids"] == ["SRC_A", "SRC_B"]
    assert merged[0]["source_records"] == [
        {"source_id": "SRC_A", "source_record_id": "row_1"},
        {"source_id": "SRC_B", "source_record_id": "row_2"},
    ]
    assert merged[0]["keywords"] == ["黄芪", "补充词"]

