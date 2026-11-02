from app.data_pipeline.validators.processed_item_validator import ProcessedItemValidator


def test_processed_item_validator_accepts_traceable_candidate():
    item = {
        "candidate_id": "HERB_CAND_SRC_000001",
        "name": "黄芪",
        "source_id": "SRC_TCM_DATA_FOR_ML",
        "source_record_id": "row_1",
        "license": "MIT",
    }

    assert ProcessedItemValidator().is_valid(item)


def test_processed_item_validator_rejects_missing_source_id():
    item = {"candidate_id": "HERB_CAND_SRC_000001", "name": "黄芪", "source_record_id": "row_1", "license": "MIT"}

    issues = ProcessedItemValidator().validate_candidate(item)

    assert "MISSING_SOURCE_ID" in {issue.code for issue in issues}


def test_processed_item_validator_rejects_missing_source_record_id():
    item = {"candidate_id": "HERB_CAND_SRC_000001", "name": "黄芪", "source_id": "SRC_TCM_DATA_FOR_ML", "license": "MIT"}

    issues = ProcessedItemValidator().validate_candidate(item)

    assert "MISSING_SOURCE_RECORD_ID" in {issue.code for issue in issues}


def test_processed_item_validator_rejects_phone_in_learning_case():
    item = {
        "case_candidate_id": "CASE_CAND_SRC_000001",
        "source_id": "SRC_TCM_DATA_FOR_ML",
        "source_record_id": "line_1",
        "license": "MIT",
        "prescription": {"items": [{"herb_name": "黄芪"}]},
        "notes": "联系电话 13812345678",
    }

    issues = ProcessedItemValidator().validate_candidate(item, candidate_id_field="case_candidate_id")

    assert "PHONE_NUMBER_DETECTED" in {issue.code for issue in issues}


def test_processed_item_validator_rejects_patient_unsafe_term():
    item = {
        "candidate_id": "PRESC_CAND_SRC_000001",
        "source_id": "SRC_TCM_DATA_FOR_ML",
        "source_record_id": "line_1",
        "license": "MIT",
        "herbs": ["黄芪"],
        "notes": "患者可以自行开方",
    }

    issues = ProcessedItemValidator().validate_candidate(item)

    assert "PATIENT_UNSAFE_TERM" in {issue.code for issue in issues}

