import json

import pytest
from sqlalchemy import select

from app.content.services.content_import_service import ContentImportError, ContentImportService
from app.db.models import KnowledgeEntry, KnowledgeSource, LearningCase, Quiz, RuleDefinition


def test_content_import_service_imports_content_data(db_session, tmp_path):
    content_dir = _content_fixture(tmp_path)
    report = ContentImportService(db_session, content_dir).import_content()

    assert report["status"] == "completed"
    assert report["success_count"] == 5
    assert db_session.scalar(select(KnowledgeSource).where(KnowledgeSource.source_id == "SRC_TCM_DATA_FOR_ML")) is not None
    assert db_session.scalar(select(KnowledgeEntry).where(KnowledgeEntry.entry_type == "herb")).title == "黄芪"
    assert db_session.scalar(select(KnowledgeEntry).where(KnowledgeEntry.entry_type == "formula")) is not None
    case = db_session.scalar(select(LearningCase))
    assert case is not None
    assert case.source_prescription_id is None
    assert case.anonymized_case["visibility"] == "student_learning"
    assert db_session.scalar(select(Quiz).where(Quiz.quiz_id == "QUIZ_SRC_TCM_DATA_FOR_ML_000001")) is not None


def test_content_import_service_is_idempotent(db_session, tmp_path):
    content_dir = _content_fixture(tmp_path)
    service = ContentImportService(db_session, content_dir)

    service.import_content()
    service.import_content()

    assert len(db_session.scalars(select(KnowledgeSource)).all()) == 1
    assert len(db_session.scalars(select(KnowledgeEntry)).all()) == 2
    assert len(db_session.scalars(select(LearningCase)).all()) == 1
    assert len(db_session.scalars(select(Quiz)).all()) == 1


def test_content_import_service_imports_expanded_content_types(db_session, tmp_path):
    content_dir = _content_fixture(tmp_path)
    _write(
        content_dir / "contraindications.json",
        {
            "items": [
                {
                    "entry_id": "CONTRA_SRC_TCM_DATA_FOR_ML_000001",
                    "title": "来源绑定风险提示",
                    "risk_type": "source_bound_demo",
                    "risk_level": "yellow",
                    "message_for_doctor": "该条仅用于测试来源绑定。",
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "keywords": ["来源绑定", "风险"],
                    "visibility": ["doctor", "student"],
                }
            ]
        },
    )
    _write(
        content_dir / "diet_precautions.json",
        {
            "items": [
                {
                    "entry_id": "DIET_SRC_TCM_DATA_FOR_ML_000001",
                    "title": "来源绑定饮食注意",
                    "message": "仅用于提示遵循医生说明。",
                    "forbidden_as_treatment": True,
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "keywords": ["饮食", "注意"],
                    "visibility": ["patient", "doctor"],
                }
            ]
        },
    )
    _write(
        content_dir / "public_education.json",
        {
            "items": [
                {
                    "entry_id": "EDU_SRC_TCM_DATA_FOR_ML_000001",
                    "topic": "来源绑定科普",
                    "safe_answer": "该条仅用于一般科普展示，具体用药请遵循医生说明。",
                    "forbidden_claims": ["自行开方"],
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "keywords": ["科普", "来源"],
                    "visibility": ["patient"],
                }
            ]
        },
    )
    _write(
        content_dir / "review_rules.json",
        {
            "items": [
                {
                    "rule_id": "RULE_PHASE9_CONTENT_SOURCE",
                    "rule_type": "content_source_demo",
                    "rule_category": "medical_review",
                    "name": "来源绑定审方规则",
                    "enabled": True,
                    "risk_level": "yellow",
                    "applies_to_roles": ["doctor", "student"],
                    "condition": {"field_missing": "usage"},
                    "message": "该条仅用于测试规则来源绑定。",
                    "evidence_query": "来源绑定",
                    "requires_doctor_confirmation": True,
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "tags": ["来源绑定"],
                }
            ]
        },
    )

    report = ContentImportService(db_session, content_dir).import_content()

    assert report["success_count"] == 9
    assert db_session.scalar(select(KnowledgeEntry).where(KnowledgeEntry.entry_type == "contraindication")) is not None
    assert db_session.scalar(select(KnowledgeEntry).where(KnowledgeEntry.entry_type == "diet_precaution")) is not None
    assert db_session.scalar(select(KnowledgeEntry).where(KnowledgeEntry.entry_type == "public_education")) is not None
    rule = db_session.scalar(select(RuleDefinition).where(RuleDefinition.rule_id == "RULE_PHASE9_CONTENT_SOURCE"))
    assert rule is not None
    assert rule.source_id == "SRC_TCM_DATA_FOR_ML"


def test_content_import_service_rejects_unsafe_content(db_session, tmp_path):
    content_dir = _content_fixture(tmp_path)
    herbs_path = content_dir / "herbs.json"
    herbs = json.loads(herbs_path.read_text(encoding="utf-8"))
    herbs["items"][0]["license_note"] = "可以自行开方"
    herbs_path.write_text(json.dumps(herbs, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ContentImportError):
        ContentImportService(db_session, content_dir).import_content()

    assert db_session.scalar(select(KnowledgeEntry)) is None


def test_content_import_retires_stale_unlabelled_tcmeval_rows_without_deleting_history(db_session, tmp_path):
    stale = LearningCase(
        title="无标签 TCMEval 测试行",
        anonymized_case={
            "case_id": "TCMEVAL_SDT_病例99",
            "source_id": "SRC_TCMEVAL_SDT",
            "visibility": "student_learning",
            "deidentified": True,
            "training_payload": {"expected_syndromes": []},
        },
        standard_analysis="来源未提供答案。",
    )
    db_session.add(stale)
    db_session.commit()

    ContentImportService(db_session, _content_fixture(tmp_path)).import_content()
    db_session.commit()
    db_session.refresh(stale)

    assert stale.anonymized_case["visibility"] == "retired_unlabelled_source"
    assert stale.anonymized_case["review_status"] == "retired_unlabelled_benchmark_row"


def _content_fixture(tmp_path):
    content_dir = tmp_path / "content" / "data"
    content_dir.mkdir(parents=True)
    _write(
        content_dir / "sources.json",
        {
            "items": [
                {
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "source_title": "TCMDataForML",
                    "source_type": "open_source_dataset",
                    "source_version": "abc123",
                    "source_year": "",
                    "evidence_level": "open_dataset_extracted",
                    "license": "MIT",
                    "license_note": "Derived from approved open-source dataset.",
                    "repository_url": "https://github.com/GuoxiangZu/TCMDataForML",
                    "processing_script": "extract_tcmdataforml.py",
                    "processing_version": "0.9.0",
                }
            ]
        },
    )
    _write(
        content_dir / "source_catalog.json",
        {
            "sources": [
                {
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "name": "TCMDataForML",
                    "repository_url": "https://github.com/GuoxiangZu/TCMDataForML",
                    "license": "MIT",
                    "commit_hash": "abc123",
                    "downloaded_at": "2026-07-07T00:00:00+00:00",
                    "processing_script": "extract_tcmdataforml.py",
                    "processing_version": "0.9.0",
                }
            ]
        },
    )
    _write(
        content_dir / "herbs.json",
        {
            "items": [
                {
                    "entry_id": "HERB_TCM_DATA_FOR_ML_000001",
                    "name": "黄芪",
                    "normalized_name": "黄芪",
                    "aliases": [],
                    "source_ids": ["SRC_TCM_DATA_FOR_ML"],
                    "source_records": [{"source_id": "SRC_TCM_DATA_FOR_ML", "source_record_id": "row_1", "source_file": "IndexName.csv"}],
                    "functions": [],
                    "cautions": [],
                    "dose_range": None,
                    "keywords": ["黄芪"],
                    "visibility": ["doctor", "student"],
                    "evidence_level": "open_dataset_extracted",
                    "license_note": "Derived from approved open-source dataset. Function/caution/dose not inferred.",
                }
            ]
        },
    )
    _write(
        content_dir / "formulas.json",
        {
            "items": [
                {
                    "entry_id": "FORMULA_TCM_DATA_FOR_ML_000001",
                    "name": "source_formula_000001",
                    "normalized_name": "source_formula_000001",
                    "components": [{"herb_name": "黄芪", "normalized_name": "黄芪", "dosage": None, "unit": None}],
                    "source_ids": ["SRC_TCM_DATA_FOR_ML"],
                    "source_records": [{"source_id": "SRC_TCM_DATA_FOR_ML", "source_record_id": "row_10", "source_file": "TCM.csv"}],
                    "functions": [],
                    "cautions": [],
                    "usage_note": "Open-source prescription composition candidate. Not a clinical recommendation.",
                    "keywords": ["黄芪", "处方组成"],
                    "visibility": ["doctor", "student"],
                    "evidence_level": "open_dataset_extracted",
                }
            ]
        },
    )
    _write(
        content_dir / "learning_cases.json",
        {
            "items": [
                {
                    "case_id": "CASE_TCM_DATA_FOR_ML_000001",
                    "title": "开源处方组成识别训练案例 000001",
                    "case_type": "prescription_composition_learning",
                    "visibility": "student_learning",
                    "deidentified": True,
                    "patient_profile": {"age_range": None, "gender": None},
                    "prescription": {"items": [{"herb_name": "黄芪", "dosage": None}], "usage": None},
                    "expected_risks": [],
                    "standard_analysis": "该案例仅用于训练处方组成识别，不包含审方风险结论。",
                    "knowledge_points": ["处方组成识别"],
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "source_record_id": "line_1",
                    "review_status": "auto_generated_learning_candidate",
                }
            ]
        },
    )
    _write(
        content_dir / "quizzes.json",
        {
            "items": [
                {
                    "quiz_id": "QUIZ_SRC_TCM_DATA_FOR_ML_000001",
                    "case_id": "CASE_TCM_DATA_FOR_ML_000001",
                    "question_type": "single_choice",
                    "question": "该处方候选中包含以下哪味药材？",
                    "options": ["黄芪", "白术"],
                    "answer": "黄芪",
                    "explanation": "该题仅考察处方组成识别，答案来自开源数据源中的处方组成字段。",
                    "knowledge_points": ["处方组成识别"],
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                }
            ]
        },
    )
    for file_name in ("contraindications.json", "diet_precautions.json", "public_education.json", "review_rules.json"):
        _write(content_dir / file_name, {"items": []})
    return content_dir


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
