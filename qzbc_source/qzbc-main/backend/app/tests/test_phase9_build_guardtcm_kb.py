import json

from app.content.validators.content_validator import ContentValidator
from app.data_pipeline.scripts.build_guardtcm_kb import build_guardtcm_kb


def test_build_guardtcm_kb_generates_content_data(tmp_path):
    processed_dir = _processed_fixture(tmp_path)
    content_dir = tmp_path / "content" / "data"
    report = build_guardtcm_kb(processed_dir=processed_dir, content_data_dir=content_dir, reports_dir=tmp_path / "reports")

    assert report["output_counts"]["herbs"] == 3
    assert report["output_counts"]["formulas"] == 1
    assert report["output_counts"]["learning_cases"] == 1
    assert report["output_counts"]["quizzes"] == 1
    for file_name in ("sources.json", "herbs.json", "formulas.json", "learning_cases.json", "quizzes.json", "source_catalog.json"):
        assert (content_dir / file_name).exists()

    validation = ContentValidator(content_dir).validate()
    assert validation.valid, validation.to_dict()


def test_build_guardtcm_kb_does_not_infer_medical_fields(tmp_path):
    processed_dir = _processed_fixture(tmp_path)
    content_dir = tmp_path / "content" / "data"
    build_guardtcm_kb(processed_dir=processed_dir, content_data_dir=content_dir, reports_dir=tmp_path / "reports")

    herbs = _items(content_dir / "herbs.json")
    formulas = _items(content_dir / "formulas.json")

    assert all(item["functions"] == [] for item in herbs)
    assert all(item["cautions"] == [] for item in herbs)
    assert all(item["dose_range"] is None for item in herbs)
    assert all(item["functions"] == [] for item in formulas)
    assert all(item["cautions"] == [] for item in formulas)
    assert "clinical recommendation" in formulas[0]["usage_note"]


def test_build_guardtcm_kb_generates_safe_learning_case_and_quiz(tmp_path):
    processed_dir = _processed_fixture(tmp_path)
    content_dir = tmp_path / "content" / "data"
    build_guardtcm_kb(processed_dir=processed_dir, content_data_dir=content_dir, reports_dir=tmp_path / "reports")

    learning_cases = _items(content_dir / "learning_cases.json")
    quizzes = _items(content_dir / "quizzes.json")
    herb_names = {item["name"] for item in _items(content_dir / "herbs.json")}

    assert learning_cases[0]["deidentified"] is True
    assert learning_cases[0]["visibility"] == "student_learning"
    assert learning_cases[0]["expected_risks"] == []
    assert quizzes[0]["answer"] in quizzes[0]["options"]
    assert set(quizzes[0]["options"]).issubset(herb_names)
    assert quizzes[0]["knowledge_points"] == ["处方组成识别"]


def test_build_guardtcm_kb_repeat_run_is_stable_for_content_files(tmp_path):
    processed_dir = _processed_fixture(tmp_path)
    content_dir = tmp_path / "content" / "data"
    build_guardtcm_kb(processed_dir=processed_dir, content_data_dir=content_dir, reports_dir=tmp_path / "reports")
    first = {name: (content_dir / name).read_text(encoding="utf-8") for name in ("sources.json", "herbs.json", "formulas.json", "learning_cases.json", "quizzes.json")}
    build_guardtcm_kb(processed_dir=processed_dir, content_data_dir=content_dir, reports_dir=tmp_path / "reports")
    second = {name: (content_dir / name).read_text(encoding="utf-8") for name in first}

    assert first == second


def _processed_fixture(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _write(
        processed_dir / "source_catalog.json",
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
        processed_dir / "herb_candidates.json",
        {
            "items": [
                _herb("HERB_CAND_1", "黄芪", "row_1"),
                _herb("HERB_CAND_2", "黄 芪", "row_2"),
                _herb("HERB_CAND_3", "白术", "row_3"),
                _herb("HERB_CAND_4", "茯苓", "row_4"),
            ]
        },
    )
    _write(
        processed_dir / "formula_candidates.json",
        {
            "items": [
                {
                    "candidate_id": "FORMULA_CAND_1",
                    "name": "source_formula_000001",
                    "normalized_name": "source_formula_000001",
                    "components": [
                        {"herb_name": "黄芪", "normalized_name": "黄芪", "dosage": None, "unit": None},
                        {"herb_name": "白术", "normalized_name": "白术", "dosage": None, "unit": None},
                    ],
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "source_record_id": "row_10",
                    "source_file": "TCM.csv",
                    "license": "MIT",
                }
            ]
        },
    )
    _write(
        processed_dir / "learning_case_candidates.json",
        {
            "items": [
                {
                    "case_candidate_id": "CASE_CAND_1",
                    "title": "candidate",
                    "deidentified": True,
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "source_record_id": "line_1",
                    "case_type": "prescription_composition_learning_candidate",
                    "patient_profile": {"age_range": None, "gender": None},
                    "prescription": {"items": [{"herb_name": "黄芪", "dosage": None}], "usage": None},
                    "expected_risks": [],
                    "knowledge_points": ["处方组成识别"],
                    "review_status": "manual_review_required",
                    "visibility": "student_learning",
                    "license": "MIT",
                }
            ]
        },
    )
    return processed_dir


def _herb(candidate_id, name, record_id):
    return {
        "candidate_id": candidate_id,
        "name": name.replace(" ", ""),
        "normalized_name": name.replace(" ", ""),
        "original_name": name,
        "aliases": [],
        "source_id": "SRC_TCM_DATA_FOR_ML",
        "source_record_id": record_id,
        "source_file": "IndexName.csv",
        "license": "MIT",
    }


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _items(path):
    return json.loads(path.read_text(encoding="utf-8"))["items"]

