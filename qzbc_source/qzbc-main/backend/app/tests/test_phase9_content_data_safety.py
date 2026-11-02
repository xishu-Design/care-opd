import json

from app.content.validators.content_validator import ContentValidator


def test_content_validator_rejects_patient_unsafe_content(tmp_path):
    content_dir = _base_content(tmp_path)
    _write(
        content_dir / "herbs.json",
        {
            "items": [
                {
                    "entry_id": "HERB_1",
                    "name": "示例",
                    "source_ids": ["SRC_TCM_DATA_FOR_ML"],
                    "visibility": ["patient"],
                    "license_note": "可以自行开方",
                }
            ]
        },
    )

    report = ContentValidator(content_dir).validate()

    assert not report.valid
    assert "PATIENT_UNSAFE_TERM" in {issue.code for issue in report.errors}


def test_content_validator_rejects_non_deidentified_learning_case(tmp_path):
    content_dir = _base_content(tmp_path)
    _write(
        content_dir / "learning_cases.json",
        {
            "items": [
                {
                    "case_id": "CASE_1",
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "visibility": "student_learning",
                    "deidentified": False,
                    "prescription": {"items": []},
                }
            ]
        },
    )

    report = ContentValidator(content_dir).validate()

    assert not report.valid
    assert "LEARNING_CASE_NOT_DEIDENTIFIED" in {issue.code for issue in report.errors}


def _base_content(tmp_path):
    content_dir = tmp_path / "content" / "data"
    content_dir.mkdir(parents=True)
    _write(
        content_dir / "sources.json",
        {
            "items": [
                {
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "source_title": "TCMDataForML",
                    "license": "MIT",
                }
            ]
        },
    )
    for file_name in ("herbs.json", "formulas.json", "learning_cases.json", "quizzes.json"):
        _write(content_dir / file_name, {"items": []})
    return content_dir


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

