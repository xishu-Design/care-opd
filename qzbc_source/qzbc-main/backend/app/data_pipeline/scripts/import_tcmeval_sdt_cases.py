"""Import the licensed TCMEval-SDT benchmark as student syndrome cases.

TCMEval-SDT is CC BY 4.0 and contains deidentified clinical descriptions plus
the benchmark's TCM pathogenesis/syndrome labels.  It does *not* contain a
prescription.  Therefore this importer deliberately creates
``syndrome_differentiation_training`` records instead of inventing a reference
formula or treating the labels as a clinical prescription recommendation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATA_FILES = ("Train_TCM_Data_v1.json", "Validation_TCM_Data_v1.json", "Test_TCM_Data_v1.json")
SOURCE_ID = "SRC_TCMEVAL_SDT"
SOURCE_URL = "https://figshare.com/articles/dataset/TCMEval-SDT/27184596"
LICENSE = "CC-BY-4.0"


def import_tcmeval_sdt_cases(*, raw_dir: Path | str, content_dir: Path | str) -> dict[str, int]:
    raw_root = Path(raw_dir)
    target = Path(content_dir)
    records: list[dict[str, Any]] = []
    source_row_count = 0
    for file_name in DATA_FILES:
        path = raw_root / file_name
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a JSON array")
        source_row_count += len(rows)
        split = file_name.removesuffix("_TCM_Data_v1.json").lower()
        # Validation/test rows intentionally omit the benchmark labels.  They
        # are useful for offline model evaluation, but are not answerable in a
        # student exercise that promises immediate source-label scoring.
        records.extend(
            item
            for row in rows
            if isinstance(row, dict)
            for item in [_case(row, split)]
            if item["training_payload"]["expected_syndromes"]
        )

    if len({item["case_id"] for item in records}) != len(records):
        raise ValueError("TCMEval-SDT source record IDs are not unique")

    learning_path = target / "learning_cases.json"
    existing = _items(learning_path)
    merged = {str(item.get("case_id")): item for item in existing if item.get("case_id")}
    previous_tcmeval_ids = {
        key
        for key, item in merged.items()
        if str(item.get("source_id") or "") == SOURCE_ID
    }
    for item in records:
        merged[item["case_id"]] = item
    for case_id in previous_tcmeval_ids - {item["case_id"] for item in records}:
        merged.pop(case_id, None)
    _write(learning_path, {"items": [merged[key] for key in sorted(merged)]})

    source = {
        "source_id": SOURCE_ID,
        "source_title": "TCMEval-SDT 中医证候辨证基准",
        "source_type": "open_source_dataset",
        "source_version": "figshare-27184596-v5",
        "source_year": "2025",
        "evidence_level": "open_dataset_teaching_benchmark",
        "license": LICENSE,
        "license_note": "CC BY 4.0. Imported only as deidentified student syndrome-differentiation training; no prescription is inferred.",
        "repository_url": SOURCE_URL,
        "processing_script": "import_tcmeval_sdt_cases.py",
        "processing_version": "1.0.0",
    }
    sources_path = target / "sources.json"
    sources = {str(item.get("source_id")): item for item in _items(sources_path) if item.get("source_id")}
    sources[SOURCE_ID] = source
    _write(sources_path, {"items": [sources[key] for key in sorted(sources)]})

    catalog_path = target / "source_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"sources": []}
    catalog_items = {str(item.get("source_id")): item for item in catalog.get("sources", []) if isinstance(item, dict) and item.get("source_id")}
    catalog_items[SOURCE_ID] = {
        "source_id": SOURCE_ID,
        "name": "TCMEval-SDT 中医证候辨证基准",
        "repository_url": SOURCE_URL,
        "license": LICENSE,
        "commit_hash": "3b666557377df69b60ab293235dd2dfa;1407860b03aa613732014da1d0abff66;d6370a3bb9b71fc2fd572c594d78dc20",
        "downloaded_at": "2026-07-19",
        "processed_at": "2026-07-19",
        "processing_script": "import_tcmeval_sdt_cases.py",
        "processing_version": "1.0.0",
        "can_process": True,
        "clinical_evidence_allowed": False,
        "patient_visible": False,
        "doctor_evidence_allowed": False,
        "student_visible": True,
        "evaluation_allowed": True,
        "notes": "CC BY 4.0 benchmark with deidentified clinical data and syndrome labels. It has no prescription field, so it is limited to student syndrome-differentiation teaching and is never used for diagnosis or prescribing.",
    }
    _write(catalog_path, {"sources": [catalog_items[key] for key in sorted(catalog_items)]})
    return {
        "imported_cases": len(records),
        "skipped_unlabelled_cases": source_row_count - len(records),
        "total_learning_cases": len(merged),
        "sources": len(sources),
    }


def _case(row: dict[str, Any], split: str) -> dict[str, Any]:
    record_id = str(row.get("Medical Record ID") or "").strip()
    if not record_id:
        raise ValueError("TCMEval-SDT record is missing Medical Record ID")
    clinical_data = _redact_clinical_data(str(row.get("Clinical Data") or ""))
    gender, age = _profile(clinical_data)
    symptoms = _terms(row.get("Clinical Information"))
    pathogenesis = _terms(row.get("TCM Pathogenesis"))
    syndromes = _terms(row.get("TCM Syndrome"))
    return {
        "case_id": f"TCMEVAL_SDT_{record_id}",
        "title": f"TCMEval-SDT 脱敏辨证案例：{record_id}",
        "case_type": "syndrome_differentiation_training",
        "visibility": "student_learning",
        "deidentified": True,
        "patient_profile": {"age": age, "age_range": _age_range(age), "gender": gender},
        "symptoms": symptoms,
        "prescription": {"items": [], "usage": None},
        "expected_risks": [],
        "standard_analysis": (
            f"来源标注的中医病机：{'；'.join(pathogenesis) or '未提供'}。\n"
            f"来源标注的中医证候：{'；'.join(syndromes) or '未提供'}。\n"
            "该案例来自公开教学基准，仅用于辨证训练；数据未提供处方组成，系统不会据此补造方剂或形成临床建议。"
        ),
        "knowledge_points": ["临床资料提取", "中医病机辨析", "中医证候辨析"],
        "source_id": SOURCE_ID,
        "source_record_id": record_id,
        "review_status": "source_labelled_teaching_benchmark",
        "source_file": f"external/tcmeval_sdt/{split}",
        "license": LICENSE,
        "training_payload": {
            "task_type": "syndrome_differentiation",
            "clinical_data": clinical_data,
            "expected_pathogenesis": pathogenesis,
            "expected_syndromes": syndromes,
            "explanatory_summary": str(row.get("Explanatory Summary") or "").strip(),
            "source_split": split,
        },
        "clinical_evidence_allowed": False,
        "patient_visible": False,
        "doctor_evidence_allowed": False,
        "student_visible": True,
        "evaluation_allowed": True,
    }


def _profile(clinical_data: str) -> tuple[str | None, int | None]:
    match = re.search(r"(男性|女性)[，,\s]*(\d{1,3})岁", clinical_data)
    if not match:
        return None, None
    return ("男" if match.group(1) == "男性" else "女"), int(match.group(2))


def _age_range(age: int | None) -> str | None:
    return f"{age // 10 * 10}-{age // 10 * 10 + 9}岁" if age is not None else None


def _terms(value: object) -> list[str]:
    return [item.strip() for item in re.split(r"[;；]", str(value or "")) if item.strip()]


def _redact_clinical_data(value: str) -> str:
    value = re.sub(r"(初诊|复诊)\s*[:：]\s*\d{4}年", r"\1年份已脱敏", value)
    return re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", "[日期已脱敏]", value)


def _items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("items", []) if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import CC BY TCMEval-SDT as student syndrome-differentiation cases.")
    parser.add_argument("--raw-dir", default="app/content/data/external/tcmeval_sdt")
    parser.add_argument("--content-dir", default="app/content/data")
    args = parser.parse_args(argv)
    print(json.dumps(import_tcmeval_sdt_cases(raw_dir=args.raw_dir, content_dir=args.content_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
