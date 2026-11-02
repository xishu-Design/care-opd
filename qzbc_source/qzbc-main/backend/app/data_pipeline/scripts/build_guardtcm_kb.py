from __future__ import annotations

import argparse
import json
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.data_pipeline.normalizers.tcm_entity_normalizer import normalize_entity_name
from app.data_pipeline.scripts.process_approved_sources import DEFAULT_PROCESSED_DIR
from app.data_pipeline.validators.duplicate_validator import merge_by_key
from app.data_pipeline.validators.license_validator import DEFAULT_REPORTS_DIR
from app.data_pipeline.validators.source_registry_validator import DATA_PIPELINE_DIR


DEFAULT_CONTENT_DATA_DIR = DATA_PIPELINE_DIR.parents[0] / "content" / "data"


def build_guardtcm_kb(
    *,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    content_data_dir: Path | str = DEFAULT_CONTENT_DATA_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> dict[str, Any]:
    started_at = _now()
    processed = Path(processed_dir)
    content_dir = Path(content_data_dir)
    content_dir.mkdir(parents=True, exist_ok=True)
    source_catalog = _load_json(processed / "source_catalog.json", {"sources": []})
    herb_candidates = _items(processed / "herb_candidates.json")
    formula_candidates = _items(processed / "formula_candidates.json")
    learning_case_candidates = _items(processed / "learning_case_candidates.json")

    herbs = _build_herbs(herb_candidates)
    formulas = _build_formulas(formula_candidates)
    learning_cases = _build_learning_cases(learning_case_candidates)
    quizzes = _build_quizzes(learning_cases, herbs)
    sources = _build_sources(source_catalog)

    sources = _merge_existing(_items(content_dir / "sources.json"), sources, "source_id")
    herbs = _merge_existing(_items(content_dir / "herbs.json"), herbs, "entry_id")
    formulas = _merge_existing(_items(content_dir / "formulas.json"), formulas, "entry_id")
    learning_cases = _merge_existing(_items(content_dir / "learning_cases.json"), learning_cases, "case_id")
    quizzes = _merge_existing(_items(content_dir / "quizzes.json"), quizzes, "quiz_id")
    relationships = _build_relationships(herbs, formulas, learning_cases)

    payloads = {
        "sources.json": {"items": sources},
        "herbs.json": {"items": herbs},
        "formulas.json": {"items": formulas},
        "learning_cases.json": {"items": learning_cases},
        "quizzes.json": {"items": quizzes},
        "source_catalog.json": source_catalog,
        "relationships.json": {"items": relationships},
        "contraindications.json": {"items": _items(content_dir / "contraindications.json")},
        "diet_precautions.json": {"items": _items(content_dir / "diet_precautions.json")},
        "public_education.json": {"items": _items(content_dir / "public_education.json")},
        "review_rules.json": {"items": _items(content_dir / "review_rules.json")},
    }
    for file_name, payload in payloads.items():
        _write_json(content_dir / file_name, payload)

    report = {
        "run_id": f"build_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "started_at": started_at,
        "ended_at": _now(),
        "input_counts": {
            "herb_candidates": len(herb_candidates),
            "formula_candidates": len(formula_candidates),
            "learning_case_candidates": len(learning_case_candidates),
        },
        "output_counts": {
            "sources": len(sources),
            "herbs": len(herbs),
            "formulas": len(formulas),
            "learning_cases": len(learning_cases),
            "quizzes": len(quizzes),
            "relationships": len(relationships),
        },
        "notes": "Generated from processed open-source candidates. No functions, cautions, dose ranges, diagnoses, or review rules were inferred.",
    }
    _write_json(content_dir / "build_report.json", report)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    _write_json(reports_path / "quality_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GuardTCM content/data from processed candidate files.")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--content-data-dir", default=str(DEFAULT_CONTENT_DATA_DIR))
    args = parser.parse_args(argv)
    report = build_guardtcm_kb(processed_dir=args.processed_dir, content_data_dir=args.content_data_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _build_sources(source_catalog: dict[str, Any]) -> list[dict[str, Any]]:
    sources = []
    for source in source_catalog.get("sources", []):
        sources.append(
            {
                "source_id": source.get("source_id", ""),
                "source_title": source.get("name", ""),
                "source_type": "open_source_dataset",
                "source_version": source.get("commit_hash", "") or source.get("downloaded_at", ""),
                "source_year": "",
                "evidence_level": "open_dataset_extracted",
                "license": source.get("license", ""),
                "license_note": "Derived from approved open-source dataset.",
                "repository_url": source.get("repository_url", ""),
                "processing_script": source.get("processing_script", ""),
                "processing_version": source.get("processing_version", ""),
            }
        )
    return sorted(sources, key=lambda item: item["source_id"])


def _build_herbs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for candidate in candidates:
        normalized = candidate.get("normalized_name") or normalize_entity_name(candidate.get("name", ""))
        if not normalized:
            continue
        prepared.append(
            {
                "entry_id": "",
                "candidate_entry_id": candidate.get("candidate_id", ""),
                "name": candidate.get("name", normalized),
                "normalized_name": normalized,
                "aliases": [],
                "source_ids": [candidate.get("source_id", "")],
                "source_records": [_source_record(candidate)],
                "functions": [],
                "cautions": [],
                "dose_range": None,
                "keywords": [normalized],
                "visibility": ["doctor", "student"],
                "evidence_level": "open_dataset_extracted",
                "license_note": "Derived from approved open-source dataset. Function/caution/dose not inferred.",
            }
        )
    merged = merge_by_key(prepared, "normalized_name")
    for item in merged:
        item["entry_id"] = item.pop("candidate_entry_id", "") or _content_id("HERB", item.get("normalized_name", ""))
    return merged


def _build_formulas(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formulas = []
    for index, candidate in enumerate(candidates, start=1):
        components = candidate.get("components", [])
        source_id = candidate.get("source_id", "")
        name = candidate.get("name") or f"source_formula_{index:06d}"
        keywords = [component.get("normalized_name") or component.get("herb_name") for component in components]
        formulas.append(
            {
                "entry_id": candidate.get("candidate_id") or _content_id("FORMULA", source_id, candidate.get("source_file", ""), candidate.get("source_record_id", "")),
                "name": name,
                "normalized_name": candidate.get("normalized_name") or normalize_entity_name(name),
                "components": components,
                "source_ids": [source_id],
                "source_records": [_source_record(candidate)],
                "functions": [],
                "cautions": [],
                "usage_note": "Open-source prescription composition candidate. Not a clinical recommendation.",
                "keywords": [keyword for keyword in [*keywords, "处方组成"] if keyword],
                "visibility": ["doctor", "student"],
                "evidence_level": "open_dataset_extracted",
            }
        )
    return formulas


def _build_learning_cases(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for index, candidate in enumerate(candidates, start=1):
        if candidate.get("deidentified") is not True or candidate.get("visibility") != "student_learning":
            continue
        source_id = candidate.get("source_id", "")
        cases.append(
            {
                "case_id": candidate.get("case_candidate_id") or _content_id("CASE", source_id, candidate.get("source_file", ""), candidate.get("source_record_id", "")),
                "title": candidate.get("title") or f"开源处方组成识别训练案例 {index:06d}",
                "case_type": candidate.get("case_type") or "prescription_composition_learning",
                "visibility": "student_learning",
                "deidentified": True,
                "patient_profile": candidate.get("patient_profile", {"age_range": None, "gender": None}),
                "symptoms": candidate.get("symptoms") or candidate.get("symptom_summary") or candidate.get("chief_complaint") or [],
                "prescription": candidate.get("prescription", {"items": [], "usage": None}),
                "expected_risks": [],
                "standard_analysis": _training_answer(candidate) or "该案例仅用于训练处方组成识别，不包含审方风险结论。",
                "knowledge_points": candidate.get("knowledge_points", ["处方组成识别"]),
                "source_id": source_id,
                "source_record_id": candidate.get("source_record_id", ""),
                "review_status": "auto_generated_learning_candidate",
                "source_file": candidate.get("source_file", ""),
                "commit_sha": candidate.get("commit_sha", ""),
                "license": candidate.get("license", ""),
                "training_payload": candidate.get("training_payload", {}),
                "clinical_evidence_allowed": candidate.get("clinical_evidence_allowed", source_id != "SRC_MTCMB"),
                "patient_visible": candidate.get("patient_visible", False),
                "doctor_evidence_allowed": candidate.get("doctor_evidence_allowed", source_id != "SRC_MTCMB"),
                "student_visible": True,
                "evaluation_allowed": candidate.get("evaluation_allowed", False),
            }
        )
    return cases


def _build_quizzes(learning_cases: list[dict[str, Any]], herbs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    herb_pool = [herb["name"] for herb in herbs if herb.get("name")]
    quizzes = []
    for index, case in enumerate(learning_cases, start=1):
        items = case.get("prescription", {}).get("items", [])
        answer = next((item.get("herb_name") for item in items if item.get("herb_name")), "")
        if not answer:
            continue
        if answer not in herb_pool:
            continue
        options = _quiz_options(answer, herb_pool)
        if len(options) < 2:
            continue
        quizzes.append(
            {
                "quiz_id": f"QUIZ_{_source_prefix(case.get('source_id', ''))}_{index:06d}",
                "case_id": case["case_id"],
                "question_type": "single_choice",
                "question": "该处方候选中包含以下哪味药材？",
                "options": options,
                "answer": answer,
                "explanation": "该题仅考察处方组成识别，答案来自开源数据源中的处方组成字段。",
                "knowledge_points": ["处方组成识别"],
                "source_id": case.get("source_id", ""),
            }
        )
    return quizzes


def _quiz_options(answer: str, herb_pool: list[str]) -> list[str]:
    options = [answer]
    for herb in sorted(herb_pool):
        if herb == answer or herb in options:
            continue
        options.append(herb)
        if len(options) == 4:
            break
    return options


def _source_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": candidate.get("source_id", ""),
        "source_record_id": candidate.get("source_record_id", ""),
        "source_file": candidate.get("source_file", ""),
        "commit_sha": candidate.get("commit_sha", ""),
        "license": candidate.get("license", ""),
    }


def _source_prefix(source_id: str) -> str:
    prefix = source_id.removeprefix("SRC_")
    return "".join(ch for ch in prefix if ch.isalnum() or ch == "_") or "OPEN_DATA"


def _items(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path, {"items": []})
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain an items list")
    return items


def _merge_existing(existing: list[dict[str, Any]], generated: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    merged = {str(item.get(key)): item for item in generated if item.get(key)}
    generated_sources = {
        str(source_id)
        for item in generated
        for source_id in (item.get("source_ids") or [item.get("source_id", "")])
        if source_id
    }
    for item in existing:
        item_sources = {str(value) for value in (item.get("source_ids") or [item.get("source_id", "")]) if value}
        generated_item = (
            item.get("evidence_level") == "open_dataset_extracted"
            or str(item.get("review_status", "")).startswith("auto_generated")
            or (key == "quiz_id" and bool(item_sources & generated_sources))
        )
        if generated_item and item_sources & generated_sources:
            continue
        if item.get(key):
            merged[str(item[key])] = item
    return [merged[item_key] for item_key in sorted(merged)]


def _content_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20].upper()
    return f"{prefix}_{digest}"


def _training_answer(candidate: dict[str, Any]) -> str:
    payload = candidate.get("training_payload")
    if not isinstance(payload, dict):
        return ""
    answer = payload.get("answer")
    return json.dumps(answer, ensure_ascii=False) if isinstance(answer, (dict, list)) else str(answer or "")


def _build_relationships(herbs: list[dict[str, Any]], formulas: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    herb_by_name = {item.get("normalized_name") or normalize_entity_name(item.get("name", "")): item for item in herbs}
    relations: dict[str, dict[str, Any]] = {}
    for formula in formulas:
        for component in formula.get("components", []):
            normalized = component.get("normalized_name") or normalize_entity_name(component.get("herb_name", ""))
            herb = herb_by_name.get(normalized)
            if herb:
                _add_relation(relations, formula["entry_id"], herb["entry_id"], "CONTAINS", formula)
        for source_id in formula.get("source_ids", []):
            _add_relation(relations, formula["entry_id"], source_id, "DERIVED_FROM", formula)
    for herb in herbs:
        for source_id in herb.get("source_ids", []):
            _add_relation(relations, herb["entry_id"], source_id, "DERIVED_FROM", herb)
    for case in cases:
        for item in case.get("prescription", {}).get("items", []):
            herb = herb_by_name.get(normalize_entity_name(item.get("herb_name", "")))
            if herb:
                _add_relation(relations, case["case_id"], herb["entry_id"], "REFERENCES", case)
        source_id = case.get("source_id", "")
        if source_id:
            _add_relation(relations, case["case_id"], source_id, "DERIVED_FROM", case)
    return [relations[key] for key in sorted(relations)]


def _add_relation(target: dict[str, dict[str, Any]], source_entity_id: str, target_entity_id: str, relation_type: str, item: dict[str, Any]) -> None:
    relation_id = _content_id("REL", source_entity_id, target_entity_id, relation_type)
    source_ids = item.get("source_ids") or [item.get("source_id", "")]
    source_record_ids = [record.get("source_record_id", "") for record in item.get("source_records", [])] or [item.get("source_record_id", "")]
    target[relation_id] = {
        "relation_id": relation_id,
        "source_entity_id": source_entity_id,
        "target_entity_id": target_entity_id,
        "relation_type": relation_type,
        "source_ids": [value for value in source_ids if value],
        "source_record_ids": [value for value in source_record_ids if value],
        "evidence_level": "dataset_explicit",
        "visibility": ["student"] if "SRC_MTCMB" in source_ids else ["doctor", "student"],
    }


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
