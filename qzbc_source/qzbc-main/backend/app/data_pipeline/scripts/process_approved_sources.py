from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.data_pipeline.extractors.tcmdataforml_extractor import TCMDataForMLExtractor
from app.data_pipeline.extractors.mtcmb_extractor import MTCMBExtractor
from app.data_pipeline.validators.license_validator import DEFAULT_RAW_DIR, DEFAULT_REPORTS_DIR
from app.data_pipeline.validators.processed_item_validator import ProcessedItemValidator
from app.data_pipeline.validators.source_registry_validator import DATA_PIPELINE_DIR, SourceRegistryValidator


DEFAULT_PROCESSED_DIR = DATA_PIPELINE_DIR / "processed"


def process_approved_sources(
    *,
    license_report_path: Path | str = DEFAULT_REPORTS_DIR / "license_report.json",
    download_report_path: Path | str = DEFAULT_REPORTS_DIR / "download_report.json",
    registry_dir: Path | str | None = None,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    source_id: str | None = None,
) -> dict[str, Any]:
    started_at = _now()
    registry = SourceRegistryValidator(registry_dir) if registry_dir else SourceRegistryValidator()
    sources_by_id = {source["source_id"]: source for source in registry.load_sources()}
    license_report = _load_json(Path(license_report_path))
    download_report = _load_json(Path(download_report_path))
    downloads_by_id = {item.get("source_id"): item for item in download_report.get("items", [])}

    selected_license_items = [item for item in license_report.get("items", []) if item.get("can_process") is True]
    if source_id:
        selected_license_items = [item for item in selected_license_items if item.get("source_id") == source_id]

    aggregate = _empty_aggregate()
    source_catalog = {"sources": []}
    report_items: list[dict[str, Any]] = []

    for license_item in selected_license_items:
        item_source_id = str(license_item.get("source_id", ""))
        source_meta = sources_by_id.get(item_source_id)
        download_item = downloads_by_id.get(item_source_id, {})
        if not source_meta:
            rejected = _rejected(item_source_id, "", "source_not_registered", "License report source_id not found in registry.")
            aggregate["rejected_items"].append(rejected)
            report_items.append(_report_item(item_source_id, "failed", "", {}, "source_not_registered"))
            continue
        try:
            extractor = _extractor_for_source(item_source_id, raw_dir, source_meta, license_item, download_item)
            extracted = extractor.extract()
            _merge(aggregate, extracted)
            source_catalog["sources"].append(_catalog_item(source_meta, license_item, download_item, extractor))
            report_items.append(_report_item(item_source_id, "success", extractor.__class__.__name__, extracted.get("stats", {}), None))
        except Exception as exc:  # pragma: no cover - defensive report path
            aggregate["rejected_items"].append(_rejected(item_source_id, "", "extractor_failed", str(exc)))
            report_items.append(_report_item(item_source_id, "failed", "", {}, str(exc)))

    validator = ProcessedItemValidator()
    aggregate = _partition_invalid_items(aggregate, validator)
    approved_case_ids = {item.get("case_candidate_id") for item in aggregate["learning_case_candidates"]}
    aggregate["approved_training_items"] = [
        item for item in aggregate["approved_training_items"] if item.get("case_candidate_id") in approved_case_ids
    ]
    _assign_item_ids(aggregate)
    report = _processing_report(started_at, report_items, aggregate)
    _write_processed_files(Path(processed_dir), Path(reports_dir), aggregate, source_catalog, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process can_process=true Phase 9 sources into intermediate candidate JSON.")
    parser.add_argument("--source", help="Process one source_id from license_report.json.")
    args = parser.parse_args(argv)
    report = process_approved_sources(source_id=args.source)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _extractor_for_source(source_id: str, raw_dir: Path | str, source_meta: dict[str, Any], license_item: dict[str, Any], download_item: dict[str, Any]):
    if source_id == "SRC_TCM_DATA_FOR_ML":
        return TCMDataForMLExtractor(raw_dir, source_meta, license_item, download_item)
    if source_id == "SRC_MTCMB":
        return MTCMBExtractor(raw_dir, source_meta, license_item, download_item)
    raise ValueError(f"No extractor registered for {source_id}")


def _empty_aggregate() -> dict[str, list[dict[str, Any]]]:
    return {
        "herb_candidates": [],
        "formula_candidates": [],
        "prescription_candidates": [],
        "learning_case_candidates": [],
        "approved_training_items": [],
        "pending_items": [],
        "rejected_items": [],
    }


def _merge(aggregate: dict[str, list[dict[str, Any]]], extracted: dict[str, Any]) -> None:
    for key in aggregate:
        aggregate[key].extend(extracted.get(key, []))


def _partition_invalid_items(aggregate: dict[str, list[dict[str, Any]]], validator: ProcessedItemValidator) -> dict[str, list[dict[str, Any]]]:
    candidate_sections = {
        "herb_candidates": "candidate_id",
        "formula_candidates": "candidate_id",
        "prescription_candidates": "candidate_id",
        "learning_case_candidates": "case_candidate_id",
    }
    for section, id_field in candidate_sections.items():
        valid_items: list[dict[str, Any]] = []
        for item in aggregate[section]:
            issues = validator.validate_candidate(item, candidate_id_field=id_field)
            if issues:
                aggregate["rejected_items"].append(
                    _rejected(
                        str(item.get("source_id", "")),
                        str(item.get("source_record_id", "")),
                        ",".join(issue.code for issue in issues),
                        str({issue.field: issue.message for issue in issues}),
                    )
                )
            else:
                valid_items.append(item)
        aggregate[section] = valid_items
    return aggregate


def _assign_item_ids(aggregate: dict[str, list[dict[str, Any]]]) -> None:
    for index, item in enumerate(aggregate["pending_items"], start=1):
        item["item_id"] = item.get("item_id") or f"PENDING_{index:06d}"
    for index, item in enumerate(aggregate["rejected_items"], start=1):
        item["item_id"] = item.get("item_id") or f"REJECTED_{index:06d}"
        item["rejected_at"] = item.get("rejected_at") or _now()


def _catalog_item(source_meta: dict[str, Any], license_item: dict[str, Any], download_item: dict[str, Any], extractor: Any) -> dict[str, Any]:
    return {
        "source_id": source_meta.get("source_id", ""),
        "name": source_meta.get("name", ""),
        "repository_url": source_meta.get("repository_url", ""),
        "license": license_item.get("detected_license") or license_item.get("declared_license") or source_meta.get("license", ""),
        "commit_hash": download_item.get("commit_hash", ""),
        "downloaded_at": download_item.get("downloaded_at", ""),
        "processed_at": _now(),
        "processing_script": getattr(__import__(extractor.__class__.__module__, fromlist=["PROCESSING_SCRIPT"]), "PROCESSING_SCRIPT", extractor.__class__.__name__),
        "processing_version": getattr(__import__(extractor.__class__.__module__, fromlist=["PROCESSING_VERSION"]), "PROCESSING_VERSION", "1.0.0"),
        "can_process": license_item.get("can_process") is True,
        "clinical_evidence_allowed": source_meta.get("clinical_evidence_allowed", True),
        "patient_visible": source_meta.get("patient_visible", False),
        "doctor_evidence_allowed": source_meta.get("doctor_evidence_allowed", True),
        "student_visible": source_meta.get("student_visible", True),
        "evaluation_allowed": source_meta.get("evaluation_allowed", False),
        "notes": source_meta.get("notes", "Parsed from an approved open-source dataset."),
    }


def _processing_report(started_at: str, report_items: list[dict[str, Any]], aggregate: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    stats = {key: len(value) for key, value in aggregate.items()}
    return {
        "run_id": f"processing_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "started_at": started_at,
        "ended_at": _now(),
        "total_sources": len(report_items),
        "processed_sources": sum(1 for item in report_items if item["status"] == "success"),
        "failed_sources": sum(1 for item in report_items if item["status"] == "failed"),
        "stats": stats,
        "items": report_items,
    }


def _report_item(source_id: str, status: str, extractor: str, stats: dict[str, Any], error: str | None) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "status": status,
        "extractor": extractor,
        "herb_candidates": int(stats.get("herb_candidates", 0)),
        "formula_candidates": int(stats.get("formula_candidates", 0)),
        "prescription_candidates": int(stats.get("prescription_candidates", 0)),
        "learning_case_candidates": int(stats.get("learning_case_candidates", 0)),
        "pending_items": int(stats.get("pending_items", 0)),
        "rejected_items": int(stats.get("rejected_items", 0)),
        "error": error,
    }


def _write_processed_files(processed_dir: Path, reports_dir: Path, aggregate: dict[str, list[dict[str, Any]]], source_catalog: dict[str, Any], report: dict[str, Any]) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, Any] = {
        "herb_candidates.json": {"items": aggregate["herb_candidates"]},
        "formula_candidates.json": {"items": aggregate["formula_candidates"]},
        "prescription_candidates.json": {"items": aggregate["prescription_candidates"]},
        "learning_case_candidates.json": {"items": aggregate["learning_case_candidates"]},
        "approved_training_items.json": {"items": aggregate["approved_training_items"]},
        "pending_items.json": {"items": aggregate["pending_items"]},
        "rejected_items.json": {"items": aggregate["rejected_items"]},
        "source_catalog.json": source_catalog,
        "processing_report.json": report,
    }
    for file_name, payload in payloads.items():
        (processed_dir / file_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "processing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rejected(source_id: str, record_id: str, reason: str, raw_summary: str) -> dict[str, Any]:
    return {"item_id": "", "source_id": source_id, "source_record_id": record_id, "reason": reason, "raw_summary": raw_summary, "rejected_at": ""}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
