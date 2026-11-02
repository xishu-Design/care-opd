from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.content.services.content_quality_service import content_quality_report
from app.content.services.content_stats_service import content_stats
from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR
from app.data_pipeline.scripts.process_approved_sources import DEFAULT_PROCESSED_DIR
from app.data_pipeline.validators.license_validator import DEFAULT_REPORTS_DIR
from app.data_pipeline.validators.source_registry_validator import SourceRegistryValidator
from app.db.models import KnowledgeEntry, LearningCase, Quiz


def phase9_source_registry_report(db: Session) -> dict[str, Any]:
    validator = SourceRegistryValidator()
    registry = validator.validate()
    approved_ids = validator.load_approved_ids()
    rejected_ids = validator.load_rejected_ids()
    download_report = _read_json(DEFAULT_REPORTS_DIR / "download_report.json", {"items": []})
    license_report = _read_json(DEFAULT_REPORTS_DIR / "license_report.json", {"items": []})
    download_by_id = {item.get("source_id"): item for item in download_report.get("items", []) if item.get("source_id")}
    license_by_id = {item.get("source_id"): item for item in license_report.get("items", []) if item.get("source_id")}
    imported_counts = _imported_counts_by_source(db)

    sources = []
    for source in validator.load_sources():
        source_id = source.get("source_id", "")
        download_item = download_by_id.get(source_id, {})
        license_item = license_by_id.get(source_id, {})
        sources.append(
            {
                "source_id": source_id,
                "name": source.get("name", ""),
                "homepage": source.get("homepage", ""),
                "repository_url": source.get("repository_url", ""),
                "license": source.get("license", ""),
                "status": source.get("status", ""),
                "approval_status": _approval_status(str(source_id), approved_ids, rejected_ids, source.get("status", "")),
                "allowed_use": source.get("allowed_use", ""),
                "downloaded": download_item.get("download_status") == "success" or bool(download_item.get("local_path")),
                "download_status": download_item.get("download_status", ""),
                "can_process": bool(license_item.get("can_process", validator.can_download(source, approved_ids, rejected_ids))),
                "license_status": license_item.get("status") or validator.license_status(source),
                "commit_hash": download_item.get("commit_hash", ""),
                "downloaded_at": download_item.get("downloaded_at", ""),
                "imported_count": imported_counts.get(str(source_id), 0),
                "expected_content": source.get("expected_content", []),
                "notes": source.get("notes", ""),
            }
        )

    return {
        "summary": registry.to_dict(),
        "sources": sources,
        "reports": {
            "download_report": download_report,
            "license_report": license_report,
        },
    }


def phase9_data_pipeline_report(db: Session) -> dict[str, Any]:
    registry = SourceRegistryValidator().validate()
    download_report = _read_json(DEFAULT_REPORTS_DIR / "download_report.json", {"items": []})
    license_report = _read_json(DEFAULT_REPORTS_DIR / "license_report.json", {"items": []})
    processing_report = _read_json(DEFAULT_REPORTS_DIR / "processing_report.json", _read_json(DEFAULT_PROCESSED_DIR / "processing_report.json", {}))
    quality_report = _read_json(DEFAULT_REPORTS_DIR / "quality_report.json", {})
    processed_counts = _json_counts(DEFAULT_PROCESSED_DIR)
    content_counts = _json_counts(DEFAULT_CONTENT_DATA_DIR)
    return {
        "summary": {
            "approved_source_count": registry.approved_count,
            "pending_source_count": registry.pending_count,
            "rejected_source_count": registry.rejected_count,
            "processed_item_count": sum(
                processed_counts.get(name, 0)
                for name in (
                    "herb_candidates.json",
                    "formula_candidates.json",
                    "prescription_candidates.json",
                    "learning_case_candidates.json",
                )
            ),
            "pending_item_count": processed_counts.get("pending_items.json", 0),
            "rejected_item_count": processed_counts.get("rejected_items.json", 0),
            "content_item_count": sum(
                content_counts.get(name, 0)
                for name in (
                    "sources.json",
                    "herbs.json",
                    "formulas.json",
                    "learning_cases.json",
                    "quizzes.json",
                )
            ),
        },
        "processed_counts": processed_counts,
        "content_counts": content_counts,
        "content_stats": content_stats(db),
        "content_quality": content_quality_report(db),
        "reports": {
            "download_report": download_report,
            "license_report": license_report,
            "processing_report": processing_report,
            "quality_report": quality_report,
        },
    }


def _imported_counts_by_source(db: Session) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in db.scalars(select(KnowledgeEntry)).all():
        source_ids = []
        metadata = entry.metadata_json if isinstance(entry.metadata_json, dict) else {}
        raw_ids = metadata.get("source_ids") or [entry.source_id]
        if isinstance(raw_ids, list):
            source_ids.extend(str(source_id) for source_id in raw_ids if source_id)
        elif raw_ids:
            source_ids.append(str(raw_ids))
        for source_id in source_ids:
            counts[source_id] = counts.get(source_id, 0) + 1
    for case in db.scalars(select(LearningCase)).all():
        payload = case.anonymized_case if isinstance(case.anonymized_case, dict) else {}
        source_id = payload.get("source_id")
        if source_id:
            counts[str(source_id)] = counts.get(str(source_id), 0) + 1
    for quiz in db.scalars(select(Quiz)).all():
        if quiz.source_id:
            counts[quiz.source_id] = counts.get(quiz.source_id, 0) + 1
    return counts


def _approval_status(source_id: str, approved_ids: set[str], rejected_ids: set[str], status: str) -> str:
    if source_id in rejected_ids or status == "rejected":
        return "rejected"
    if source_id in approved_ids:
        return "approved"
    return "pending"


def _json_counts(directory: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not directory.exists():
        return counts
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path, {})
        if not isinstance(payload, dict):
            counts[path.name] = 0
        elif isinstance(payload.get("items"), list):
            counts[path.name] = len(payload["items"])
        elif isinstance(payload.get("sources"), list):
            counts[path.name] = len(payload["sources"])
        else:
            counts[path.name] = 0
    return counts


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else default
