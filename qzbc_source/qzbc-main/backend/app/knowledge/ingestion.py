from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeClaim, KnowledgeEntry, KnowledgeIngestionRun, KnowledgeSource, LightRAGRetrievalLog, utcnow
from app.knowledge.entity_quality import is_valid_tcm_herb_label
from app.services.herb_incompatibility_service import incompatibilities_for_herb

DATA_DIR = Path(__file__).resolve().parent / "data"

ENTRY_FILES = {
    "herb": "herbs.json",
    "formula": "formulas.json",
    "contraindication": "contraindications.json",
    "diet_precaution": "diet_precautions.json",
    "public_education": "public_education.json",
}


def ingest_local_knowledge(db: Session) -> dict[str, Any]:
    run = KnowledgeIngestionRun(status="running")
    db.add(run)
    db.flush()
    try:
        sources = _load_json("sources.json")
        source_map = _upsert_sources(db, sources)
        entries_count = 0
        file_report: dict[str, int] = {}
        for entry_type, file_name in ENTRY_FILES.items():
            raw_items = _load_json(file_name)
            for raw in raw_items:
                _upsert_entry(db, _normalize_entry(entry_type, raw, source_map))
                entries_count += 1
            file_report[file_name] = len(raw_items)

        run.status = "completed"
        run.sources_count = len(sources)
        run.entries_count = entries_count
        run.report_json = {"files": file_report}
        run.finished_at = utcnow()
        db.flush()
        return {
            "status": run.status,
            "sources_count": run.sources_count,
            "entries_count": run.entries_count,
            "report": run.report_json,
        }
    except Exception as exc:
        run.status = "failed"
        run.report_json = {"error": str(exc)}
        run.finished_at = utcnow()
        db.flush()
        raise


def ensure_knowledge_loaded(db: Session) -> None:
    entries_count = db.scalar(select(func.count(KnowledgeEntry.id))) or 0
    if entries_count == 0:
        ingest_local_knowledge(db)


def knowledge_health(db: Session) -> dict[str, Any]:
    entries_count = db.scalar(select(func.count(KnowledgeEntry.id))) or 0
    sources_count = db.scalar(select(func.count(KnowledgeSource.id))) or 0
    last_run = db.scalar(select(KnowledgeIngestionRun).order_by(KnowledgeIngestionRun.started_at.desc()))
    return {
        "knowledge_backend": "local_knowledge",
        "status": "ok",
        "entries_count": entries_count,
        "sources_count": sources_count,
        "last_ingestion_at": last_run.finished_at.isoformat() if last_run and last_run.finished_at else "",
    }


def list_sources(db: Session) -> list[dict[str, Any]]:
    ensure_knowledge_loaded(db)
    rows = db.scalars(select(KnowledgeSource).order_by(KnowledgeSource.source_id)).all()
    return [
        {
            "source_id": row.source_id,
            "source_title": row.source_title,
            "source_type": row.source_type,
            "source_version": row.source_version,
            "source_year": row.source_year,
            "evidence_level": row.evidence_level,
            "license_note": row.license_note,
        }
        for row in rows
    ]


def list_knowledge_entries(
    db: Session,
    *,
    role: str,
    entry_type: str | None = None,
    query: str | None = None,
    source_id: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Return a role-filtered, provenance-first view of local knowledge entries."""
    ensure_knowledge_loaded(db)
    rows_query = select(KnowledgeEntry).order_by(KnowledgeEntry.entry_type, KnowledgeEntry.title, KnowledgeEntry.id)
    if entry_type:
        rows_query = rows_query.where(KnowledgeEntry.entry_type == entry_type)
    if source_id:
        rows_query = rows_query.where(KnowledgeEntry.source_id == source_id)

    lowered_query = (query or "").strip().lower()
    visible_rows = []
    for row in db.scalars(rows_query).all():
        if not _visible_to_role(row, role):
            continue
        if entry_type == "formula" and row.source_id == "SRC_DRAGON_TCM":
            continue
        if entry_type == "herb" and not _is_catalog_ready_herb(row):
            continue
        if lowered_query and not _matches_catalog_query(row, lowered_query):
            continue
        visible_rows.append(row)

    total = len(visible_rows)
    start = (page - 1) * page_size
    return {
        "items": [_to_catalog_item(row, db) for row in visible_rows[start : start + page_size]],
        "page": page,
        "page_size": page_size,
        "total": total,
    }


def get_knowledge_entry(db: Session, *, role: str, source_id: str, entry_id: str) -> dict[str, Any] | None:
    """Return a single role-filtered entry without exposing unsafe raw metadata."""
    ensure_knowledge_loaded(db)
    row = db.scalar(
        select(KnowledgeEntry).where(
            KnowledgeEntry.source_id == source_id,
            KnowledgeEntry.entry_id == entry_id,
        )
    )
    if row is None or not _visible_to_role(row, role):
        return None
    item = _to_catalog_item(row, db)
    item.update(
        {
            "content": row.content,
            "metadata": _safe_catalog_metadata(row, db),
        }
    )
    return item


def search_knowledge(
    db: Session,
    *,
    query: str,
    entry_type: str | None = None,
    limit: int = 5,
    retrieved_by: str = "search_evidence",
    log_request: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ensure_knowledge_loaded(db)
    rows_query = select(KnowledgeEntry)
    if entry_type:
        rows_query = rows_query.where(KnowledgeEntry.entry_type == entry_type)
    rows = db.scalars(rows_query).all()
    role = (log_request or {}).get("role")
    scored = []
    for row in rows:
        if not _visible_to_role(row, role):
            continue
        score = _score(row, query)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    evidence = [_to_evidence(row, min(score, 1.0), retrieved_by) for score, row in scored[:limit]]
    if log_request is not None:
        db.add(
            LightRAGRetrievalLog(
                session_id=log_request.get("session_id"),
                query=query,
                role=log_request.get("role", ""),
                task_type=log_request.get("task_type", ""),
                evidence_json=evidence,
            )
        )
        db.flush()
    return evidence


def _load_json(file_name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / file_name
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{file_name} must contain a JSON list")
    return data


def _upsert_sources(db: Session, raw_sources: list[dict[str, Any]]) -> dict[str, KnowledgeSource]:
    source_map: dict[str, KnowledgeSource] = {}
    for raw in raw_sources:
        _require(raw, "source_id", "source_title", "source_type")
        source = db.scalar(select(KnowledgeSource).where(KnowledgeSource.source_id == raw["source_id"]))
        if not source:
            source = KnowledgeSource(source_id=raw["source_id"], source_title=raw["source_title"], source_type=raw["source_type"])
        source.source_title = raw["source_title"]
        source.source_type = raw["source_type"]
        source.source_version = raw.get("source_version", "")
        source.source_year = str(raw.get("source_year", ""))
        source.evidence_level = raw.get("evidence_level", "demo")
        source.license_note = raw.get("license_note", "")
        db.add(source)
        source_map[source.source_id] = source
    return source_map


def _upsert_entry(db: Session, normalized: dict[str, Any]) -> None:
    entry = db.scalar(
        select(KnowledgeEntry).where(
            KnowledgeEntry.source_id == normalized["source_id"],
            KnowledgeEntry.entry_id == normalized["entry_id"],
        )
    )
    if not entry:
        entry = KnowledgeEntry(entry_id=normalized["entry_id"], source_id=normalized["source_id"], entry_type=normalized["entry_type"], title=normalized["title"])
    entry.entry_type = normalized["entry_type"]
    entry.title = normalized["title"]
    entry.content = normalized["content"]
    entry.keywords_json = normalized["keywords"]
    entry.source_title = normalized["source_title"]
    entry.source_type = normalized["source_type"]
    entry.metadata_json = normalized["metadata"]
    entry.updated_at = utcnow()
    db.add(entry)


def _normalize_entry(entry_type: str, raw: dict[str, Any], source_map: dict[str, KnowledgeSource]) -> dict[str, Any]:
    _require(raw, "id", "source_id")
    source = source_map.get(raw["source_id"])
    if not source:
        raise ValueError(f"Unknown source_id {raw['source_id']} for {raw['id']}")
    title = _title_for(entry_type, raw)
    return {
        "entry_id": raw["id"],
        "entry_type": entry_type,
        "title": title,
        "content": _content_for(entry_type, raw, title),
        "keywords": _keywords_for(entry_type, raw, title),
        "source_id": raw["source_id"],
        "source_title": source.source_title,
        "source_type": source.source_type,
        "metadata": raw,
    }


def _title_for(entry_type: str, raw: dict[str, Any]) -> str:
    if entry_type in {"herb", "formula"}:
        _require(raw, "name")
        return raw["name"]
    if entry_type == "contraindication":
        _require(raw, "title")
        return raw["title"]
    if entry_type == "diet_precaution":
        return raw.get("related_entity", "饮食注意事项")
    if entry_type == "public_education":
        return raw.get("topic", "中医药科普")
    return raw.get("title", raw["id"])


def _content_for(entry_type: str, raw: dict[str, Any], title: str) -> str:
    if entry_type == "herb":
        return f"{title}，{raw.get('category', '')}，功能：{'、'.join(raw.get('functions', []))}。慎用：{'、'.join(raw.get('cautions', []))}。参考剂量：{raw.get('dose_range', '')}。"
    if entry_type == "formula":
        return f"{title}，组成：{'、'.join(raw.get('components', []))}。功能：{'、'.join(raw.get('functions', []))}。注意：{'、'.join(raw.get('cautions', []))}。"
    if entry_type == "contraindication":
        return f"{title}：{raw.get('message', '')}"
    if entry_type == "diet_precaution":
        return raw.get("message", "")
    if entry_type == "public_education":
        return raw.get("safe_answer", "")
    return str(raw)


def _keywords_for(entry_type: str, raw: dict[str, Any], title: str) -> list[str]:
    keywords = [title, *raw.get("keywords", [])]
    if entry_type in {"herb", "formula"}:
        keywords.extend(raw.get("aliases", []))
        keywords.extend(raw.get("components", []))
        keywords.extend(raw.get("functions", []))
    if entry_type == "public_education":
        keywords.extend(raw.get("question_patterns", []))
    if entry_type == "contraindication":
        keywords.append(raw.get("entity_name", ""))
    return sorted({keyword for keyword in keywords if keyword})


def _score(row: KnowledgeEntry, query: str) -> float:
    query = (query or "").strip()
    if not query:
        return 0.0
    lowered = query.lower()
    title = row.title.lower()
    content = row.content.lower()
    keywords = [str(keyword).lower() for keyword in row.keywords_json]
    metadata_text = json.dumps(row.metadata_json, ensure_ascii=False).lower()

    score = 0.0
    if title and title in lowered:
        score += 0.55
    if lowered == title:
        score += 0.25
    for keyword in keywords:
        if keyword and keyword in lowered:
            score += 0.18
    for token in _query_tokens(lowered):
        if token in title:
            score += 0.18
        if token in content:
            score += 0.1
        if token in metadata_text:
            score += 0.05
    return score


def _query_tokens(query: str) -> list[str]:
    separators = (" ", "，", ",", "。", "；", ";", "、")
    normalized = query
    for separator in separators:
        normalized = normalized.replace(separator, " ")
    tokens = [token.strip() for token in normalized.split(" ") if token.strip()]
    if not tokens and query:
        tokens = [query]
    return tokens


def _to_evidence(row: KnowledgeEntry, confidence: float, retrieved_by: str) -> dict[str, Any]:
    raw_metadata = row.metadata_json or {}
    source_catalog = raw_metadata.get("source_catalog") if isinstance(raw_metadata.get("source_catalog"), dict) else {}
    source_ids = raw_metadata.get("source_ids") or [row.source_id]
    source_records = raw_metadata.get("source_records") or [{"source_id": row.source_id, "source_record_id": row.entry_id}]
    original_source = raw_metadata.get("original_source") or source_catalog.get("name") or source_catalog.get("source_title") or row.source_title
    license_value = raw_metadata.get("license") or source_catalog.get("license") or ""
    metadata = {
        "entry_type": row.entry_type,
        "source_title": row.source_title,
        "source_ids": source_ids,
        "source_records": source_records,
        "original_source": original_source,
        "license": license_value,
        "license_note": raw_metadata.get("license_note") or source_catalog.get("license_note") or "",
        "repository_url": raw_metadata.get("repository_url") or source_catalog.get("repository_url") or "",
        "commit_hash": raw_metadata.get("commit_hash") or source_catalog.get("commit_hash") or "",
        "processing_script": raw_metadata.get("processing_script") or source_catalog.get("processing_script") or "",
        "processing_version": raw_metadata.get("processing_version") or source_catalog.get("processing_version") or "",
        "evidence_level": raw_metadata.get("evidence_level") or source_catalog.get("evidence_level") or "",
        "visibility": raw_metadata.get("visibility") or [],
        "demo_only": _is_demo_evidence(row, raw_metadata, source_catalog),
    }
    if row.entry_type == "herb":
        metadata.update(
            {
                "name": raw_metadata.get("name", row.title),
                "normalized_name": raw_metadata.get("normalized_name", row.title),
                "aliases": raw_metadata.get("aliases", []),
                "keywords": raw_metadata.get("keywords", []),
                "functions": raw_metadata.get("functions", []),
                "cautions": raw_metadata.get("cautions", []),
                "dose_range": raw_metadata.get("dose_range"),
            }
        )
    if row.entry_type == "formula":
        metadata.update(
            {
                "name": raw_metadata.get("name", row.title),
                "normalized_name": raw_metadata.get("normalized_name", row.title),
                "components": raw_metadata.get("components", []),
                "keywords": raw_metadata.get("keywords", []),
                "functions": raw_metadata.get("functions", []),
                "cautions": raw_metadata.get("cautions", []),
                "usage_note": raw_metadata.get("usage_note", ""),
                "not_clinical_recommendation": True,
            }
        )

    evidence = {
        "source_id": row.source_id,
        "source_ids": source_ids,
        "source_records": source_records,
        "original_source": original_source,
        "entry_id": row.entry_id,
        "title": row.title,
        "snippet": row.content[:240],
        "source_type": row.source_type,
        "confidence": round(max(confidence, 0.01), 2),
        "metadata": metadata,
        "retrieved_by": retrieved_by,
    }
    if row.entry_type == "formula":
        evidence["components"] = metadata["components"]
        evidence["not_clinical_recommendation"] = True
    return evidence


def _visible_to_role(row: KnowledgeEntry, role: str | None) -> bool:
    metadata = row.metadata_json or {}
    visibility = metadata.get("visibility")
    if visibility:
        values = visibility if isinstance(visibility, list) else [visibility]
        allowed = {str(value) for value in values}
        if role == "student" and not ({"student", "student_learning"} & allowed):
            return False
        if role != "student" and not (role and role in allowed):
            return False
    return _source_policy_allows(metadata, role)


def _source_policy_allows(metadata: dict[str, Any], role: str | None) -> bool:
    if role not in {"doctor", "student"}:
        return True
    catalog = metadata.get("source_catalog") if isinstance(metadata.get("source_catalog"), dict) else {}
    if role == "doctor":
        return catalog.get("doctor_evidence_allowed", metadata.get("doctor_evidence_allowed", True)) is not False
    return catalog.get("student_visible", metadata.get("student_visible", True)) is not False


def _matches_catalog_query(row: KnowledgeEntry, query: str) -> bool:
    metadata = row.metadata_json or {}
    values = [row.title, row.entry_id, row.source_id, row.source_title, *[str(value) for value in row.keywords_json or []]]
    aliases = metadata.get("aliases") or []
    values.extend(str(value) for value in aliases)
    return any(query in value.lower() for value in values if value)


def _to_catalog_item(row: KnowledgeEntry, db: Session | None = None) -> dict[str, Any]:
    metadata = row.metadata_json or {}
    source_catalog = metadata.get("source_catalog") if isinstance(metadata.get("source_catalog"), dict) else {}
    components = metadata.get("components") if isinstance(metadata.get("components"), list) else []
    component_names = [
        str(component.get("normalized_name") or component.get("herb_name") or "")
        for component in components
        if isinstance(component, dict) and (component.get("normalized_name") or component.get("herb_name"))
    ]
    claims = _claims_for_catalog(db, row.title) if db is not None and row.entry_type in {"herb", "formula"} else []
    catalog_functions = _unique_strings(
        [
            *(_string_list(metadata.get("display_functions")) or _string_list(metadata.get("functions"))),
            *_string_list(metadata.get("chpa_efficacies")),
        ]
    )
    catalog_symptoms = _unique_strings(
        [
            *_string_list(metadata.get("symptoms")),
            *_string_list(metadata.get("chpa_symptoms")),
        ]
    )
    catalog_syndromes = _string_list(metadata.get("syndromes"))
    indirect_functions = _string_list(metadata.get("indirect_functions"))
    indirect_symptoms = _string_list(metadata.get("indirect_symptoms"))
    indirect_syndromes = _string_list(metadata.get("indirect_syndromes"))
    related_formulas = metadata.get("related_formulas") if isinstance(metadata.get("related_formulas"), list) else []
    claimed_functions = sorted({claim.object_label for claim in claims if claim.predicate in {"HAS_TCM_FUNCTION", "HAS_EFFICACY", "HAS_FUNCTION"}})
    display_title = str(metadata.get("display_name") or row.title)
    incompatibilities = incompatibilities_for_herb(display_title) if row.entry_type == "herb" else []
    return {
        "entry_id": row.entry_id,
        "source_id": row.source_id,
        "entry_type": row.entry_type,
        "title": display_title,
        "keywords": [str(value) for value in row.keywords_json or []],
        "aliases": [str(value) for value in metadata.get("aliases", []) if value],
        "component_names": component_names,
        "source_title": row.source_title,
        "source_type": row.source_type,
        "visibility": _string_list(metadata.get("visibility")),
        "license": str(metadata.get("license") or source_catalog.get("license") or ""),
        "license_note": str(metadata.get("license_note") or source_catalog.get("license_note") or ""),
        "commit_hash": str(metadata.get("commit_hash") or source_catalog.get("commit_hash") or ""),
        "source_records": _safe_source_records(metadata.get("source_records")),
        "content_preview": row.content[:240],
        # Catalog imports carry structured functions directly. Claims are an
        # optional enrichment, not the sole source of the table column.
        "functions": catalog_functions or claimed_functions,
        "symptoms": catalog_symptoms or sorted({claim.object_label for claim in claims if claim.predicate in {"ASSOCIATED_WITH_TCM_SYMPTOM", "ASSOCIATED_WITH_TCM_SYNDROME", "ASSOCIATED_WITH", "REFERENCES_SYMPTOM"}}),
        "syndromes": catalog_syndromes,
        "chpa_efficacies": _string_list(metadata.get("chpa_efficacies")),
        "chpa_symptoms": _string_list(metadata.get("chpa_symptoms")),
        "chpa_relation_note": str(metadata.get("chpa_relation_note") or ""),
        "chpa_translation_review_status": str(metadata.get("chpa_translation_review_status") or ""),
        "indirect_functions": indirect_functions,
        "indirect_symptoms": indirect_symptoms,
        "indirect_syndromes": indirect_syndromes,
        "related_formulas": related_formulas[:20],
        "formula_relation_note": str(metadata.get("formula_relation_note") or ""),
        "formula_relations": sorted({claim.object_label for claim in claims if claim.predicate == "REFERENCES_FORMULA"}),
        "claim_status": sorted({claim.review_status for claim in claims}),
        "incompatible_herbs": incompatibilities,
    }


def _is_catalog_ready_herb(row: KnowledgeEntry) -> bool:
    """Catalog visibility requires a normalized Chinese name, not inferred efficacy."""
    metadata = row.metadata_json or {}
    name = str(metadata.get("display_name") or row.title)
    return is_valid_tcm_herb_label(name)


def _safe_catalog_metadata(row: KnowledgeEntry, db: Session | None = None) -> dict[str, Any]:
    metadata = row.metadata_json or {}
    source_catalog = metadata.get("source_catalog") if isinstance(metadata.get("source_catalog"), dict) else {}
    display_title = str(metadata.get("display_name") or row.title)
    return {
        "source_ids": _string_list(metadata.get("source_ids")) or [row.source_id],
        "source_records": _safe_source_records(metadata.get("source_records")),
        "original_source": str(metadata.get("original_source") or source_catalog.get("name") or row.source_title),
        "repository_url": str(metadata.get("repository_url") or source_catalog.get("repository_url") or ""),
        "license": str(metadata.get("license") or source_catalog.get("license") or ""),
        "license_note": str(metadata.get("license_note") or source_catalog.get("license_note") or ""),
        "commit_hash": str(metadata.get("commit_hash") or source_catalog.get("commit_hash") or ""),
        "processing_script": str(metadata.get("processing_script") or source_catalog.get("processing_script") or ""),
        "processing_version": str(metadata.get("processing_version") or source_catalog.get("processing_version") or ""),
        "evidence_level": str(metadata.get("evidence_level") or source_catalog.get("evidence_level") or ""),
        "visibility": _string_list(metadata.get("visibility")),
        "incompatible_herbs": incompatibilities_for_herb(display_title) if row.entry_type == "herb" else [],
        "aliases": [str(value) for value in metadata.get("aliases", []) if value],
        "component_names": _to_catalog_item(row)["component_names"],
        "indirect_functions": _string_list(metadata.get("indirect_functions")),
        "indirect_symptoms": _string_list(metadata.get("indirect_symptoms")),
        "indirect_syndromes": _string_list(metadata.get("indirect_syndromes")),
        "chpa_efficacies": _string_list(metadata.get("chpa_efficacies")),
        "chpa_symptoms": _string_list(metadata.get("chpa_symptoms")),
        "chpa_relation_note": str(metadata.get("chpa_relation_note") or ""),
        "chpa_translation_review_status": str(metadata.get("chpa_translation_review_status") or ""),
        "related_formulas": (metadata.get("related_formulas") if isinstance(metadata.get("related_formulas"), list) else [])[:20],
        "formula_relation_note": str(metadata.get("formula_relation_note") or ""),
        "claims": [
            {
                "predicate": claim.predicate,
                "object_label": claim.object_label,
                "source_id": claim.source_id,
                "source_record_id": claim.source_record_id,
                "source_revision": claim.source_revision,
                "license_status": claim.license_status,
                "distribution_scope": claim.distribution_scope,
                "content_risk_level": claim.content_risk_level,
                "review_status": claim.review_status,
                "source_excerpt": claim.source_excerpt,
            }
            for claim in _claims_for_catalog(db, row.title)
        ] if db is not None else [],
    }


def _safe_source_records(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {
            "source_id": str(item.get("source_id", "")),
            "source_record_id": str(item.get("source_record_id", "")),
            "source_file": str(item.get("source_file", "")),
        }
        for item in value
        if isinstance(item, dict)
    ]


def _string_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value] if value else []
    return [str(item) for item in values if item]


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _claims_for_catalog(db: Session | None, title: str) -> list[KnowledgeClaim]:
    if db is None:
        return []
    return db.scalars(select(KnowledgeClaim).where(KnowledgeClaim.subject_label == title).order_by(KnowledgeClaim.predicate, KnowledgeClaim.object_label)).all()


def _is_demo_evidence(row: KnowledgeEntry, metadata: dict[str, Any], source_catalog: dict[str, Any]) -> bool:
    evidence_level = str(metadata.get("evidence_level") or source_catalog.get("evidence_level") or "").lower()
    source_id = row.source_id.lower()
    source_type = row.source_type.lower()
    return evidence_level == "demo" or "mock" in source_id or "sample" in source_type


def _require(payload: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in payload or payload[key] in (None, "")]
    if missing:
        raise ValueError(f"Missing required knowledge fields: {', '.join(missing)}")
