from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR
from app.db.models import KnowledgeEntry, LightRAGDocManifest
from app.lightrag_integration.config import DEFAULT_REPORTS_DIR, get_lightrag_config
from app.lightrag_integration.schemas import LightRAGDoc
from sqlalchemy import delete, select
from sqlalchemy.orm import Session


CONTENT_FILES = {
    "herb": "herbs.json",
    "formula": "formulas.json",
    "contraindication": "contraindications.json",
    "diet_precaution": "diet_precautions.json",
    "public_education": "public_education.json",
    "case": "learning_cases.json",
}


def build_lightrag_docs(
    *,
    content_data_dir: Path | str = DEFAULT_CONTENT_DATA_DIR,
    output_path: Path | str | None = None,
    report_path: Path | str = DEFAULT_REPORTS_DIR / "lightrag_doc_build_report.json",
) -> dict[str, Any]:
    started_at = _now()
    content_dir = Path(content_data_dir)
    config = get_lightrag_config()
    output = Path(output_path) if output_path else config.input_docs_path
    source_map = _source_map(_items(content_dir / "sources.json"))
    source_catalog = _source_catalog_map(content_dir / "source_catalog.json")
    docs: list[LightRAGDoc] = []
    skipped: list[dict[str, str]] = []

    for entry_type, file_name in CONTENT_FILES.items():
        for index, item in enumerate(_items(content_dir / file_name), start=1):
            if entry_type == "case" and _source_id(item) != "SRC_MTCMB":
                continue
            doc, issue = _doc_from_item(entry_type, item, index, source_map, source_catalog)
            if issue:
                skipped.append(issue | {"file_name": file_name})
                continue
            docs.append(doc)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for doc in docs:
            file.write(json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "run_id": f"lightrag_docs_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "started_at": started_at,
        "ended_at": _now(),
        "status": "success",
        "content_data_dir": str(content_dir),
        "output_path": str(output),
        "docs_total": len(docs) + len(skipped),
        "docs_written": len(docs),
        "skipped": len(skipped),
        "errors": skipped,
        "entry_type_counts": _entry_type_counts(docs),
    }
    _write_json(Path(report_path), report)
    return report


def load_lightrag_docs(path: Path | str | None = None) -> list[LightRAGDoc]:
    config = get_lightrag_config()
    docs_path = Path(path) if path else config.input_docs_path
    if not docs_path.exists():
        return []
    docs = []
    with docs_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                docs.append(LightRAGDoc.model_validate_json(line))
    return docs


def persist_lightrag_doc_manifest(db: Session, *, snapshot_id: str, docs_path: Path | str) -> int:
    path = Path(docs_path)
    docs = load_lightrag_docs(path)
    db.execute(delete(LightRAGDocManifest).where(LightRAGDocManifest.snapshot_id == snapshot_id))
    for doc in docs:
        metadata = doc.metadata
        source_id = str(metadata.get("source_id") or "")
        db.add(
            LightRAGDocManifest(
                doc_id=doc.doc_id,
                snapshot_id=snapshot_id,
                file_path=str(metadata.get("source_file") or path),
                entry_ids_json=[doc.entry_id],
                source_ids_json=[source_id] if source_id else [],
                visibility_json=list(metadata.get("visibility") or []),
                license_json={source_id: str(metadata.get("license") or "")} if source_id else {},
                content_hash=hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
            )
        )
    db.commit()
    return len(docs)


def build_lightrag_docs_from_db(
    db: Session,
    *,
    output_path: Path | str | None = None,
    report_path: Path | str = DEFAULT_REPORTS_DIR / "lightrag_doc_build_report.json",
) -> dict[str, Any]:
    started_at = _now()
    config = get_lightrag_config()
    output = Path(output_path) if output_path else config.input_docs_path
    docs: list[LightRAGDoc] = []
    skipped: list[dict[str, str]] = []
    rows = db.scalars(select(KnowledgeEntry).order_by(KnowledgeEntry.entry_type, KnowledgeEntry.entry_id)).all()
    for index, row in enumerate(rows, start=1):
        if not row.source_id:
            skipped.append({"code": "MISSING_SOURCE_ID", "entry_id": row.entry_id, "entry_type": row.entry_type})
            continue
        metadata = dict(row.metadata_json or {})
        metadata.setdefault("source_id", row.source_id)
        metadata.setdefault("source_title", row.source_title)
        metadata.setdefault("entry_id", row.entry_id)
        metadata.setdefault("entry_type", row.entry_type)
        docs.append(
            LightRAGDoc(
                doc_id=_stable_doc_id(row.entry_type, row.entry_id, row.source_id),
                entry_id=row.entry_id,
                entry_type=row.entry_type,
                title=row.title,
                text=_db_text(row),
                metadata=metadata,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for doc in docs:
            file.write(json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "run_id": f"lightrag_docs_db_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "started_at": started_at,
        "ended_at": _now(),
        "status": "success",
        "source": "database",
        "output_path": str(output),
        "docs_total": len(docs) + len(skipped),
        "docs_written": len(docs),
        "skipped": len(skipped),
        "errors": skipped,
        "entry_type_counts": _entry_type_counts(docs),
    }
    _write_json(Path(report_path), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GuardTCM LightRAG JSONL documents from Phase 9 content/data.")
    parser.add_argument("--content-data-dir", default=str(DEFAULT_CONTENT_DATA_DIR))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = build_lightrag_docs(content_data_dir=args.content_data_dir, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _doc_from_item(
    entry_type: str,
    item: dict[str, Any],
    index: int,
    source_map: dict[str, dict[str, Any]],
    source_catalog: dict[str, dict[str, Any]],
) -> tuple[LightRAGDoc | None, dict[str, str] | None]:
    entry_id = str(item.get("entry_id") or item.get("case_id") or item.get("id") or "")
    source_id = _source_id(item)
    if not entry_id:
        return None, {"code": "MISSING_ENTRY_ID", "entry_id": "", "source_id": source_id}
    if not source_id:
        return None, {"code": "MISSING_SOURCE_ID", "entry_id": entry_id, "source_id": ""}

    source = source_map.get(source_id, {})
    catalog = source_catalog.get(source_id, {})
    license_value = item.get("license") or catalog.get("license") or source.get("license") or ""
    title = _title(entry_type, item)
    metadata = {
        "source_id": source_id,
        "source_title": source.get("source_title") or catalog.get("name") or source_id,
        "license": license_value,
        "source_record_id": _source_record_id(item),
        "source_file": _source_file(item),
        "source_records": item.get("source_records", []),
        "visibility": item.get("visibility", []),
        "evidence_level": item.get("evidence_level") or source.get("evidence_level") or "",
        "entry_type": entry_type,
        "entry_id": entry_id,
        "processing_script": item.get("processing_script") or catalog.get("processing_script") or source.get("processing_script") or "",
        "processing_version": item.get("processing_version") or catalog.get("processing_version") or source.get("processing_version") or "",
        "repository_url": item.get("repository_url") or catalog.get("repository_url") or source.get("repository_url") or "",
        "raw": item,
    }
    return (
        LightRAGDoc(
            doc_id=_stable_doc_id(entry_type, entry_id, source_id),
            entry_id=entry_id,
            entry_type=entry_type,
            title=title,
            text=_text(entry_type, item, source_id, license_value),
            metadata=metadata,
        ),
        None,
    )


def _text(entry_type: str, item: dict[str, Any], source_id: str, license_value: str) -> str:
    source_record = _source_record_id(item) or _source_file(item) or "未提供"
    if entry_type == "herb":
        return "\n".join(
            [
                "【实体类型】中药",
                f"【名称】{item.get('name') or '未提供'}",
                f"【别名】{_join(item.get('aliases'))}",
                f"【来源ID】{source_id}",
                f"【原始记录】{source_record}",
                f"【许可证】{license_value or '未提供'}",
                f"【已知字段】{_known_fields(item)}",
                f"【缺失字段】{_missing_fields(item)}",
                "【说明】本条目来自开源数据源处理结果，仅用于知识检索和教学展示，不作为患者自行用药依据。",
            ]
        )
    if entry_type == "formula":
        return "\n".join(
            [
                "【实体类型】方剂/处方组成候选",
                f"【名称】{item.get('name') or '未提供'}",
                f"【组成】{_components(item.get('components', []))}",
                f"【来源ID】{source_id}",
                f"【原始记录】{source_record}",
                f"【许可证】{license_value or '未提供'}",
                "【说明】该条目来自开源处方组成数据，仅表示数据中出现过的组成关系，不代表临床推荐。",
            ]
        )
    if entry_type == "diet_precaution":
        return "\n".join(
            [
                "【实体类型】饮食注意事项",
                f"【标题】{item.get('title') or item.get('related_entity') or '未提供'}",
                f"【内容】{item.get('message') or '未提供'}",
                f"【来源ID】{source_id}",
                "【安全边界】仅作为注意事项，不作为治疗性食疗方案。",
            ]
        )
    if entry_type == "public_education":
        return "\n".join(
            [
                "【实体类型】患者科普",
                f"【主题】{item.get('topic') or '未提供'}",
                f"【内容】{item.get('safe_answer') or '未提供'}",
                f"【来源ID】{source_id}",
                "【安全边界】仅作一般科普，不替代医生诊疗意见。",
            ]
        )
    if entry_type == "case":
        payload = item.get("training_payload", {}) if isinstance(item.get("training_payload"), dict) else {}
        return "\n".join(
            [
                "【实体类型】脱敏教学案例",
                f"【标题】{item.get('title') or '未提供'}",
                f"【问题】{payload.get('question') or item.get('standard_analysis') or '未提供'}",
                f"【来源ID】{source_id}",
                f"【原始记录】{source_record}",
                "【用途边界】仅用于学生训练与系统评测，不得作为患者内容或医生临床证据。",
            ]
        )
    return "\n".join(
        [
            "【实体类型】禁忌/慎用/风险知识",
            f"【标题】{item.get('title') or '未提供'}",
            f"【风险类型】{item.get('risk_type') or '未提供'}",
            f"【内容】{item.get('message_for_doctor') or item.get('message') or '未提供'}",
            f"【来源ID】{source_id}",
            f"【许可证】{license_value or '未提供'}",
        ]
    )


def _source_id(item: dict[str, Any]) -> str:
    source_ids = item.get("source_ids")
    if isinstance(source_ids, list) and source_ids:
        return str(source_ids[0])
    return str(item.get("source_id") or "")


def _stable_doc_id(entry_type: str, entry_id: str, source_id: str) -> str:
    digest = hashlib.sha256(f"{source_id}:{entry_type}:{entry_id}".encode("utf-8")).hexdigest()[:20].upper()
    return f"DOC_{entry_type.upper()}_{digest}"


def _source_record_id(item: dict[str, Any]) -> str:
    records = item.get("source_records")
    if isinstance(records, list) and records:
        return str(records[0].get("source_record_id") or "")
    return str(item.get("source_record_id") or "")


def _source_file(item: dict[str, Any]) -> str:
    records = item.get("source_records")
    if isinstance(records, list) and records:
        return str(records[0].get("source_file") or "")
    return str(item.get("source_file") or "")


def _title(entry_type: str, item: dict[str, Any]) -> str:
    if entry_type in {"herb", "formula"}:
        return str(item.get("name") or item.get("entry_id") or "")
    if entry_type == "public_education":
        return str(item.get("topic") or item.get("entry_id") or "")
    return str(item.get("title") or item.get("related_entity") or item.get("entry_id") or "")


def _known_fields(item: dict[str, Any]) -> str:
    fields = item.get("fields_present")
    if isinstance(fields, list) and fields:
        return _join(fields)
    return _join([key for key in ("name", "aliases", "source_records", "keywords") if item.get(key)])


def _missing_fields(item: dict[str, Any]) -> str:
    fields = item.get("fields_missing")
    if isinstance(fields, list) and fields:
        return _join(fields)
    missing = [key for key in ("functions", "cautions", "dose_range") if not item.get(key)]
    return _join(missing) if missing else "未提供"


def _join(value: Any) -> str:
    if isinstance(value, list):
        return "、".join(str(item) for item in value if item) or "未提供"
    return str(value or "未提供")


def _components(components: list[Any]) -> str:
    values = []
    for component in components:
        if isinstance(component, dict):
            values.append(str(component.get("herb_name") or component.get("normalized_name") or ""))
        else:
            values.append(str(component))
    return _join(values)


def _db_text(row: KnowledgeEntry) -> str:
    return "\n".join(
        [
            f"【实体类型】{row.entry_type}",
            f"【标题】{row.title}",
            f"【内容】{row.content or '未提供'}",
            f"【来源ID】{row.source_id}",
            "【说明】本条目来自 GuardTCM 已导入的结构化知识库，仅用于检索增强，不替代医生判断。",
        ]
    )


def _items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = _read_json(path)
    items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain an items list")
    return items


def _source_map(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("source_id")): item for item in items if item.get("source_id")}


def _source_catalog_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = _read_json(path)
    sources = data.get("sources", []) if isinstance(data, dict) else []
    return {str(item.get("source_id")): item for item in sources if item.get("source_id")}


def _entry_type_counts(docs: list[LightRAGDoc]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in docs:
        counts[doc.entry_type] = counts.get(doc.entry_type, 0) + 1
    return counts


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
