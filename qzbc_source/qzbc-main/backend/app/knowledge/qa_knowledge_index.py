from __future__ import annotations

import json
import math
import re
import sqlite3
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import KnowledgeEntry, LearningCase


HAN_SEQUENCE_RE = re.compile(r"[\u3400-\u9fff]{2,}")
LATIN_WORD_RE = re.compile(r"[A-Za-z0-9_-]{2,}")
QUESTION_PREFIX_RE = re.compile(r"^问题：")
ANSWER_PREFIX_RE = re.compile(r"^回答：")
STOP_GRAMS = {
    "什么",
    "怎么",
    "如何",
    "哪些",
    "是否",
    "可以",
    "请问",
    "患者",
    "中医",
    "有关",
    "一下",
    "药物",
    "治疗",
    "功效",
    "作用",
    "的药",
    "物是",
    "是什",
    "什么",
    "的是",
}
GENERIC_QUERY_FRAGMENTS = (
    "什么", "药物", "治疗", "功效", "作用", "哪些", "的是", "如何", "怎么", "时候", "患者", "的", "是",
    "请问", "详细", "讲解", "解释", "介绍", "问题", "复习", "给出", "区别", "继续", "具体",
    "我", "你", "他", "她", "当", "给",
)


def build_qa_knowledge_index(
    db: Session,
    *,
    index_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    path = _index_path(index_path)
    report = Path(report_path) if report_path else path.with_name("build_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".next.db")
    if temporary.exists():
        temporary.unlink()

    connection = sqlite3.connect(temporary)
    source_counts: dict[str, int] = {}
    total = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            """
            CREATE VIRTUAL TABLE qa_documents USING fts5(
                doc_id UNINDEXED,
                source_id UNINDEXED,
                entry_type UNINDEXED,
                title,
                question,
                answer,
                search_terms,
                visibility UNINDEXED,
                review_status UNINDEXED,
                doctor_allowed UNINDEXED,
                patient_allowed UNINDEXED,
                student_allowed UNINDEXED,
                metadata_json UNINDEXED,
                tokenize = 'unicode61'
            )
            """
        )
        batch: list[tuple[Any, ...]] = []
        for document in _documents(db):
            batch.append(_row(document))
            source_id = str(document["source_id"])
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            total += 1
            if len(batch) >= 1000:
                _insert_batch(connection, batch)
                batch.clear()
        if batch:
            _insert_batch(connection, batch)
        connection.commit()
        connection.execute("INSERT INTO qa_documents(qa_documents) VALUES ('optimize')")
        connection.commit()
    finally:
        connection.close()

    temporary.replace(path)
    payload = {
        "status": "success",
        "generated_at": datetime.now(UTC).isoformat(),
        "index_path": str(path),
        "documents": total,
        "source_counts": source_counts,
        "retrieval": "sqlite_fts_candidate_generation+bge_m3_rerank",
        "role_policy": {
            "student": ["SRC_MTCMB", "SRC_SHENNONG_TCM_DATASET"],
            "doctor": "only explicitly doctor-reviewed QA records",
            "patient": "no raw QA records",
        },
    }
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def search_qa_knowledge(
    *,
    query: str,
    role: str,
    limit: int = 5,
    index_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.qa_knowledge_enabled or role == "patient":
        return []
    path = _index_path(index_path)
    if not path.exists() or not query.strip():
        return []
    role_column = {
        "doctor": "doctor_allowed",
        "student": "student_allowed",
    }.get(role)
    if role_column is None:
        return []

    match_query = _fts_query(query)
    if not match_query:
        return []
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"""
            SELECT *, bm25(qa_documents, 0.0, 0.0, 0.0, 1.2, 2.0, 0.8, 4.0) AS lexical_rank
            FROM qa_documents
            WHERE qa_documents MATCH ? AND {role_column} = '1'
            ORDER BY lexical_rank
            LIMIT ?
            """,
            (match_query, max(limit, settings.qa_knowledge_candidate_limit)),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        return []

    candidates = [dict(row) for row in rows]
    required_anchors = _required_domain_anchors(query)
    if required_anchors:
        candidates = [
            row
            for row in candidates
            if any(anchor in f"{row['title']} {row['question']}" for anchor in required_anchors)
        ]
        if not candidates:
            return []
    informative = set(_informative_terms(query))
    if informative:
        matched = [
            (
                len(informative & set(_search_terms(f"{row['title']} {row['question']}"))),
                row,
            )
            for row in candidates
        ]
        candidates = [
            row
            for overlap, row in sorted(matched, key=lambda item: (-item[0], float(item[1]["lexical_rank"])))
            if overlap > 0
        ]
        if not candidates:
            return []
    ranked, rerank_used = _rerank_with_bge_m3(query, candidates)
    evidence = []
    for index, row in enumerate(ranked[:limit]):
        metadata = json.loads(row.get("metadata_json") or "{}")
        metadata.update(
            {
                "source_id": row["source_id"],
                "entry_type": row["entry_type"],
                "visibility": json.loads(row.get("visibility") or "[]"),
                "review_status": row.get("review_status") or "",
                "retrieval_mode": "qa_fts+bge_m3" if rerank_used else "qa_fts",
                "non_clinical_authority": row["source_id"] in {"SRC_MTCMB", "SRC_SHENNONG_TCM_DATASET"},
            }
        )
        evidence.append(
            {
                "source_id": row["source_id"],
                "source_ids": [row["source_id"]],
                "entry_id": row["doc_id"],
                "entry_type": row["entry_type"],
                "title": row["title"],
                "snippet": _snippet(row),
                "confidence": round(max(0.1, 1.0 - index * 0.08), 2),
                "retrieved_by": "qa_knowledge_bge_m3" if rerank_used else "qa_knowledge_fts",
                "retrieval_mode": metadata["retrieval_mode"],
                "original_source": metadata.get("original_source") or row["source_id"],
                "source_records": metadata.get("source_records") or [],
                "metadata": metadata,
            }
        )
    return evidence


def qa_knowledge_health(index_path: Path | str | None = None) -> dict[str, Any]:
    path = _index_path(index_path)
    if not path.exists():
        return {"status": "unavailable", "documents": 0, "index_path": str(path)}
    connection = sqlite3.connect(path)
    try:
        count = int(connection.execute("SELECT count(*) FROM qa_documents").fetchone()[0])
    finally:
        connection.close()
    return {"status": "ok" if count else "degraded", "documents": count, "index_path": str(path)}


def _documents(db: Session) -> Iterable[dict[str, Any]]:
    qa_entries = db.scalars(
        select(KnowledgeEntry)
        .where(KnowledgeEntry.entry_type == "instruction_qa")
        .order_by(KnowledgeEntry.id)
    )
    for row in qa_entries:
        metadata = dict(row.metadata_json or {})
        question, answer = _split_qa_content(row.content)
        if not question or not answer:
            continue
        visibility = _strings(metadata.get("visibility")) or ["student"]
        yield {
            "doc_id": row.entry_id,
            "source_id": row.source_id,
            "entry_type": "instruction_qa",
            "title": row.title or question,
            "question": question,
            "answer": answer,
            "visibility": visibility,
            "review_status": str(metadata.get("review_status") or "imported_unreviewed"),
            "doctor_allowed": _allowed(metadata, "doctor"),
            "patient_allowed": False,
            "student_allowed": _allowed(metadata, "student", default=True),
            "metadata": {
                "original_source": row.source_title,
                "source_records": [
                    {
                        "source_id": row.source_id,
                        "source_record_id": str(metadata.get("source_record_id") or row.entry_id),
                        "source_file": str(metadata.get("source_file") or ""),
                    }
                ],
                "license": metadata.get("license", ""),
                "content_risk_level": metadata.get("content_risk_level", ""),
            },
        }

    learning_cases = db.scalars(select(LearningCase).order_by(LearningCase.id))
    for row in learning_cases:
        case = row.anonymized_case if isinstance(row.anonymized_case, dict) else {}
        if case.get("source_id") != "SRC_MTCMB" or case.get("deidentified") is not True:
            continue
        payload = case.get("training_payload") if isinstance(case.get("training_payload"), dict) else {}
        question = str(payload.get("question") or "").strip()
        answer = _learning_answer(payload, row.standard_analysis)
        if not question or not answer:
            continue
        yield {
            "doc_id": str(case.get("case_id") or f"LEARNING_CASE_{row.id}"),
            "source_id": "SRC_MTCMB",
            "entry_type": "learning_qa",
            "title": row.title,
            "question": _question_with_options(question, payload.get("options")),
            "answer": answer,
            "visibility": ["student_learning"],
            "review_status": str(case.get("review_status") or "approved_training_only"),
            "doctor_allowed": False,
            "patient_allowed": False,
            "student_allowed": True,
            "metadata": {
                "original_source": str(payload.get("original_source") or "MTCMB"),
                "source_records": [
                    {
                        "source_id": "SRC_MTCMB",
                        "source_record_id": str(case.get("source_record_id") or ""),
                        "source_file": str(case.get("source_file") or ""),
                    }
                ],
                "license": "Apache-2.0",
                "training_only": True,
            },
        }


def _row(document: dict[str, Any]) -> tuple[Any, ...]:
    return (
        document["doc_id"],
        document["source_id"],
        document["entry_type"],
        document["title"],
        document["question"],
        document["answer"],
        " ".join(_search_terms(f"{document['title']} {document['question']}")),
        json.dumps(document["visibility"], ensure_ascii=False),
        document["review_status"],
        "1" if document["doctor_allowed"] else "0",
        "1" if document["patient_allowed"] else "0",
        "1" if document["student_allowed"] else "0",
        json.dumps(document["metadata"], ensure_ascii=False),
    )


def _insert_batch(connection: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
    connection.executemany(
        """
        INSERT INTO qa_documents(
            doc_id, source_id, entry_type, title, question, answer, search_terms,
            visibility, review_status, doctor_allowed, patient_allowed,
            student_allowed, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


def _fts_query(query: str) -> str:
    terms = _search_terms(query)
    if not terms:
        return ""
    preferred = _informative_terms(query)
    selected = (preferred or terms)[:16]
    return " OR ".join(f'search_terms:"{term.replace(chr(34), chr(34) * 2)}"' for term in selected)


def _search_terms(text: str) -> list[str]:
    terms: list[str] = []
    for sequence in HAN_SEQUENCE_RE.findall(text):
        if len(sequence) <= 4:
            terms.append(sequence)
        for width in (2, 3, 4):
            terms.extend(sequence[index : index + width] for index in range(max(0, len(sequence) - width + 1)))
    terms.extend(word.lower() for word in LATIN_WORD_RE.findall(text))
    return list(dict.fromkeys(term for term in terms if term.strip()))


def _informative_terms(text: str) -> list[str]:
    return [
        term
        for term in _search_terms(text)
        if term not in STOP_GRAMS and not any(fragment in term for fragment in GENERIC_QUERY_FRAGMENTS)
    ]


def _required_domain_anchors(query: str) -> tuple[str, ...]:
    compact = re.sub(r"\s+", "", query)
    if "步骤" in compact and any(term in compact for term in ("看病", "诊断", "诊疗", "接诊")):
        return ("看病", "诊断", "诊疗", "接诊", "问诊", "四诊", "辨证", "辩证", "施治")
    return ()


def _rerank_with_bge_m3(query: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    settings = get_settings()
    if not settings.qa_knowledge_rerank_enabled:
        return rows, False
    texts = [query, *[f"{row['title']}\n{row['question']}\n{row['answer']}"[:1800] for row in rows]]
    try:
        request = urllib.request.Request(
            settings.ollama_base_url.rstrip("/") + "/api/embed",
            data=json.dumps({"model": settings.ollama_embedding_model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=settings.ollama_generate_timeout_seconds) as response:
            embeddings = json.loads(response.read().decode()).get("embeddings", [])
        if len(embeddings) != len(texts):
            return rows, False
        query_vector = embeddings[0]
        scored = [
            (_cosine(query_vector, embeddings[index + 1]), index, row)
            for index, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [row for _, _, row in scored], True
    except Exception:
        return rows, False


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _split_qa_content(content: str) -> tuple[str, str]:
    question = answer = ""
    for line in content.splitlines():
        if QUESTION_PREFIX_RE.match(line):
            question = QUESTION_PREFIX_RE.sub("", line, count=1).strip()
        elif ANSWER_PREFIX_RE.match(line):
            answer = ANSWER_PREFIX_RE.sub("", line, count=1).strip()
        elif answer:
            answer += "\n" + line.strip()
    return question, answer


def _learning_answer(payload: dict[str, Any], standard_analysis: str) -> str:
    raw_answer = str(payload.get("answer") or standard_analysis or "").strip()
    options = payload.get("options")
    if isinstance(options, dict):
        key = raw_answer.rstrip(".、 ").upper()
        value = options.get(key) or options.get(raw_answer)
        return f"{raw_answer}：{value}" if value else raw_answer
    if isinstance(options, list) and raw_answer:
        key = raw_answer.rstrip(".、 ").upper()
        index = ord(key) - ord("A") if len(key) == 1 and "A" <= key <= "Z" else -1
        if 0 <= index < len(options):
            return f"{raw_answer}：{_strip_option_prefix(str(options[index]))}"
    return raw_answer


def _question_with_options(question: str, options: Any) -> str:
    if isinstance(options, dict):
        rendered = "；".join(f"{key}.{value}" for key, value in options.items())
    elif isinstance(options, list):
        rendered = "；".join(str(value) for value in options)
    else:
        rendered = ""
    return f"{question}\n选项：{rendered}" if rendered else question


def _strip_option_prefix(value: str) -> str:
    return re.sub(r"^[A-Z][.、．]\s*", "", value.strip(), count=1)


def _snippet(row: dict[str, Any]) -> str:
    return f"问题：{row['question']}\n参考答案：{row['answer']}"[:1200]


def _allowed(metadata: dict[str, Any], role: str, *, default: bool = False) -> bool:
    catalog = metadata.get("source_catalog") if isinstance(metadata.get("source_catalog"), dict) else {}
    key = "doctor_evidence_allowed" if role == "doctor" else "student_visible"
    return bool(catalog.get(key, metadata.get(key, default)))


def _strings(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value] if value else []
    return [str(item) for item in values if item]


def _index_path(value: Path | str | None) -> Path:
    path = Path(value or get_settings().qa_knowledge_index_path)
    if path.is_absolute():
        return path
    return Path.cwd().parent / path if Path.cwd().name == "backend" else Path.cwd() / path
