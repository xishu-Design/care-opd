from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeClaim, KnowledgeEntry, KnowledgeSource, utcnow
from app.knowledge.entity_quality import is_valid_tcm_herb_label


SHENNONG_SOURCE_ID = "SRC_SHENNONG_TCM_DATASET"
SHENNONG_SOURCE_TITLE = "ShenNong TCM Dataset"
_HERB_QUERY = re.compile(r"(?P<herb>[\u4e00-\u9fff]{2,8})(?:有|具备|有什么|有哪些).{0,8}(?:功效|作用|主治)")

# Curated teaching vocabulary. A relationship is created only when its term or
# alias literally occurs in the source text; this maps wording to a stable TCM label.
_TCM_FUNCTIONS = {
    "补气": ("补气", "益气"),
    "升阳": ("升阳", "升阳举陷"),
    "固表": ("固表", "固表止汗"),
    "利水消肿": ("利水消肿", "利水"),
    "托毒生肌": ("托毒生肌",),
    "健脾": ("健脾",),
    "养血": ("养血", "补血"),
    "活血化瘀": ("活血化瘀", "活血祛瘀"),
    "行气": ("行气", "理气"),
    "疏肝理气": ("疏肝理气",),
    "化湿": ("化湿", "燥湿"),
    "清热": ("清热",),
    "解毒": ("解毒",),
    "温中散寒": ("温中散寒", "温中"),
    "祛风除湿": ("祛风除湿",),
    "化痰": ("化痰",),
    "止咳平喘": ("止咳平喘",),
    "安神": ("安神", "养心安神", "宁心安神"),
    "平肝潜阳": ("平肝潜阳",),
    "息风止痉": ("息风止痉",),
    "润肺止咳": ("润肺止咳",),
    "消食化积": ("消食化积", "消食"),
    "止痛": ("止痛",),
}
_TCM_SYMPTOMS = {
    "乏力": ("乏力",),
    "气短": ("气短",),
    "自汗": ("自汗",),
    "盗汗": ("盗汗",),
    "心悸": ("心悸",),
    "失眠": ("失眠",),
    "多梦": ("多梦",),
    "食少": ("食少", "纳少"),
    "便溏": ("便溏",),
    "脘腹胀满": ("脘腹胀满",),
    "面色萎黄": ("面色萎黄",),
    "舌淡": ("舌淡",),
    "脉细": ("脉细",),
}
_TCM_SYNDROMES = {
    "气虚证": ("气虚证", "气虚体弱", "气虚"),
    "血虚证": ("血虚证", "血虚"),
    "阴虚证": ("阴虚证", "阴虚"),
    "阳虚证": ("阳虚证", "阳虚"),
    "脾虚证": ("脾虚证", "脾虚"),
    "脾胃虚弱证": ("脾胃虚弱证", "脾胃虚弱"),
    "气血两虚证": ("气血两虚证", "气血两虚"),
    "心脾两虚证": ("心脾两虚证", "心脾两虚"),
    "心肾不交证": ("心肾不交证", "心肾不交"),
    "阴虚火旺证": ("阴虚火旺证", "阴虚火旺"),
    "肝郁气滞证": ("肝郁气滞证", "肝郁气滞"),
    "痰湿证": ("痰湿证", "痰湿"),
    "湿热证": ("湿热证", "湿热"),
    "瘀血证": ("瘀血证", "瘀血"),
    "食积证": ("食积证", "食积"),
}


def import_shennong_jsonl(db: Session, *, path: Path | str, revision: str, allow_internal: bool, replace_source: bool = True) -> dict[str, Any]:
    if not allow_internal:
        raise ValueError("ALLOW_UNVERIFIED_INTERNAL_SOURCES=true is required for internal knowledge import")
    file_path = Path(path)
    source = _upsert_source(db, revision)
    herb_names = sorted(
        {
            str(value)
            for value in db.scalars(select(KnowledgeEntry.title).where(KnowledgeEntry.entry_type == "herb")).all()
            if value and is_valid_tcm_herb_label(str(value))
        },
        key=len,
        reverse=True,
    )
    if replace_source:
        db.execute(delete(KnowledgeClaim).where(KnowledgeClaim.source_id == SHENNONG_SOURCE_ID))
        db.execute(delete(KnowledgeEntry).where(KnowledgeEntry.source_id == SHENNONG_SOURCE_ID))
        db.flush()
    imported = claims = skipped = 0
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            query = str(row.get("query") or row.get("instruction") or "").strip()
            response = str(row.get("response") or row.get("output") or "").strip()
            if not query or not response:
                skipped += 1
                continue
            entry_id = _stable_id("DOC", revision, str(line_number), query, response)
            entry = db.scalar(select(KnowledgeEntry).where(KnowledgeEntry.source_id == SHENNONG_SOURCE_ID, KnowledgeEntry.entry_id == entry_id))
            if entry is None:
                entry = KnowledgeEntry(entry_id=entry_id, source_id=SHENNONG_SOURCE_ID, entry_type="instruction_qa", title=query[:240])
            entry.content = f"问题：{query}\n回答：{response}"
            entry.keywords_json = _keywords(query)
            entry.source_title = source.source_title
            entry.source_type = source.source_type
            entry.metadata_json = {
                "source_id": SHENNONG_SOURCE_ID,
                "source_ids": [SHENNONG_SOURCE_ID],
                "source_record_id": f"line_{line_number}",
                "source_revision": revision,
                "license": "Apache-2.0",
                "license_status": "verified",
                "distribution_scope": "internal_only",
                "content_risk_level": "generated_medical_content",
                "visibility": ["doctor", "student"],
                "doctor_evidence_allowed": False,
                "student_visible": True,
                "non_clinical_authority": True,
            }
            entry.updated_at = utcnow()
            db.add(entry)
            imported += 1
            claims += _extract_claims(db, query=query, response=response, revision=revision, record_id=f"line_{line_number}", herb_names=herb_names)
            if imported % 500 == 0:
                db.flush()
                db.expunge_all()
                source = _upsert_source(db, revision)
    db.flush()
    return {"source_id": SHENNONG_SOURCE_ID, "documents": imported, "claims": claims, "skipped": skipped, "revision": revision}


def list_claims_for_subject(db: Session, subject: str) -> list[KnowledgeClaim]:
    return db.scalars(select(KnowledgeClaim).where(KnowledgeClaim.subject_label == subject).order_by(KnowledgeClaim.predicate, KnowledgeClaim.object_label)).all()


def normalize_shennong_claims(db: Session) -> dict[str, int]:
    """Replace permissive generated-text claims with source-grounded TCM terms."""
    rows = db.scalars(select(KnowledgeClaim).where(KnowledgeClaim.source_id == SHENNONG_SOURCE_ID)).all()
    records = [
        (row.subject_label, row.subject_type, row.source_revision, row.source_record_id, row.source_excerpt)
        for row in rows
    ]
    db.execute(delete(KnowledgeClaim).where(KnowledgeClaim.source_id == SHENNONG_SOURCE_ID))
    db.flush()
    imported = 0
    seen: set[tuple[str, str, str, str, str]] = set()
    for subject, subject_type, revision, record_id, excerpt in records:
        if subject_type == "herb" and not is_valid_tcm_herb_label(subject):
            continue
        for predicate, label in _tcm_terms(excerpt):
            key = (subject, predicate, label, revision, record_id)
            if key in seen:
                continue
            seen.add(key)
            imported += _upsert_claim(db, subject, subject_type, predicate, label, revision, record_id, excerpt)
    db.flush()
    return {"source_records": len(records), "claims": imported}


def _upsert_source(db: Session, revision: str) -> KnowledgeSource:
    source = db.scalar(select(KnowledgeSource).where(KnowledgeSource.source_id == SHENNONG_SOURCE_ID))
    if source is None:
        source = KnowledgeSource(source_id=SHENNONG_SOURCE_ID, source_title=SHENNONG_SOURCE_TITLE, source_type="huggingface_instruction_dataset")
    source.source_version = revision
    source.evidence_level = "internal_non_clinical_generated"
    source.license_note = "Apache-2.0; internal-only generated medical content. Never use as patient-facing or clinical evidence."
    db.add(source)
    return source


def _extract_claims(db: Session, *, query: str, response: str, revision: str, record_id: str, herb_names: list[str]) -> int:
    match = _HERB_QUERY.search(query)
    if not match:
        return 0
    herb = next((name for name in herb_names if name in query), "")
    if not herb:
        return 0
    count = 0
    for predicate, value in _tcm_terms(response):
        count += _upsert_claim(db, herb, "herb", predicate, value, revision, record_id, response)
    return count


def _tcm_terms(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for predicate, vocabulary in (
        ("HAS_TCM_FUNCTION", _TCM_FUNCTIONS),
        ("ASSOCIATED_WITH_TCM_SYMPTOM", _TCM_SYMPTOMS),
        ("ASSOCIATED_WITH_TCM_SYNDROME", _TCM_SYNDROMES),
    ):
        for canonical, aliases in vocabulary.items():
            if any(alias in text for alias in aliases):
                found.append((predicate, canonical))
    return found


def _upsert_claim(db: Session, subject: str, subject_type: str, predicate: str, object_label: str, revision: str, record_id: str, response: str) -> int:
    if object_label == subject or subject in object_label:
        return 0
    claim_id = _stable_id("CLAIM", subject, predicate, object_label, SHENNONG_SOURCE_ID, revision, record_id)
    claim = db.scalar(select(KnowledgeClaim).where(KnowledgeClaim.claim_id == claim_id))
    if claim is None:
        claim = KnowledgeClaim(claim_id=claim_id, subject_label=subject, subject_type=subject_type, predicate=predicate, object_label=object_label, source_id=SHENNONG_SOURCE_ID, source_record_id=record_id)
    claim.source_revision = revision
    claim.source_excerpt = response[:480]
    claim.license_status = "verified"
    claim.distribution_scope = "internal_only"
    claim.content_risk_level = "non_clinical"
    claim.review_status = "imported"
    claim.extraction_method = "tcm_controlled_vocabulary_v2"
    claim.metadata_json = {
        "non_clinical_authority": True,
        "doctor_evidence_allowed": False,
        "visibility": ["doctor", "student"],
        "terminology_layer": "tcm_controlled_vocabulary",
    }
    claim.updated_at = utcnow()
    db.add(claim)
    return 1


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:48]


def _keywords(query: str) -> list[str]:
    match = _HERB_QUERY.search(query)
    return [match.group("herb")] if match else []
