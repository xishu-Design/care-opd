from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR
from app.db.models import DietPrecaution, PatientInstruction, Prescription
from app.llm.service import LLMService
from app.services.herb_incompatibility_service import incompatibilities_for_herb


DIET_PRECAUTIONS_FILE = DEFAULT_CONTENT_DATA_DIR / "diet_precautions.json"
SOURCE_URLS = {
    "SRC_DB4403_T142_2021": "https://amr.sz.gov.cn/attachment/0/755/755971/8581454.pdf",
    "SRC_NMPA_DIET_EDU_2020": "https://mpa.xizang.gov.cn/yp/ypkp/202005/t20200518_140692.html",
    "SRC_BJTCM_DIET_EDU_2019": "https://zyj.beijing.gov.cn/sy/whkp/201912/t20191219_1309514.html",
    "SRC_BJTCM_DIET_EDU_2025": "https://zyj.beijing.gov.cn/sy/whkp/202507/t20250718_4152127.html",
}
_SPACE_RE = re.compile(r"[\s\u3000]+")


def upsert_patient_guidance(db: Session, prescription: Prescription) -> dict[str, Any]:
    """Create patient-visible guidance from the final doctor-confirmed snapshot.

    The doctor's text and all interaction lists are deterministic. Ollama may
    only make the doctor's diet note easier to understand; it cannot add
    interaction records.
    """

    instruction_content = _instruction_content(prescription)
    instruction = db.scalar(select(PatientInstruction).where(PatientInstruction.prescription_id == prescription.id))
    if not instruction:
        instruction = PatientInstruction(prescription_id=prescription.id, content="", safety_statement="")
        db.add(instruction)
    instruction.content = instruction_content
    instruction.safety_statement = "本页转达医生已确认的信息；如实际药袋标签与本页不同，请先联系开方医生或药师核对。"

    food_interactions = food_interactions_for_herbs(item.herb_name for item in prescription.items)
    herb_interactions = herb_incompatibilities_for_prescription(item.herb_name for item in prescription.items)
    evidence = _retrieved_evidence(food_interactions, herb_interactions)
    doctor_note = (prescription.doctor_notes or "").strip()
    packet = {
        "packet_id": f"patient_diet_prescription_{prescription.id}_{int(datetime.now(timezone.utc).timestamp())}",
        "role": "patient",
        "task_type": "patient_diet_guidance",
        "visibility": "patient_safe",
        "retrieved_evidence": evidence,
        "structured_context": {
            "prescription_id": prescription.id,
            "doctor_note": doctor_note,
            "usage_method": (prescription.usage_method or "").strip(),
            "prescribed_herbs": [item.herb_name for item in prescription.items],
            "food_interactions": food_interactions,
            "herb_interactions": herb_interactions,
        },
    }
    llm_result = LLMService(db).generate_patient_diet_guidance(
        packet,
        {
            "input_text": f"请通俗解释医生饮食备注：{doctor_note or '医生未填写饮食备注'}",
            "user_id": prescription.patient_id,
            "constraints": {
                "preserve_doctor_note": True,
                "no_new_interactions": True,
                "allowed_source_ids": sorted({item["source_id"] for item in evidence if item.get("source_id")}),
            },
        },
    )
    explanation = _llm_explanation(llm_result.structured_output)
    if not explanation:
        explanation = "医生未填写可供扩展的饮食备注；请按处方用法执行，如需确认具体食物，请咨询开方医生或药师。"

    source_ids = sorted(
        {
            *(item.get("source_id", "") for item in food_interactions),
            *(item.get("source_id", "") for item in herb_interactions),
        }
        - {""}
    )
    details = {
        "doctor_note": doctor_note,
        "doctor_usage_method": (prescription.usage_method or "").strip(),
        "item_notes": [
            {"herb_name": item.herb_name, "note": item.notes.strip()}
            for item in prescription.items
            if item.notes and item.notes.strip()
        ],
        "ai_explanation": explanation,
        "food_interactions": food_interactions,
        "herb_interactions": herb_interactions,
        "food_evidence_missing": not food_interactions,
        "source_ids": source_ids,
        "sources": [{"source_id": source_id, "url": SOURCE_URLS.get(source_id, "")} for source_id in source_ids],
        "llm": {
            "provider": llm_result.provider,
            "model": llm_result.model,
            "fallback_used": llm_result.metadata.fallback_used,
            "degraded": llm_result.metadata.degraded,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    diet = db.scalar(select(DietPrecaution).where(DietPrecaution.prescription_id == prescription.id))
    if not diet:
        diet = DietPrecaution(prescription_id=prescription.id, content="", details_json={})
        db.add(diet)
    diet.content = explanation
    diet.details_json = details
    db.flush()
    return details


def patient_guidance_payload(prescription: Prescription, diet: DietPrecaution | None) -> dict[str, Any]:
    details = dict(diet.details_json or {}) if diet else {}
    details.setdefault("doctor_note", (prescription.doctor_notes or "").strip())
    details.setdefault("doctor_usage_method", (prescription.usage_method or "").strip())
    details.setdefault(
        "item_notes",
        [
            {"herb_name": item.herb_name, "note": item.notes.strip()}
            for item in prescription.items
            if item.notes and item.notes.strip()
        ],
    )
    details.setdefault("ai_explanation", diet.content if diet else "")
    details.setdefault("food_interactions", food_interactions_for_herbs(item.herb_name for item in prescription.items))
    details.setdefault("herb_interactions", herb_incompatibilities_for_prescription(item.herb_name for item in prescription.items))
    details.setdefault("food_evidence_missing", not details["food_interactions"])
    details["content"] = diet.content if diet else details.get("ai_explanation", "")
    return details


def food_interactions_for_herbs(herb_names: Iterable[str]) -> list[dict[str, Any]]:
    present = {_normalize(name): str(name).strip() for name in herb_names if _normalize(name)}
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in _diet_records():
        for entity in record.get("related_entities", []):
            normalized = _normalize(entity)
            if normalized not in present:
                continue
            key = (str(record.get("entry_id", "")), normalized)
            if key in seen:
                continue
            seen.add(key)
            source_id = str(record.get("source_id") or "")
            matches.append(
                {
                    "herb_name": present[normalized],
                    "food_items": [str(value) for value in record.get("food_items", []) if value],
                    "food_categories": [str(value) for value in record.get("food_categories", []) if value],
                    "relation_label": str(record.get("advice_type") or "传统服药食忌参考"),
                    "message": str(record.get("message") or ""),
                    "source_id": source_id,
                    "source_record_id": str(record.get("source_record_id") or ""),
                    "source_url": SOURCE_URLS.get(source_id, ""),
                }
            )
    return sorted(matches, key=lambda item: (item["herb_name"], "、".join(item["food_items"])))


def herb_incompatibilities_for_prescription(herb_names: Iterable[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_name in herb_names:
        prescribed = str(raw_name or "").strip()
        if not prescribed:
            continue
        for item in incompatibilities_for_herb(prescribed):
            partner = str(item.get("herb_name") or "").strip()
            category = _display_incompatibility_label(str(item.get("category") or "传统配伍禁忌"))
            key = (_normalize(prescribed), _normalize(partner), category)
            if not partner or key in seen:
                continue
            seen.add(key)
            source_id = str(item.get("source_id") or "")
            results.append(
                {
                    "herb_name": prescribed,
                    "incompatible_with": partner,
                    "relation_label": category,
                    "source_id": source_id,
                    "source_record_id": str(item.get("source_record_id") or ""),
                    "source_url": SOURCE_URLS.get(source_id, ""),
                }
            )
    return sorted(results, key=lambda item: (item["herb_name"], item["relation_label"], item["incompatible_with"]))


def _instruction_content(prescription: Prescription) -> str:
    usage = (prescription.usage_method or "").strip() or "医生尚未填写处方级用法，请先联系医生或药师确认后再服用。"
    doctor_note = (prescription.doctor_notes or "").strip()
    lines = [f"医生确认的处方用法：{usage}"]
    if doctor_note:
        lines.append(f"医生备注原文：{doctor_note}")
    item_notes = [f"{item.herb_name}：{item.notes.strip()}" for item in prescription.items if item.notes and item.notes.strip()]
    if item_notes:
        lines.append("单味药备注：" + "；".join(item_notes))
    lines.append("出现不适、过敏或症状加重时，请停止自行判断并及时联系医生或药师。")
    return "\n".join(lines)


def _retrieved_evidence(food_items: list[dict[str, Any]], herb_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in food_items:
        evidence.append(
            {
                "evidence_type": "traditional_diet_precaution",
                "title": f"{item['herb_name']}—{'、'.join(item['food_items'])}",
                "snippet": item["message"],
                "source_id": item["source_id"],
                "source_record_id": item["source_record_id"],
            }
        )
    for item in herb_items:
        evidence.append(
            {
                "evidence_type": "traditional_herb_incompatibility",
                "title": f"{item['herb_name']}—{item['incompatible_with']}",
                "snippet": f"{item['relation_label']}：{item['herb_name']}与{item['incompatible_with']}。",
                "source_id": item["source_id"],
                "source_record_id": item["source_record_id"],
            }
        )
    return evidence


def _llm_explanation(structured: dict[str, Any]) -> str:
    values: list[str] = []
    summary = str(structured.get("summary") or "").strip()
    if summary:
        values.append(summary)
    for section in structured.get("sections", []) if isinstance(structured.get("sections"), list) else []:
        if not isinstance(section, dict):
            continue
        content = str(section.get("content") or "").strip()
        if content and content not in values:
            values.append(content)
    return "\n\n".join(values)


@lru_cache(maxsize=4)
def _load_diet_records(path: str, _mtime_ns: int, _size: int) -> tuple[dict[str, Any], ...]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return tuple(item for item in payload.get("items", []) if isinstance(item, dict))


def _diet_records() -> tuple[dict[str, Any], ...]:
    stat = DIET_PRECAUTIONS_FILE.stat()
    return _load_diet_records(str(DIET_PRECAUTIONS_FILE.resolve()), stat.st_mtime_ns, stat.st_size)


def _normalize(value: object) -> str:
    return _SPACE_RE.sub("", str(value or "")).strip().lower()


def _display_incompatibility_label(value: str) -> str:
    normalized = value.strip()
    if normalized in {"十八反", "十八反（传统配伍禁忌）"}:
        return "十八反（传统配伍禁忌）"
    if normalized in {"十九畏", "十九畏（传统配伍禁忌）"}:
        return "十九畏（传统配伍禁忌）"
    return normalized or "传统配伍禁忌"
