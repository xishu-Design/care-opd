"""Deterministic answers for a doctor's own, authorized business records.

These answers are not medical knowledge generation.  A model must never infer a
patient's prescription from a formula it happens to know, so questions about a
named bound patient's current prescription bypass the generative provider.
"""

from __future__ import annotations

from typing import Any

from app.llm.schemas import GeneratedResponse, GeneratedSection


_PRESCRIPTION_QUERY_TERMS = ("处方", "开的方", "开方", "开的药", "药材", "组成", "服法", "怎么服", "医嘱", "备注")
_STATUS_LABELS = {
    "draft": "草稿",
    "reviewing": "审方中",
    "doctor_confirmed": "医生已确认，待发布",
    "published_to_patient": "已发布给患者",
    "deidentified_for_learning": "已脱敏用于教学",
}


def doctor_direct_response(input_text: str, evidence_packet: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """Resolve a named patient's prescription from the authorized roster only."""
    context = evidence_packet.get("structured_context") if isinstance(evidence_packet, dict) else {}
    roster = context.get("doctor_patient_roster") if isinstance(context, dict) else []
    if not isinstance(roster, list):
        return None

    resolution_text = _resolution_text(input_text, evidence_packet)
    patient = _named_patient(resolution_text, roster)
    is_follow_up = _looks_like_prescription_follow_up(input_text)
    if not _asks_about_prescription(input_text) and not (patient is not None and is_follow_up):
        return None
    if patient is None:
        return None
    return _prescription_response(patient).model_dump(mode="json"), "doctor_bound_patient_prescription"


def _asks_about_prescription(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return any(term in compact for term in _PRESCRIPTION_QUERY_TERMS)


def _looks_like_prescription_follow_up(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return any(term in compact for term in ("它", "这个", "上述", "上一个", "前一条", "那张", "那份", "注意事项", "用法", "备注"))


def _resolution_text(input_text: str, evidence_packet: dict[str, Any]) -> str:
    """Use prior *user questions* only; past model answers are never facts."""
    packet_context = evidence_packet.get("conversation_context") if isinstance(evidence_packet, dict) else {}
    turns = packet_context.get("recent_turns") if isinstance(packet_context, dict) else []
    previous_questions = [
        str(turn.get("question") or "").strip()
        for turn in turns[-2:]
        if isinstance(turn, dict) and str(turn.get("question") or "").strip()
    ]
    return "\n".join([*previous_questions, str(input_text or "")])


def _named_patient(text: str, roster: list[dict[str, Any]]) -> dict[str, Any] | None:
    compact = "".join(str(text or "").split())
    matches: list[dict[str, Any]] = []
    for patient in roster:
        if not isinstance(patient, dict):
            continue
        names = (str(patient.get("display_name") or "").strip(), str(patient.get("username") or "").strip())
        if any(name and "".join(name.split()) in compact for name in names):
            matches.append(patient)
    return matches[0] if len(matches) == 1 else None


def _prescription_response(patient: dict[str, Any]) -> GeneratedResponse:
    display_name = str(patient.get("display_name") or patient.get("username") or "该患者").strip()
    prescription = patient.get("latest_prescription") if isinstance(patient.get("latest_prescription"), dict) else {}
    if not prescription:
        return GeneratedResponse(
            title="患者处方查询",
            summary=f"系统未查到你为{display_name}建立的处方记录。",
            warnings=["只查询当前登录医生的有效绑定患者；不会根据通用方剂知识推测患者处方。"],
            next_actions=["确认是否已为该患者创建处方", "如处方由其他医生开立，请在对应诊疗记录中核对"],
            disclaimer="以上结果来自当前医生账号的授权处方记录。",
        )

    items = prescription.get("items") if isinstance(prescription.get("items"), list) else []
    herb_text = "、".join(
        f"{item.get('herb_name')}{item.get('dose') or ''}{item.get('unit') or ''}"
        for item in items
        if isinstance(item, dict) and str(item.get("herb_name") or "").strip()
    ) or "处方未录入药材明细"
    status = str(prescription.get("status") or "").strip()
    status_text = _STATUS_LABELS.get(status, status or "未记录")
    title = str(prescription.get("title") or "未命名处方").strip()
    usage = str(prescription.get("usage_method") or "未填写").strip()
    note = str(prescription.get("doctor_notes") or "未填写").strip()

    return GeneratedResponse(
        title="患者处方查询",
        summary=f"你为{display_name}最近建立的处方是《{title}》，当前状态：{status_text}。以下内容直接来自该处方记录，不按经典方剂名称推断。",
        sections=[
            GeneratedSection(heading="处方药材", content=herb_text, source_ids=[]),
            GeneratedSection(heading="处方用法", content=usage, source_ids=[]),
            GeneratedSection(heading="医生备注", content=note, source_ids=[]),
        ],
        warnings=["这是处方记录转述；药材组成、用法和备注应以处方详情页为最终核对依据。"],
        next_actions=["如需改动处方，请在医生端编辑后重新审方", "发布给患者前核对药材、剂量、用法和过敏风险"],
        disclaimer="以上结果来自当前登录医生的授权处方记录，不构成自动辨证或自动改方。",
    )
