from __future__ import annotations

import re
from typing import Any


_PERSONAL_MARKERS = ("我", "我的", "本人", "自己", "患者")
_SYMPTOM_TERMS = (
    "疼", "痛", "疲劳", "头昏", "头晕", "眩晕", "咳", "失眠", "发热", "发烧", "恶心", "呕吐",
    "腹泻", "便秘", "乏力", "胸闷", "心慌", "眼干", "眼涩", "视物", "不舒服", "症状",
)
_ADVICE_TERMS = ("怎么办", "怎么处理", "怎么调理", "怎么改善", "吃什么", "饮食", "注意什么", "需要注意", "从中医")
_MEDICINE_SUFFIX_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,12}(?:膏|丸|散|汤|方|颗粒|胶囊|片|口服液|药)$")
_MEDICINE_INFO_TERMS = ("是什么", "干什么", "干啥", "作用", "功效", "组成", "成分", "怎么用", "用途")


def patient_direct_response(input_text: str, evidence_packet: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """Return a deterministic patient-safe answer when generation would be unsafe.

    Recent assistant messages are dialogue context, never evidence. This policy therefore
    resolves follow-ups with the previous *question* only and does not inherit medical
    claims or medicine names from a prior answer.
    """
    resolved = _resolved_patient_query(input_text, evidence_packet)
    if _is_personal_symptom_request(resolved):
        response = _symptom_response(resolved)
        return response, "patient_personal_symptom_safety"
    if not _has_verified_evidence(evidence_packet) and _looks_like_medicine_information_request(resolved):
        response = {
            "title": "未检索到可核验的药品或方剂记录",
            "summary": (
                "当前患者安全知识库没有检索到与该名称对应的可靠记录。我不能仅凭名称补全它的用途、功效或组成；"
                "该名称也可能存在错别字，或属于商品名、院内制剂、地方制剂。请核对包装上的完整名称、批准文号和说明书，"
                "再向医生或药师确认。在名称和来源核实前，不要自行购买或使用。"
            ),
            "sections": [],
            "warnings": ["无可追溯证据时，系统不会根据模型记忆猜测药物或方剂信息。"],
            "next_actions": ["核对包装全名、批准文号和说明书。", "将核实后的信息交给医生或药师确认。"],
            "disclaimer": "仅供一般健康科普，不能替代医生诊疗或药师用药指导。",
        }
        return response, "patient_ungrounded_medicine_information"
    return None


def response_display_text(response: dict[str, Any]) -> str:
    parts = [str(response.get("summary") or "").strip()]
    sections = response.get("sections") if isinstance(response.get("sections"), list) else []
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading") or "").strip()
        content = str(section.get("content") or "").strip()
        if heading and content:
            parts.append(f"{heading}\n{content}")
    disclaimer = str(response.get("disclaimer") or "").strip()
    if disclaimer:
        parts.append(disclaimer)
    return "\n\n".join(part for part in parts if part)


def _resolved_patient_query(input_text: str, packet: dict[str, Any]) -> str:
    current = input_text.strip()
    context = packet.get("conversation_context") if isinstance(packet.get("conversation_context"), dict) else {}
    if not context.get("follow_up_detected"):
        return current
    turns = context.get("recent_turns") if isinstance(context.get("recent_turns"), list) else []
    previous_question = ""
    if turns and isinstance(turns[-1], dict):
        previous_question = str(turns[-1].get("question") or "").strip()
    return "\n".join(part for part in (previous_question, current) if part)


def _is_personal_symptom_request(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return (
        any(marker in compact for marker in _PERSONAL_MARKERS)
        and any(term in compact for term in _SYMPTOM_TERMS)
        and any(term in compact for term in _ADVICE_TERMS)
    )


def _symptom_response(text: str) -> dict[str, Any]:
    eye_related = any(term in text for term in ("眼", "视力", "视物"))
    dizzy_related = any(term in text for term in ("头昏", "头晕", "眩晕"))
    if eye_related:
        summary = (
            "眼痛或眼疲劳不能仅凭聊天判断病因，也不能直接等同于中医的“肝火”，更不能解释为现代医学中的肝功能异常。"
            "如果与长时间近距离用眼有关，可以先停止用眼并休息、眨眼、远眺；不要揉眼，也不要自行使用中药或眼药。"
        )
        if dizzy_related:
            summary += "同时头昏时应先坐下或躺下、补充水分、缓慢起身，在症状消失前不要驾驶或操作危险设备。"
        self_care = (
            "饮食保持规律、均衡并适量饮水即可；没有证据表明某一种蔬菜、水果或所谓“清肝”食物可以治疗这组症状，"
            "也不需要根据“肝火”自行大范围忌口。"
        )
        urgent = (
            "若眼痛突然或明显加重，伴视力下降或复视、明显红眼、畏光、眼外伤，或伴剧烈头痛、恶心呕吐，应尽快就医。"
            "若头昏反复不缓解，或伴说话困难、肢体无力或麻木、晕厥，也应及时就医。"
        )
        tcm = (
            "从中医角度，证候判断需要结合症状持续时间、诱因以及舌脉等信息；仅凭“眼痛、疲劳、头昏”不能确定证候，"
            "因此不能据此推荐“清肝明目”类中药。"
        )
    else:
        summary = "仅凭聊天中的个人症状不能可靠判断病因或中医证候，也不适合据此推荐药材或方剂。"
        self_care = "先记录症状出现时间、诱因、持续时长和伴随表现，保持规律饮食与休息；症状持续、反复或加重时请就医。"
        urgent = "如出现突发剧痛、意识异常、呼吸困难、晕厥、进行性无力或其他明显异常，请及时就医。"
        tcm = "中医辨证同样需要完整问诊及舌脉等信息，不能用一两个症状直接套用证候或中药。"
    return {
        "title": "个人症状的安全处理建议",
        "summary": summary,
        "sections": [
            {"heading": "现在可以做什么", "content": self_care, "source_ids": []},
            {"heading": "何时需要就医", "content": urgent, "source_ids": []},
            {"heading": "中医角度的边界", "content": tcm, "source_ids": []},
        ],
        "warnings": ["不要根据聊天回答自行诊断、辨证或使用中药。"],
        "next_actions": ["若症状持续或反复，优先到眼科或综合门诊评估。"] if eye_related else ["症状持续、反复或加重时就医。"],
        "disclaimer": "仅供一般健康科普，不能替代医生诊疗。",
    }


def _has_verified_evidence(packet: dict[str, Any]) -> bool:
    structured = packet.get("structured_context") if isinstance(packet, dict) else {}
    if isinstance(structured, dict) and (
        structured.get("published_prescriptions") or structured.get("patient_self")
    ):
        return True
    evidence = packet.get("retrieved_evidence") if isinstance(packet, dict) else []
    if not isinstance(evidence, list):
        return False
    return any(isinstance(item, dict) and item.get("source_id") for item in evidence)


def _looks_like_medicine_information_request(text: str) -> bool:
    compact = re.sub(r"[\s？?，,。.!！]", "", text)
    if not any(term in compact for term in _MEDICINE_INFO_TERMS):
        return False
    candidates = re.findall(r"[\u4e00-\u9fff]{2,16}(?:膏|丸|散|汤|方|颗粒|胶囊|片|口服液|药)", compact)
    return bool(candidates) or bool(_MEDICINE_SUFFIX_PATTERN.search(compact))
