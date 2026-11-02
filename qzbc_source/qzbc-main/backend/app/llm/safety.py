from __future__ import annotations

import json
import re

from app.llm.schemas import LLMRequest, LLMResult


PATIENT_FORBIDDEN = (
    "建议服用",
    "可以加量",
    "可以停药",
    "替换成",
    "我判断你是",
    "你属于",
    "处方如下",
    "食疗方治疗",
    "建议你用",
    "建议你吃",
    "你可以服用",
    "你应该服用",
    "适合你服用",
    "按此方服用",
)
PATIENT_TREATMENT_CLAIMS = ("常用于治疗", "可以治疗", "能够治疗", "能够缓解", "可缓解", "治愈", "缓解症状")
_PATIENT_TREATMENT_CLAIM_PATTERN = re.compile(
    r"[^。；;！!？?\n]*(?:"
    r"常用于治疗|可以治疗|能够治疗|能够缓解|可缓解|治愈|"
    r"(?:常用于|多用于|可用于|有助于|可帮助|能帮助)[^。；;！!？?\n]*(?:咳嗽|失眠|疼痛|胃痛|腹痛|头痛|疾病|病症|症状)|"
    r"(?:缓解|改善)[^。；;！!？?\n]*(?:咳嗽|失眠|疼痛|胃痛|腹痛|头痛|疾病|病症|症状)"
    r")[^。；;！!？?\n]*(?:[。；;！!？?]|$)"
)
_PATIENT_MEDICATION_RECOMMENDATION_PATTERN = re.compile(
    r"[^。；;！!？?\n]*(?:可以考虑|建议|可以|应该|可).{0,12}(?:使用|服用|选用|吃).{0,20}"
    r"(?:中药|药材|方剂|丸|散|膏|汤)[^。；;！!？?\n]*(?:[。；;！!？?]|$)"
)
DOCTOR_FORBIDDEN = ("无需医生确认", "可直接发布", "保证有效", "已为患者确定最终处方", "已确定最终处方")
STUDENT_FORBIDDEN = ("手机号", "身份证", "真实姓名", "精确就诊时间", "真实患者 ID", "医生内部患者列表")


def apply_output_safety(request: LLMRequest, result: LLMResult) -> LLMResult:
    content = result.content + " " + str(result.structured_output)
    flags: list[str] = []
    if request.role == "patient":
        forbidden_flags = [f"patient_forbidden:{token}" for token in PATIENT_FORBIDDEN if token in content]
        if _PATIENT_MEDICATION_RECOMMENDATION_PATTERN.search(content):
            forbidden_flags.append("patient_forbidden:personal_medication_recommendation")
        treatment_flags = ["patient_treatment_claim_sanitized"] if _PATIENT_TREATMENT_CLAIM_PATTERN.search(content) else []
        flags = forbidden_flags + treatment_flags
        if forbidden_flags:
            result.content = "我可以提供一般中医药科普，但不能把药材直接作为疾病治疗结论，也不能为你诊断、开方、推荐服用、改药、加量或停药。涉及个人症状或正在使用的药物，请咨询医生或药师。"
            result.structured_output["summary"] = result.content
            result.structured_output["sections"] = []
            result.structured_output["warnings"] = ["患者端生成内容已被 Safety Guard 降级为安全回复。"]
            result.structured_output["next_actions"] = ["如需了解个体情况，请咨询医生或药师。"]
        elif treatment_flags:
            result.structured_output = _sanitize_patient_treatment_claims(result.structured_output)
            result.content = json.dumps(result.structured_output, ensure_ascii=False)
    elif request.role == "doctor":
        grounded_review = _doctor_review_grounded_response(request.evidence_packet) if request.task_type == "doctor_review_text" else None
        if grounded_review is not None:
            result.structured_output = grounded_review
            result.content = json.dumps(grounded_review, ensure_ascii=False)
            result.safety_flags.append("doctor_review_structured_grounding")
            content = result.content
        flags = [f"doctor_forbidden:{token}" for token in DOCTOR_FORBIDDEN if token in content]
        if flags:
            result.content = (
                result.content.replace("无需医生确认", "需要医生确认")
                .replace("可直接发布", "确认后再单独发布")
                .replace("保证有效", "仍需医生结合临床复核")
                .replace("已为患者确定最终处方", "仅提供辅助审方意见，最终处方由医生确认")
                .replace("已确定最终处方", "尚需医生确认最终处方")
            )
            result.structured_output["summary"] = result.content
            result.structured_output.setdefault("warnings", []).append("医生端输出必须经过医生确认。")
    elif request.role == "student":
        # MTCMB quiz explanations are an explicit teaching task: their input is
        # already limited to the question, options and answer key.  Replacing
        # that response with the old one-line source echo made a successful
        # Ollama call look like it had not answered at all.
        grounded_training = None
        if request.constraints.get("student_training_reference_only"):
            # A clinical-training card may show a source reference formula, but
            # it must never let a model turn that into a treatment claim for the
            # deidentified symptoms. Keep Ollama as an optional enhancer in
            # other student tasks; here a structured reference is authoritative.
            grounded_training = student_case_reference_response(request.input_text)
        elif request.task_type != "student_quiz_explanation":
            grounded_training = student_training_reference_response(request.evidence_packet)
        if grounded_training is not None:
            result.structured_output = grounded_training
            result.content = json.dumps(grounded_training, ensure_ascii=False)
            result.safety_flags.append("student_training_qa_grounded")
        elif _packet_has_source(request.evidence_packet, "SRC_SHENNONG_TCM_DATASET"):
            result.structured_output.setdefault("warnings", []).append(
                "ShenNong 问答属于生成式教学参考，可能存在术语或事实偏差，请对照教材和已复核知识。"
            )
            result.safety_flags.append("student_generated_qa_reference")
        flags = [f"student_forbidden:{token}" for token in STUDENT_FORBIDDEN if token in content]
        if flags:
            result.content = "教学内容包含疑似未脱敏信息，已被安全策略拦截。"
            result.structured_output["summary"] = result.content
            result.structured_output["warnings"] = ["学习端输出不得包含真实患者身份。"]
    if flags:
        result.safety_flags.extend(flags)
        if request.role != "patient" or any(flag.startswith("patient_forbidden") for flag in flags):
            result.metadata.degraded = True
    return result


def student_training_reference_response(evidence_packet: dict) -> dict | None:
    evidence = evidence_packet.get("retrieved_evidence") if isinstance(evidence_packet, dict) else []
    items = [item for item in evidence if isinstance(item, dict)]
    if not items or any(str(item.get("source_id") or "") != "SRC_MTCMB" for item in items):
        return None
    references = []
    for item in items[:3]:
        snippet = str(item.get("snippet") or "").strip()
        if snippet and snippet not in references:
            references.append(snippet)
    if not references:
        return None
    first_answer = _reference_answer(references[0])
    return {
        "title": "题库教学参考",
        "summary": f"根据 MTCMB 题库记录，标准答案为：{first_answer}",
        "sections": [
            {
                "heading": "题库依据",
                "content": "\n\n".join(references),
                "source_ids": ["SRC_MTCMB"],
            },
            {
                "heading": "使用边界",
                "content": "这里只复述题库答案。题库未提供的药性、证候、剂量和临床依据不作补充，需要结合教材或已复核知识进一步辨析。",
                "source_ids": ["SRC_MTCMB"],
            },
        ],
        "warnings": ["MTCMB 内容仅用于学习与评测，不作为临床诊疗证据。"],
        "next_actions": ["对照教材核对该题涉及的药材功效与鉴别要点。"],
        "disclaimer": "仅用于中医药教学，不作为临床诊疗依据。",
    }


def student_case_reference_response(input_text: str) -> dict:
    """Make a clinical-training reference explicit without inferring efficacy."""
    text = str(input_text or "")
    facts = text.split("请只解释", 1)[0].strip() or "脱敏案例字段未提供"
    formula_match = re.search(r"参考方药=([^。；;]+)", facts)
    formula = formula_match.group(1).strip() if formula_match else "原始来源未提供"
    return {
        "title": "脱敏病例教学参考",
        "summary": f"本案例的结构化参考方药为：{formula}。这是教学对照信息，不能据此推断药材正在治疗该病例。",
        "sections": [
            {"heading": "脱敏病例事实", "content": facts, "source_ids": []},
            {"heading": "教学对照", "content": "学生应以标准解析和结构化评分中的一致、缺失、额外药材为准；相同或相近功效不代表可以替代参考药材。", "source_ids": []},
            {"heading": "使用边界", "content": "不得把参考方药写成对该脱敏病例的治疗结论，也不得生成剂量、用法或临床处方建议。", "source_ids": []},
        ],
        "warnings": ["模型生成内容已被教学结构化参考约束，评分由确定性对照决定。"],
        "next_actions": ["核对标准解析", "根据缺失和额外药材完成复盘"],
        "disclaimer": "仅用于脱敏中医药教学，不作为临床诊疗依据。",
    }


def sanitize_stream_segment(role: str, text: str) -> str:
    """Sanitize complete sentences before they are exposed over SSE."""
    if role == "patient":
        if any(token in text for token in PATIENT_FORBIDDEN) or _PATIENT_MEDICATION_RECOMMENDATION_PATTERN.search(text):
            return "涉及个人诊疗或用药决定时，请咨询医生或药师；这里仅提供一般中医药科普。"
        return _PATIENT_TREATMENT_CLAIM_PATTERN.sub("相关传统应用信息不能作为个人治疗结论。", text)
    if role == "doctor":
        return (
            text.replace("无需医生确认", "需要医生确认")
            .replace("可直接发布", "确认后再单独发布")
            .replace("保证有效", "仍需医生结合临床复核")
        )
    if role == "student" and any(token in text for token in STUDENT_FORBIDDEN):
        return "疑似包含未脱敏信息的内容已被教学端安全策略拦截。"
    return text


def _sanitize_patient_treatment_claims(value):
    if isinstance(value, str):
        return _PATIENT_TREATMENT_CLAIM_PATTERN.sub("相关传统应用信息不能作为个人治疗结论。", value)
    if isinstance(value, list):
        return [_sanitize_patient_treatment_claims(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_patient_treatment_claims(item) for key, item in value.items()}
    return value


def _reference_answer(snippet: str) -> str:
    marker = "参考答案："
    if marker in snippet:
        answer = snippet.split(marker, 1)[1].splitlines()[0].strip()
        return answer or "请查看题库原始记录"
    return "请查看题库原始记录"


def _packet_has_source(packet: dict, source_id: str) -> bool:
    evidence = packet.get("retrieved_evidence") if isinstance(packet, dict) else []
    return any(isinstance(item, dict) and item.get("source_id") == source_id for item in evidence)


def _doctor_review_grounded_response(packet: dict) -> dict | None:
    context = packet.get("structured_context") if isinstance(packet.get("structured_context"), dict) else {}
    prescription = context.get("prescription") if isinstance(context.get("prescription"), dict) else {}
    items = prescription.get("items") if isinstance(prescription.get("items"), list) else []
    if not items:
        return None

    item_lines = []
    for item in items:
        if not isinstance(item, dict) or not item.get("herb_name"):
            continue
        dose = " ".join(str(value).strip() for value in (item.get("dose"), item.get("unit")) if value)
        item_lines.append(f"{item['herb_name']} {dose}".strip())
    rule_review = context.get("rule_review") if isinstance(context.get("rule_review"), dict) else {}
    risk_level = str(rule_review.get("risk_level") or "red")
    review_summary = str(rule_review.get("review_summary") or "结构化规则结果缺失，必须由医生人工复核。")
    findings = packet.get("rule_findings") if isinstance(packet.get("rule_findings"), list) else []
    source_ids = sorted(
        {str(item.get("source_id")) for item in findings if isinstance(item, dict) and item.get("source_id")}
    )
    finding_lines = [
        f"[{str(item.get('risk_level') or '').upper()}] {item.get('message')}"
        for item in findings
        if isinstance(item, dict) and item.get("message")
    ]
    allergy_summary = str(context.get("patient_allergy_summary") or "未记录")
    matched = [str(value) for value in context.get("allergy_matched_herbs") or [] if value]
    incompatibility_matches = [
        item for item in context.get("herb_incompatibility_matches") or [] if isinstance(item, dict)
    ]
    incompatibility_text = "；".join(
        f"{item.get('herb_a')}—{item.get('herb_b')}（{item.get('category')}）"
        for item in incompatibility_matches
        if item.get("herb_a") and item.get("herb_b")
    )
    warnings = [str(item.get("message")) for item in findings if isinstance(item, dict) and item.get("message")]
    if not findings:
        warnings.append("结构化规则未触发风险不等于处方绝对安全，仍需医生人工确认。")
    return {
        "title": f"处方{prescription.get('prescription_id') or ''}结构化审查说明",
        "summary": f"风险等级：{risk_level}。{review_summary}",
        "sections": [
            {"heading": "本次处方组成", "content": "、".join(item_lines), "source_ids": []},
            {
                "heading": "患者过敏史核对",
                "content": f"已记录：{allergy_summary}。" + (f"匹配冲突：{'、'.join(matched)}。" if matched else "本次未匹配到同名药材冲突。"),
                "source_ids": source_ids,
            },
            {
                "heading": "结构化规则结果",
                "content": "\n".join(finding_lines) if finding_lines else "本次没有规则卡片；仍须医生结合完整病史复核。",
                "source_ids": source_ids,
            },
            {
                "heading": "药材配伍禁忌核对",
                "content": (
                    f"匹配到：{incompatibility_text}。已触发红色预警，必须由中药师或医生复核。"
                    if incompatibility_text
                    else "本次未匹配到已收录的十八反、十九畏药对；未命中不等于不存在其他相互作用。"
                ),
                "source_ids": ["SRC_DB4403_T142_2021"] if incompatibility_text else [],
            },
        ],
        "warnings": warnings,
        "next_actions": ["核对处方组成、剂量、处方级用法、患者过敏史和药材配伍禁忌。", "修改处方或患者资料后必须重新审方。"],
        "disclaimer": "规则引擎负责风险触发，生成层只负责展示；最终处方必须由医生确认。",
    }
