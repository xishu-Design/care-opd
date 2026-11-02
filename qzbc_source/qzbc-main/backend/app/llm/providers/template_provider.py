from __future__ import annotations

from time import perf_counter
from typing import Any

from app.llm.fallback import compact_text, evidence_count, extract_source_ids
from app.llm.providers.base import BaseLLMProvider
from app.llm.schemas import GeneratedResponse, GeneratedSection, LLMGrounding, LLMMetadata, LLMRequest, LLMResult, ProviderHealth, generated_to_dict


class TemplateProvider(BaseLLMProvider):
    name = "template"
    model = "guardtcm-template-v1"

    def health(self) -> ProviderHealth:
        return ProviderHealth(name=self.name, configured=True, reachable=True, status="ok", latency_ms=0, model=self.model)

    def generate(self, request: LLMRequest) -> LLMResult:
        started = perf_counter()
        source_ids = extract_source_ids(request.evidence_packet)
        business_context_count = _business_context_count(request.evidence_packet)
        generated = self._generate_structured(request, source_ids)
        content = generated.summary or "\n".join(section.content for section in generated.sections)
        return LLMResult(
            request_id=request.request_id,
            success=True,
            provider=self.name,
            model=self.model,
            content=content,
            structured_output=generated_to_dict(generated),
            grounding=LLMGrounding(
                used_evidence_count=evidence_count(request.evidence_packet) + business_context_count,
                missing_evidence=evidence_count(request.evidence_packet) == 0 and business_context_count == 0,
                source_ids=source_ids,
            ),
            metadata=LLMMetadata(latency_ms=int((perf_counter() - started) * 1000)),
        )

    def _generate_structured(self, request: LLMRequest, source_ids: list[str]) -> GeneratedResponse:
        packet = request.evidence_packet
        rule_findings = packet.get("rule_findings", []) or []
        retrieved = packet.get("retrieved_evidence", []) or []
        if request.task_type == "doctor_review_text":
            return GeneratedResponse(
                title="医生审方辅助说明",
                summary="基于 Evidence Packet 汇总审方风险、证据和医生确认要点；最终处方仍需医生结合实际情况确认。",
                sections=[
                    GeneratedSection(
                        heading="风险解释",
                        content=compact_text(rule_findings or "未发现结构化规则返回的高风险项，仍建议医生人工复核。"),
                        source_ids=source_ids,
                    ),
                    GeneratedSection(
                        heading="证据说明",
                        content=compact_text(retrieved or "当前证据不足，请结合患者情况和处方信息判断。"),
                        source_ids=source_ids,
                    ),
                ],
                warnings=["LLM 只负责表达，不负责风险判断。", "不要绕过 HumanConfirm 或自动发布给患者。"],
                next_actions=["医生复核风险卡片", "必要时修改处方后重新审方", "确认后再单独发布给患者"],
                disclaimer="本内容仅供医生参考，最终处方需由医生结合实际情况判断。",
            )
        if request.task_type == "student_analysis":
            return GeneratedResponse(
                title="脱敏案例标准解析",
                summary="基于 student_learning Evidence Packet 生成教学解析、风险点和复习建议。",
                sections=[
                    GeneratedSection(heading="风险点讲解", content=compact_text(rule_findings or packet.get("case_summary", "")), source_ids=source_ids),
                    GeneratedSection(heading="知识卡", content="请围绕证据来源、风险等级和审方理由进行复习。", source_ids=source_ids),
                ],
                warnings=["不得还原真实患者身份。"],
                next_actions=["复习风险规则", "完成练习题", "对照标准解析修正审方思路"],
                disclaimer="本内容仅用于教学训练，不作为临床诊疗依据。",
            )
        if request.task_type == "student_quiz_explanation":
            question = str(request.constraints.get("question") or request.input_text or "题库未提供题干")
            answer = str(request.constraints.get("answer") or "题库未提供标准答案")
            options = request.constraints.get("options") if isinstance(request.constraints.get("options"), list) else []
            option_text = "；".join(str(item) for item in options if str(item).strip()) or "原始题库未提供选项"
            return GeneratedResponse(
                title="题库教学讲解",
                summary=f"本题标准答案为：{answer}。请先将答案与原题选项逐项核对。",
                sections=[
                    GeneratedSection(heading="题干与选项", content=f"题干：{question}\n选项：{option_text}", source_ids=source_ids),
                    GeneratedSection(heading="复习方法", content="从六淫的性质、易伤脏腑和常见临床表现三个层次复习；本模板未新增任何题库外结论。", source_ids=source_ids),
                ],
                warnings=["当前为本地模板讲解；如需更详细的理论说明，请确认 Ollama 服务和模型可用后重新生成。"],
                next_actions=["核对标准答案", "回到测验库完成下一题"],
                disclaimer="本讲解仅用于中医药教学，不作为临床诊疗依据。",
            )
        if request.task_type == "doctor_ai_chat":
            context = packet.get("structured_context") if isinstance(packet.get("structured_context"), dict) else {}
            roster = context.get("doctor_patient_roster") if isinstance(context.get("doctor_patient_roster"), list) else []
            if _asks_about_patient_roster(request.input_text) or _mentions_roster_patient(request.input_text, roster):
                return _doctor_roster_response(roster)
            return GeneratedResponse(
                title="医生专业问答",
                summary="已根据当前可追溯知识证据整理专业回答；证据不足的部分需由医生结合临床信息进一步核实。",
                sections=[
                    GeneratedSection(heading="证据检索结果", content=_retrieved_evidence_text(retrieved, "当前未检索到直接匹配的规范知识条目。"), source_ids=source_ids),
                    GeneratedSection(heading="专业判断边界", content="本回答用于知识检索和临床决策辅助，不能替代辨证、审方与人工确认。", source_ids=source_ids),
                ],
                warnings=["不得将证据不足的内容直接用于处方。"],
                next_actions=["核对证候与治法", "复核方剂组成和禁忌", "结合患者实际情况人工确认"],
                disclaimer="本内容仅供医生专业参考，最终诊疗决策由医生负责。",
            )
        if request.task_type == "student_ai_chat":
            return GeneratedResponse(
                title="中医药教学问答",
                summary="已根据脱敏教学证据整理概念解释和复习思路。",
                sections=[
                    GeneratedSection(heading="概念解释", content=_retrieved_evidence_text(retrieved, "当前未检索到直接匹配的教学知识条目。"), source_ids=source_ids),
                    GeneratedSection(heading="辨析思路", content="建议按证候、治法、方剂和药材四个层次梳理知识关系。", source_ids=source_ids),
                ],
                warnings=["教学内容不能直接作为临床诊疗建议。"],
                next_actions=["用自己的话复述概念", "对照知识图谱梳理关系", "完成一个相似问题的辨析"],
                disclaimer="本内容仅用于中医药教学训练，不作为临床诊疗依据。",
            )
        if request.task_type == "quiz_generation":
            return GeneratedResponse(
                title="审方练习题",
                summary="根据教学版证据生成模板练习题。",
                sections=[
                    GeneratedSection(heading="判断题", content="该案例的风险判断应基于 Evidence Packet，而不是模型自由推断。", source_ids=source_ids),
                    GeneratedSection(heading="选择题", content="以下哪项最符合本案例的审方复核重点？A. 风险卡片 B. 患者真实身份 C. 无依据改方", source_ids=source_ids),
                ],
                disclaimer="本内容仅用于教学训练，不作为临床诊疗依据。",
            )
        if request.task_type == "patient_diet_guidance":
            context = packet.get("structured_context") if isinstance(packet.get("structured_context"), dict) else {}
            doctor_note = str(context.get("doctor_note") or "医生未填写饮食备注").strip()
            return GeneratedResponse(
                title="处方饮食注意说明",
                summary=f"医生备注原文：{doctor_note}",
                sections=[
                    GeneratedSection(
                        heading="通俗说明",
                        content="请以医生原话为准，饮食安排保持简单、规律；不清楚具体食物是否适合时，请先询问开方医生或药师。",
                        source_ids=[],
                    )
                ],
                warnings=["模板降级未补充新的药食相互作用结论。"],
                next_actions=["按医生已确认的备注执行", "对未收录的具体食物先咨询医生或药师"],
                disclaimer="本说明用于解释医生已确认的处方备注，不替代医生或药师的个体化指导。",
            )
        context = packet.get("structured_context") if isinstance(packet.get("structured_context"), dict) else {}
        prescriptions = context.get("published_prescriptions") if isinstance(context.get("published_prescriptions"), list) else []
        if request.task_type == "patient_education" and _asks_about_current_prescription(request.input_text):
            return _patient_prescription_response(prescriptions)
        patient_self = context.get("patient_self") if isinstance(context.get("patient_self"), dict) else {}
        if request.task_type == "patient_education" and _asks_about_own_profile(request.input_text):
            return _patient_profile_response(patient_self)
        return GeneratedResponse(
            title="中医药科普说明",
            summary="这是基于患者安全版 Evidence Packet 的通俗说明，不能替代医生诊断和治疗。",
            sections=[
                GeneratedSection(heading="通俗解释", content=_patient_evidence_text(retrieved, packet.get("case_summary", "这是一般中医药科普信息。")), source_ids=source_ids),
                GeneratedSection(heading="安全提醒", content="如涉及开方、改药、加量、停药或症状变化，请咨询你的医生。", source_ids=source_ids),
            ],
            warnings=["不诊断、不开方、不改药、不加量、不停药。"],
            next_actions=["按医生已发布说明执行", "如有不适及时联系医生"],
            disclaimer="本内容仅用于中医药科普和医嘱说明，不替代医生诊断和治疗。",
        )


def _patient_evidence_text(retrieved: object, fallback: object) -> str:
    """Keep template fallback readable; raw evidence dictionaries are never patient copy."""
    items = retrieved if isinstance(retrieved, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        snippet = str(item.get("snippet") or item.get("content") or "").strip()
        if snippet:
            return snippet
    return compact_text(fallback or "这是一般中医药科普信息。")


def _retrieved_evidence_text(retrieved: object, fallback: str) -> str:
    items = retrieved if isinstance(retrieved, list) else []
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("content") or "").strip()
        key = (title, snippet)
        if not snippet or key in seen:
            continue
        seen.add(key)
        lines.append(f"{title}：{snippet}" if title else snippet)
        if len(lines) >= 3:
            break
    return "\n".join(lines) or fallback


def _business_context_count(packet: dict[str, Any]) -> int:
    context = packet.get("structured_context") if isinstance(packet.get("structured_context"), dict) else {}
    count = 0
    for key in ("patient_self", "published_prescriptions", "doctor_patient_roster"):
        value = context.get(key)
        if isinstance(value, list):
            count += len(value)
        elif value:
            count += 1
    return count


def _asks_about_current_prescription(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return any(term in compact for term in ("处方", "开的药", "给我开的", "怎么服", "服药", "医嘱", "医生备注"))


def _asks_about_patient_roster(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return any(term in compact for term in ("我的患者", "我有哪些患者", "现在的患者", "名下患者", "患者有谁", "病人有谁"))


def _mentions_roster_patient(text: str, roster: list[dict[str, Any]]) -> bool:
    compact = "".join(str(text or "").split())
    return any(
        str(patient.get(key) or "").strip() in compact
        for patient in roster
        if isinstance(patient, dict)
        for key in ("display_name", "username")
        if str(patient.get(key) or "").strip()
    )


def _asks_about_own_profile(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return any(term in compact for term in ("我的资料", "我的档案", "我的年龄", "我的性别", "我的过敏", "我的既往史", "我现在的症状"))


def _patient_prescription_response(prescriptions: list[dict[str, Any]]) -> GeneratedResponse:
    current = next((item for item in prescriptions if isinstance(item, dict) and item.get("is_current")), None)
    if current is None:
        current = next((item for item in prescriptions if isinstance(item, dict)), None)
    if not current:
        return GeneratedResponse(
            title="当前没有可见处方",
            summary="系统没有查到医生已发布给你的处方。草稿、审方中或尚未发布的处方不会在患者端问答中展示。",
            warnings=["请联系开方医生确认处方是否已经发布。"],
            next_actions=["到“处方查看”核对", "如仍未显示，请联系医生"],
            disclaimer="患者端只读取当前账号本人已发布的处方。",
        )
    items = current.get("items") if isinstance(current.get("items"), list) else []
    item_text = "、".join(
        f"{item.get('herb_name')}{item.get('dose') or ''}{item.get('unit') or ''}"
        for item in items
        if isinstance(item, dict) and item.get("herb_name")
    ) or "处方未列出药材明细"
    instruction = current.get("patient_instruction") if isinstance(current.get("patient_instruction"), dict) else {}
    diet = current.get("diet_precaution") if isinstance(current.get("diet_precaution"), dict) else {}
    usage = str(instruction.get("doctor_usage_method") or current.get("usage_method") or "医生未填写用法").strip()
    doctor_note = str(instruction.get("doctor_note") or diet.get("doctor_note") or "医生未填写备注").strip()
    food_rows = diet.get("food_interactions") if isinstance(diet.get("food_interactions"), list) else []
    food_text = "；".join(
        f"{row.get('herb_name')}：{row.get('message')}"
        for row in food_rows
        if isinstance(row, dict) and row.get("message")
    ) or "当前可追溯资料未收录本方与具体食物的相互作用记录，这不代表绝对没有相互作用。"
    return GeneratedResponse(
        title="本人当前已发布处方说明",
        summary=f"系统已读取医生当前发布的《{current.get('title') or '处方'}》，开方医生：{current.get('doctor_display_name') or '未记录'}。",
        sections=[
            GeneratedSection(heading="处方药材", content=item_text, source_ids=[]),
            GeneratedSection(heading="医生确认的服法", content=usage, source_ids=[]),
            GeneratedSection(heading="医生备注原文", content=doctor_note, source_ids=[]),
            GeneratedSection(heading="饮食与同服提醒", content=food_text, source_ids=[]),
        ],
        warnings=["不要根据问答自行增减药、改变剂量或停药。"],
        next_actions=["以处方查看页和医生原文为准", "对具体食物或同时服用的其他药物不确定时咨询医生或药师"],
        disclaimer="以上内容来自当前账号本人已发布处方，仅作医嘱转述，不构成自动诊断或改方。",
    )


def _doctor_roster_response(roster: list[dict[str, Any]]) -> GeneratedResponse:
    if not roster:
        return GeneratedResponse(
            title="当前名下患者",
            summary="系统未查到与当前医生账号有效绑定的患者。",
            next_actions=["前往患者管理绑定或新建患者"],
            disclaimer="仅显示当前登录医生有权限访问的患者。",
        )
    lines = []
    for index, patient in enumerate(roster, start=1):
        latest = patient.get("latest_prescription") if isinstance(patient.get("latest_prescription"), dict) else {}
        demographics = "、".join(
            value for value in (str(patient.get("sex") or "性别未填"), f"{patient.get('age')}岁" if patient.get("age") is not None else "年龄未填")
        )
        detail = f"{index}. {patient.get('display_name') or patient.get('username')}（{demographics}）"
        if patient.get("current_symptoms"):
            detail += f"；当前症状：{patient.get('current_symptoms')}"
        if latest:
            detail += f"；最近处方：{latest.get('title')}（{latest.get('status')}）"
        lines.append(detail)
    return GeneratedResponse(
        title="当前名下患者",
        summary=f"当前账号共有 {len(roster)} 位有效绑定患者。",
        sections=[GeneratedSection(heading="患者列表", content="\n".join(lines), source_ids=[])],
        warnings=["患者资料只应用于当前诊疗工作，不得向无关人员披露。"],
        next_actions=["如需查看完整过敏史、既往史或处方，请进入对应患者详情页"],
        disclaimer="以上信息来自当前登录医生的授权绑定关系。",
    )


def _patient_profile_response(patient_self: dict[str, Any]) -> GeneratedResponse:
    profile = patient_self.get("profile") if isinstance(patient_self.get("profile"), dict) else {}
    if not profile:
        return GeneratedResponse(
            title="本人健康档案",
            summary="当前账号尚未建立完整健康档案。",
            next_actions=["前往健康档案页查看", "请医生补充必要的诊疗资料"],
            disclaimer="患者端只读取当前账号本人的健康档案。",
        )
    lines = [
        f"姓名：{profile.get('display_name') or '未填写'}",
        f"性别：{profile.get('sex') or '未填写'}",
        f"年龄：{profile.get('age') if profile.get('age') is not None else '未填写'}",
        f"当前症状：{profile.get('current_symptoms') or '未填写'}",
        f"身体状态：{profile.get('body_state') or '未填写'}",
        f"既往史：{patient_self.get('medical_history') or '未填写'}",
        f"过敏史：{patient_self.get('allergy_history') or '未填写'}",
    ]
    return GeneratedResponse(
        title="本人健康档案",
        summary="以下内容来自当前登录账号的结构化健康档案。",
        sections=[GeneratedSection(heading="档案摘要", content="\n".join(lines), source_ids=[])],
        warnings=["如资料不准确，请联系你的医生核对后更新。"],
        next_actions=["过敏史或症状发生变化时及时告知医生"],
        disclaimer="本页是既有资料转述，不根据档案自动诊断或开方。",
    )
