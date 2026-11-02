import re

from sqlalchemy.orm import Session

from app.db.models import SafetyLog, User
from app.mcp.contracts import ToolRequest, ToolResult, ToolSpec


PATIENT_BLOCK_KEYWORDS = (
    "开方",
    "开个方",
    "开一个中药方",
    "中药方",
    "诊断",
    "加量",
    "改药",
    "停药",
    "判断证型",
    "直接告诉我吃什么",
    "不看医生",
    "推荐替代药",
    "替代处方",
    "换药",
    "改方",
    "改成",
    "用什么药",
    "具体克数",
    "网上方子",
    "食疗方",
    "不想看医生",
    "肾虚还是",
    "什么病",
    "治疗方案",
    "治疗性食疗",
    "食疗方治疗",
    "属于什么证型",
    "什么证型",
    "适合我吗",
    "适合我用",
    "我能吃",
    "我可以吃",
    "我该吃",
    "我能用",
    "我该用",
    "使用哪些中药",
    "用哪些中药",
    "有哪些中药",
    "哪些中药",
    "哪些药材",
    "推荐中药",
    "推荐药材",
    "吃什么中药",
)
STUDENT_BLOCK_KEYWORDS = ("手机号", "身份证", "完整病历", "真实姓名", "联系方式", "张三")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


_PATIENT_MEDICATION_SELECTION_PATTERN = re.compile(
    r"(?:用|使用|吃|服用|选择|选用|推荐).{0,8}(?:哪些|什么|哪种|哪味).{0,5}(?:中药|药材|方剂|药)"
    r"|(?:哪些|什么|哪种|哪味).{0,5}(?:中药|药材|方剂|药)"
)


def is_patient_medication_selection_request(text: str) -> bool:
    return bool(_PATIENT_MEDICATION_SELECTION_PATTERN.search(re.sub(r"\s+", "", text)))


def check_patient_education_request(db: Session, user: User, text: str) -> dict:
    medication_selection = is_patient_medication_selection_request(text)
    if _contains_any(text, PATIENT_BLOCK_KEYWORDS) or medication_selection:
        reply = (
            "我不能根据个人症状为你选择或推荐中药、药材或方剂，也不能把上一轮未经核验的药名当作依据。"
            "我可以解释经过核验的一般中医药知识；涉及个人症状和用药决定，请咨询医生或药师。"
        )
        log = SafetyLog(
            user_id=user.id,
            role=user.role,
            input_text=text,
            allowed=False,
            reason="patient_medical_request_not_allowed",
            safe_reply=reply,
        )
        db.add(log)
        db.commit()
        return {"allowed": False, "reason": log.reason, "safe_reply": reply}

    reply = "这是一般中医药科普信息，不能替代医生诊疗意见。如涉及用药调整，请咨询你的医生。"
    log = SafetyLog(
        user_id=user.id,
        role=user.role,
        input_text=text,
        allowed=True,
        reason="public_tcm_education_allowed",
        safe_reply=reply,
    )
    db.add(log)
    db.commit()
    return {"allowed": True, "reason": log.reason, "safe_reply": reply}


def check_student_request(db: Session, user: User, text: str) -> dict:
    if _contains_any(text, STUDENT_BLOCK_KEYWORDS):
        reply = "学习端只能访问脱敏教学案例，不能访问真实患者身份信息或完整病历。"
        db.add(
            SafetyLog(
                user_id=user.id,
                role=user.role,
                input_text=text,
                allowed=False,
                reason="student_real_patient_access_not_allowed",
                safe_reply=reply,
            )
        )
        db.commit()
        return {"allowed": False, "reason": "student_real_patient_access_not_allowed", "safe_reply": reply}
    return {"allowed": True, "reason": "student_learning_request_allowed", "safe_reply": ""}


def pre_check_tool_request(spec: ToolSpec, request: ToolRequest) -> ToolResult | None:
    joined_arguments = " ".join(str(value) for value in request.arguments.values())
    if request.context.role == "patient":
        forbidden_tools = {
            "create_prescription_draft",
            "run_prescription_review",
            "run_mock_prescription_review",
            "create_deidentified_learning_case",
            "get_patient_profile",
        }
        if request.tool_name in forbidden_tools:
            return _blocked_result(request, "PATIENT_TOOL_NOT_ALLOWED", "patient role cannot call clinical prescription tools")
        if _contains_any(joined_arguments, PATIENT_BLOCK_KEYWORDS):
            return _blocked_result(request, "PATIENT_MEDICAL_REQUEST_NOT_ALLOWED", "patient medical request is not allowed")
    if request.context.role == "student":
        forbidden_tools = {"get_patient_profile", "get_patient_medical_history", "get_patient_allergy_history"}
        if request.tool_name in forbidden_tools:
            return _blocked_result(request, "STUDENT_REAL_PATIENT_TOOL_NOT_ALLOWED", "student cannot access real patient tools")
        if _contains_any(joined_arguments, STUDENT_BLOCK_KEYWORDS):
            return _blocked_result(request, "STUDENT_REAL_PATIENT_ACCESS_NOT_ALLOWED", "student cannot access real patient information")
    if spec.risk_level == "high" and request.context.role != "doctor" and spec.requires_confirmation:
        return _blocked_result(request, "HIGH_RISK_TOOL_REQUIRES_DOCTOR", "high risk tools require doctor context")
    return None


def post_check_tool_result(result: ToolResult, request: ToolRequest) -> ToolResult | None:
    if not result.success or not result.data:
        return None
    output_text = str(result.data)
    if request.context.role == "patient":
        forbidden = ("rule_id", "doctor_internal", "graph_reasoning_path", "内部推理链")
        if any(token in output_text for token in forbidden):
            return _blocked_result(request, "PATIENT_OUTPUT_CONTAINS_INTERNAL_INFO", "patient output contains internal information")
        # These two adapters return authoritative self-scoped records. Their
        # stored safety statement may contain wording such as “不构成自动开方”,
        # which is a boundary disclaimer rather than medical advice. Treating
        # that substring as unsafe previously discarded the entire current
        # prescription and forced the AI to answer generically.
        trusted_self_record = request.tool_name in {"get_my_profile", "get_my_published_prescriptions"}
        if not trusted_self_record and _contains_any(output_text, ("开方", "加量", "停药", "改药", "治疗这个病")):
            return _blocked_result(request, "PATIENT_OUTPUT_MEDICAL_ADVICE_NOT_ALLOWED", "patient output contains unsafe medical advice")
    if request.context.role == "student":
        if _contains_any(output_text, STUDENT_BLOCK_KEYWORDS):
            return _blocked_result(request, "STUDENT_OUTPUT_CONTAINS_REAL_PATIENT_INFO", "student output contains real patient information")
    return None


def sanitize_patient_memory_context(data: dict) -> tuple[dict, int]:
    """Drop unsafe historical memory items individually instead of blocking all memory.

    Patient memories are context, not medical evidence. A past blocked request may contain
    words such as "开方" while other entries contain useful explanation preferences. The
    previous all-or-nothing output check discarded both, which made Mem0 appear unavailable.
    """
    items = data.get("memory_used") if isinstance(data.get("memory_used"), list) else []
    blocked_tokens = (
        "开方",
        "处方如下",
        "建议服用",
        "建议你用",
        "建议你吃",
        "加量",
        "停药",
        "改药",
        "治疗这个病",
        "具体克数",
    )
    safe_items = []
    filtered_count = 0
    for item in items:
        text = str(item)
        if any(token in text for token in blocked_tokens):
            filtered_count += 1
            continue
        safe_items.append(item)
    return {**data, "memory_used": safe_items}, filtered_count


def _blocked_result(request: ToolRequest, error_code: str, message: str) -> ToolResult:
    return ToolResult(
        request_id=request.request_id,
        tool_name=request.tool_name,
        allowed=False,
        success=False,
        data=None,
        risk_level="high",
        requires_confirmation=False,
        error_code=error_code,
        message=message,
        metadata={"blocked_by": "safety_guard"},
    )
