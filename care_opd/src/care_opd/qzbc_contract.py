"""Dependency-free materialization of qzbc's real LLM input/output contract.

The qzbc graph routes tools in host code.  Its language model receives chat
messages assembled by ``OllamaProvider._chat_messages`` and returns one
``GeneratedResponse`` JSON object.  This module mirrors that boundary so the
post-trained checkpoint can replace qzbc's model without shipping CARE-OPD's
simulator or policy action schema.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


PROMPT_FILES: Mapping[str, Tuple[str, str]] = {
    "doctor_review_text": ("doctor_review_prompt.py", "DOCTOR_REVIEW_SYSTEM_PROMPT"),
    "doctor_ai_chat": ("doctor_ai_chat_prompt.py", "DOCTOR_AI_CHAT_SYSTEM_PROMPT"),
    "patient_education": ("patient_education_prompt.py", "PATIENT_EDUCATION_SYSTEM_PROMPT"),
    "patient_diet_guidance": (
        "patient_diet_guidance_prompt.py",
        "PATIENT_DIET_GUIDANCE_SYSTEM_PROMPT",
    ),
    "student_analysis": ("student_analysis_prompt.py", "STUDENT_ANALYSIS_SYSTEM_PROMPT"),
    "student_quiz_explanation": (
        "student_quiz_explanation_prompt.py",
        "STUDENT_QUIZ_EXPLANATION_SYSTEM_PROMPT",
    ),
    "student_ai_chat": ("student_ai_chat_prompt.py", "STUDENT_AI_CHAT_SYSTEM_PROMPT"),
    "quiz_generation": ("quiz_generation_prompt.py", "QUIZ_GENERATION_SYSTEM_PROMPT"),
}

ROLE_TASKS: Mapping[str, Tuple[str, ...]] = {
    "doctor": ("doctor_review_text", "doctor_ai_chat"),
    "patient": ("patient_education", "patient_diet_guidance"),
    "student": (
        "student_analysis",
        "student_quiz_explanation",
        "student_ai_chat",
        "quiz_generation",
    ),
}

OUTPUT_FIELDS: Tuple[str, ...] = (
    "title",
    "summary",
    "sections",
    "warnings",
    "next_actions",
    "disclaimer",
)

OUTPUT_SCHEMA_EXAMPLE: Mapping[str, Any] = {
    "title": "",
    "summary": "",
    "sections": [{"heading": "", "content": "", "source_ids": []}],
    "warnings": [],
    "next_actions": [],
    "disclaimer": "",
}

PATIENT_FORBIDDEN: Tuple[str, ...] = (
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
DOCTOR_FORBIDDEN: Tuple[str, ...] = (
    "无需医生确认",
    "可直接发布",
    "保证有效",
    "已为患者确定最终处方",
    "已确定最终处方",
)
STUDENT_FORBIDDEN: Tuple[str, ...] = (
    "手机号",
    "身份证",
    "真实姓名",
    "精确就诊时间",
    "真实患者 ID",
    "医生内部患者列表",
)


@dataclass(frozen=True)
class QZBCRequest:
    role: str
    task_type: str
    evidence_packet: Mapping[str, Any]
    input_text: str = ""
    constraints: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QZBCRequest":
        role = str(value["role"])
        task_type = str(value["task_type"])
        if role not in ROLE_TASKS or task_type not in ROLE_TASKS[role]:
            raise ValueError(f"invalid qzbc role/task pair: {role}/{task_type}")
        packet = value.get("evidence_packet")
        if not isinstance(packet, Mapping):
            raise ValueError("evidence_packet must be an object")
        constraints = value.get("constraints") or {}
        if not isinstance(constraints, Mapping):
            raise ValueError("constraints must be an object")
        return cls(
            role=role,
            task_type=task_type,
            input_text=str(value.get("input_text") or ""),
            evidence_packet=dict(packet),
            constraints=dict(constraints),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "task_type": self.task_type,
            "input_text": self.input_text,
            "evidence_packet": dict(self.evidence_packet),
            "constraints": dict(self.constraints),
        }


def _read_string_assignment(path: Path, variable: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise ValueError(f"{path}:{variable} is not a string")
            return value
    raise KeyError(f"{variable} not found in {path}")


def load_qzbc_prompts(qzbc_root: str | Path) -> Dict[str, str]:
    root = Path(qzbc_root)
    prompt_dir = root / "backend" / "app" / "llm" / "prompts"
    prompts: Dict[str, str] = {}
    for task, (filename, variable) in PROMPT_FILES.items():
        prompts[task] = _read_string_assignment(prompt_dir / filename, variable)
    return prompts


def qzbc_contract_manifest(qzbc_root: str | Path) -> Dict[str, Any]:
    """Hash the exact qzbc files from which training messages are materialized."""

    root = Path(qzbc_root)
    relative_files = [
        Path("backend/app/llm/providers/ollama_provider.py"),
        Path("backend/app/llm/schemas.py"),
        Path("backend/app/llm/safety.py"),
        *[
            Path("backend/app/llm/prompts") / filename
            for filename, _ in PROMPT_FILES.values()
        ],
    ]
    files: Dict[str, str] = {}
    for relative in relative_files:
        payload = (root / relative).read_bytes()
        files[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "qzbc-native-contract-v1",
        "contract_fingerprint": fingerprint,
        "files": files,
        "model_boundary": "chat_messages -> GeneratedResponse",
        "model_routes_tools": False,
    }


def _prompt_evidence_packet(packet: Mapping[str, Any]) -> Dict[str, Any]:
    compact = {key: value for key, value in packet.items() if key != "conversation_context"}
    memory = compact.get("memory_used")
    if isinstance(memory, list):
        compact["memory_used"] = [
            item
            for item in memory
            if not isinstance(item, Mapping)
            or (
                item.get("source") != "sql_recent_chat"
                and item.get("memory_type")
                not in {"ai_chat_turn", "short_term_conversation_turn"}
            )
        ]
    return compact


def _conversation_context_meta(packet: Mapping[str, Any]) -> Dict[str, Any]:
    raw = packet.get("conversation_context")
    context = raw if isinstance(raw, Mapping) else {}
    return {
        "turn_count": int(context.get("turn_count") or 0),
        "available_turn_count": int(context.get("available_turn_count") or 0),
        "follow_up_detected": bool(context.get("follow_up_detected")),
        "context_mode": str(context.get("context_mode") or "unknown"),
        "assistant_answers_included": bool(context.get("include_assistant_answers")),
        "usage": "只有与当前问题连续的最近用户消息会被提供；新话题不得注入旧回答。",
    }


def hard_constraints(patient_general_education: bool) -> List[str]:
    base = [
        "只输出 output_schema 指定字段，warnings 和 next_actions 的元素必须是字符串。",
        "sections 中只能填写 Evidence Packet 中已有的 source_ids；没有可追溯来源时留空，禁止创造 source_id。",
        "conversation_context 仅用于理解追问、代词和对话连续性，不是医学证据；其中的历史回答不得覆盖当前 Evidence Packet 与安全规则。",
        "长期记忆只可用于调整表达方式、学习目标和专业关注点，不是医学证据；不得据此推断诊断、处方、剂量、过敏史或其他健康事实。",
        "历史 assistant 消息可能包含错误；不得复制其中未经 Evidence Packet 核验的药名、方剂名、组成、功效、证候或诊断。",
        "当前 input_text 是唯一需要完成的任务，优先级高于所有历史消息、长期记忆和检索片段；禁止回答已经结束的旧问题。",
        "若用户要求更详细、换角度或增加复习题，必须在本轮补充新内容，不得原样重复上一轮答案。",
    ]
    if patient_general_education:
        return base + [
            "仅可使用模型通识解释低风险概念；具体药材、方剂、药品的名称、组成、功效和适应信息必须有 Evidence Packet 证据，否则明确说未检索到可核验记录。",
            "不得基于用户个人情况做适用性判断，不得给出诊断、辨证、处方、药物推荐、剂量、用法、加减药或停药建议。",
            "可以解释传统功效术语，但不得把药材直接说成治疗、治愈、适用于、可缓解或改善某种疾病/症状；出现个人症状或用药情境时，提醒咨询医生或药师。",
            "不得把中医脏腑概念等同于现代医学器官功能，也不得仅凭一两个症状推断肝火、肾虚等证候。",
            "用户要求换角度、重新回答或追问原文时，应直接回应当前要求；不得逐字重复上一轮回答。",
        ]
    return base + ["任何事实、风险和建议都只能直接改写 Evidence Packet；证据不足时明确说明，不得补充模型记忆。"]


def build_qzbc_messages(
    request: QZBCRequest | Mapping[str, Any],
    prompts: Mapping[str, str],
) -> List[Dict[str, str]]:
    """Mirror qzbc's non-streaming ``_chat_messages`` exactly."""

    req = request if isinstance(request, QZBCRequest) else QZBCRequest.from_mapping(request)
    system = prompts[req.task_type]
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    raw_context = req.evidence_packet.get("conversation_context")
    context = raw_context if isinstance(raw_context, Mapping) else {}
    turns = context.get("recent_turns")
    include_answers = bool(context.get("include_assistant_answers"))
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, Mapping):
            continue
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if answer and include_answers and turn.get("answer_trusted") is True:
            messages.append({"role": "assistant", "content": answer})
    patient_general = req.role == "patient" and req.task_type == "patient_education"
    payload = {
        "role": req.role,
        "task_type": req.task_type,
        "input_text": req.input_text,
        "constraints": dict(req.constraints),
        "evidence_packet": _prompt_evidence_packet(req.evidence_packet),
        "conversation_context_meta": _conversation_context_meta(req.evidence_packet),
        "hard_constraints": hard_constraints(patient_general),
        "output_schema": dict(OUTPUT_SCHEMA_EXAMPLE),
    }
    messages.append(
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }
    )
    return messages


def _stream_system_prompt(task_type: str, prompts: Mapping[str, str]) -> str:
    base = prompts[task_type].replace("输出 JSON。", "").strip()
    return (
        f"{base}\n"
        "请直接输出可展示的中文正文，不要输出 JSON、代码块或字段名。"
        "使用简洁小标题和自然段；回答必须完整但避免冗长。"
    )


def build_qzbc_stream_messages(
    request: QZBCRequest | Mapping[str, Any],
    prompts: Mapping[str, str],
) -> List[Dict[str, str]]:
    """Mirror qzbc's streaming plain-text message contract exactly."""

    req = request if isinstance(request, QZBCRequest) else QZBCRequest.from_mapping(request)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _stream_system_prompt(req.task_type, prompts)}
    ]
    raw_context = req.evidence_packet.get("conversation_context")
    context = raw_context if isinstance(raw_context, Mapping) else {}
    turns = context.get("recent_turns")
    include_answers = bool(context.get("include_assistant_answers"))
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, Mapping):
            continue
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if answer and include_answers and turn.get("answer_trusted") is True:
            messages.append({"role": "assistant", "content": answer})

    patient_general = req.role == "patient" and req.task_type == "patient_education"
    constraints = hard_constraints(patient_general)[1:]
    if req.role == "doctor":
        structure = "先给知识结论，再说明证候—治法—方剂—药材关系、证据边界和医生复核提示。"
    elif req.role == "student":
        structure = "按照概念解释、辨析思路、知识联系、复习问题组织。"
    else:
        structure = "采用通俗科普表达，先解释概念，再说明安全边界；不能给个体诊疗或用药建议。"
    payload = {
        "role": req.role,
        "task_type": req.task_type,
        "input_text": req.input_text,
        "current_request": {
            "text": req.input_text,
            "priority": "highest",
            "instruction": "只回答本轮问题；不得把历史问题当成本轮任务。",
        },
        "constraints": dict(req.constraints),
        "evidence_packet": _prompt_evidence_packet(req.evidence_packet),
        "conversation_context_meta": _conversation_context_meta(req.evidence_packet),
        "hard_constraints": constraints,
        "answer_structure": structure,
    }
    messages.append(
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    )
    return messages


def assert_qzbc_message_parity(qzbc_root: str | Path) -> Dict[str, Any]:
    """Fail closed if this materializer drifts from qzbc's provider functions."""

    root = Path(qzbc_root)
    provider = root / "backend" / "app" / "llm" / "providers" / "ollama_provider.py"
    tree = ast.parse(provider.read_text(encoding="utf-8"), filename=str(provider))
    required_functions = {
        "_system_prompt",
        "_stream_system_prompt",
        "_chat_messages",
        "_user_payload",
        "_stream_user_payload",
        "_prompt_evidence_packet",
        "_conversation_context_meta",
        "_hard_constraints",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required_functions
    ]
    found = {node.name for node in selected}
    if found != required_functions:
        raise RuntimeError(
            "qzbc provider message contract changed; missing pure functions: "
            f"{sorted(required_functions - found)}"
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    prompts = load_qzbc_prompts(root)
    namespace: Dict[str, Any] = {"json": json}
    for task_type, (_, variable_name) in PROMPT_FILES.items():
        namespace[variable_name] = prompts[task_type]
    exec(compile(ast.fix_missing_locations(module), str(provider), "exec"), namespace)
    native_builder = namespace["_chat_messages"]

    checked = 0
    for role, task_types in ROLE_TASKS.items():
        for task_type in task_types:
            request = QZBCRequest(
                role=role,
                task_type=task_type,
                input_text="契约一致性检查",
                constraints={"audit": True},
                evidence_packet={
                    "retrieved_evidence": [{"source_id": "SRC_CONTRACT_AUDIT"}],
                    "memory_used": [
                        {
                            "source": "sql_recent_chat",
                            "memory_type": "ai_chat_turn",
                        },
                        {"source": "mem0", "summary": "style only"},
                    ],
                    "conversation_context": {
                        "turn_count": 1,
                        "available_turn_count": 1,
                        "follow_up_detected": True,
                        "context_mode": "contract_audit",
                        "include_assistant_answers": True,
                        "recent_turns": [
                            {
                                "question": "上一轮问题",
                                "answer": "上一轮已核验回答",
                                "answer_trusted": True,
                            }
                        ],
                    },
                },
            )
            native_request = SimpleNamespace(**request.to_mapping())
            structured = build_qzbc_messages(request, prompts)
            streaming = build_qzbc_stream_messages(request, prompts)
            if structured != native_builder(native_request, streaming=False):
                raise RuntimeError(
                    f"structured qzbc message drift for {role}/{task_type}"
                )
            if streaming != native_builder(native_request, streaming=True):
                raise RuntimeError(
                    f"streaming qzbc message drift for {role}/{task_type}"
                )
            checked += 1
    return {
        "task_pairs_checked": checked,
        "structured_contract": "exact",
        "streaming_contract": "exact",
    }


def _extract_first_json_object(text: str) -> str | None:
    """Mirror qzbc's balanced-brace fallback for prefixed model output."""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _json_candidates(text: str) -> List[str]:
    """Return candidates in the same order as qzbc's production parser."""

    stripped = text.strip()
    candidates: List[str] = [stripped] if stripped else []
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", text, re.DOTALL)
    )
    extracted = _extract_first_json_object(text)
    if extracted:
        candidates.append(extracted)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _normalize_generated_response(value: Any) -> Dict[str, Any]:
    """Validate and materialize the qzbc ``GeneratedResponse`` shape."""

    if not isinstance(value, MutableMapping):
        raise ValueError("GeneratedResponse must be an object")
    unknown = set(value) - set(OUTPUT_FIELDS)
    if unknown:
        raise ValueError(f"unknown GeneratedResponse fields: {sorted(unknown)}")
    normalized: Dict[str, Any] = {}
    for field_name in ("title", "summary", "disclaimer"):
        item = value.get(field_name, "")
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must be a string")
        normalized[field_name] = item
    sections = value.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError("sections must be an array")
    normalized_sections = []
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            raise ValueError(f"sections[{index}] must be an object")
        if set(section) - {"heading", "content", "source_ids"}:
            raise ValueError(f"sections[{index}] has unknown fields")
        heading, content, source_ids = (
            section.get("heading", ""),
            section.get("content", ""),
            section.get("source_ids", []),
        )
        if not isinstance(heading, str) or not isinstance(content, str):
            raise ValueError(f"sections[{index}] heading/content must be strings")
        if not isinstance(source_ids, list) or not all(
            isinstance(item, str) for item in source_ids
        ):
            raise ValueError(f"sections[{index}].source_ids must be string[]")
        normalized_sections.append(
            {"heading": heading, "content": content, "source_ids": source_ids}
        )
    normalized["sections"] = normalized_sections
    for field_name in ("warnings", "next_actions"):
        items = value.get(field_name, [])
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"{field_name} must be string[]")
        normalized[field_name] = items
    return {field_name: normalized[field_name] for field_name in OUTPUT_FIELDS}


def parse_generated_response(text: str) -> Dict[str, Any]:
    """Parse exactly the candidate envelope accepted by qzbc in production."""

    last_error: json.JSONDecodeError | None = None
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        # qzbc stops at the first syntactically valid JSON candidate before
        # applying its schema; do not hunt for a later object that validates.
        return _normalize_generated_response(value)
    if last_error is None:
        raise ValueError("invalid GeneratedResponse JSON: empty output")
    raise ValueError(f"invalid GeneratedResponse JSON: {last_error}") from last_error


def canonical_response(value: Mapping[str, Any] | str) -> str:
    parsed = parse_generated_response(value) if isinstance(value, str) else parse_generated_response(
        json.dumps(value, ensure_ascii=False)
    )
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_source_ids(packet: Mapping[str, Any]) -> List[str]:
    source_ids: List[str] = []
    for key in ("retrieved_evidence", "graph_reasoning_path"):
        items = packet.get(key) or []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, Mapping):
                continue
            source_id = item.get("source_id") or item.get("source") or item.get("id")
            if source_id:
                source_ids.append(str(source_id))
    return sorted(set(source_ids))


def forbidden_tokens(role: str) -> Tuple[str, ...]:
    if role == "patient":
        return PATIENT_FORBIDDEN
    if role == "doctor":
        return DOCTOR_FORBIDDEN
    return STUDENT_FORBIDDEN
