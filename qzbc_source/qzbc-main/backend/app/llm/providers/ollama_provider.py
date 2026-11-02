from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import AsyncIterator
from time import perf_counter

import httpx

from app.core.config import get_settings
from app.llm.output_parser import parse_generated_response_result
from app.llm.providers.base import BaseLLMProvider, ProviderGenerationError, ProviderUnavailable
from app.llm.prompts.doctor_review_prompt import DOCTOR_REVIEW_SYSTEM_PROMPT
from app.llm.prompts.doctor_ai_chat_prompt import DOCTOR_AI_CHAT_SYSTEM_PROMPT
from app.llm.prompts.doctor_case_reasoning_prompt import DOCTOR_CASE_REASONING_SYSTEM_PROMPT
from app.llm.prompts.patient_education_prompt import PATIENT_EDUCATION_SYSTEM_PROMPT
from app.llm.prompts.patient_diet_guidance_prompt import PATIENT_DIET_GUIDANCE_SYSTEM_PROMPT
from app.llm.prompts.quiz_generation_prompt import QUIZ_GENERATION_SYSTEM_PROMPT
from app.llm.prompts.student_analysis_prompt import STUDENT_ANALYSIS_SYSTEM_PROMPT
from app.llm.prompts.student_quiz_explanation_prompt import STUDENT_QUIZ_EXPLANATION_SYSTEM_PROMPT
from app.llm.prompts.student_ai_chat_prompt import STUDENT_AI_CHAT_SYSTEM_PROMPT
from app.llm.schemas import GeneratedResponse, LLMGrounding, LLMMetadata, LLMRequest, LLMResult, ProviderHealth


_CASE_REASONING_SECTIONS = (
    (
        "病例要点",
        (r"病例要点",),
    ),
    (
        "中医辨病与候选辨证",
        (r"中医辨病与候选辨证", r"中医辨病与辨证", r"辨病与候选辨证"),
    ),
    (
        "病因病机及“症状/舌脉—判断”证据链",
        (
            r"病因病机及[“\"]?症状[／/]舌脉[—-]判断[”\"]?证据链",
            r"病因病机及证据链",
            r"病因病机与证据链",
        ),
    ),
    (
        "治则治法",
        (r"治则治法",),
    ),
    (
        "方药思路与加减原则",
        (r"方药思路与加减原则", r"方药思路及加减原则", r"方药与加减原则"),
    ),
    (
        "需要鉴别或进一步复核的信息",
        (
            r"需要鉴别或进一步复核的信息",
            r"鉴别与进一步复核",
            r"鉴别及进一步复核",
            r"鉴别诊断与复核",
        ),
    ),
)
from app.llm.fallback import evidence_count, extract_source_ids


class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    def health(self) -> ProviderHealth:
        settings = get_settings()
        if not settings.ollama_base_url or not settings.ollama_model:
            return ProviderHealth(
                name=self.name,
                configured=False,
                reachable=False,
                status="misconfigured",
                model=settings.ollama_model or None,
                last_error="OLLAMA_BASE_URL and OLLAMA_MODEL are required",
            )
        started = perf_counter()
        try:
            req = urllib.request.Request(settings.ollama_base_url.rstrip("/") + "/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=settings.ollama_health_timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return ProviderHealth(
                name=self.name,
                configured=True,
                reachable=False,
                status="unavailable",
                latency_ms=None,
                model=settings.ollama_model,
                last_error=_error_text(exc),
            )

        latency_ms = int((perf_counter() - started) * 1000)
        models = [str(item.get("name") or item.get("model") or "") for item in raw.get("models", []) if isinstance(item, dict)]
        if settings.ollama_model not in models:
            return ProviderHealth(
                name=self.name,
                configured=True,
                reachable=True,
                status="misconfigured",
                latency_ms=latency_ms,
                model=settings.ollama_model,
                last_error=f"model_not_found:{settings.ollama_model}",
            )
        return ProviderHealth(name=self.name, configured=True, reachable=True, status="ok", latency_ms=latency_ms, model=settings.ollama_model)

    def generate(self, request: LLMRequest) -> LLMResult:
        settings = get_settings()
        if not settings.ollama_base_url or not settings.ollama_model:
            raise ProviderUnavailable("Ollama is not configured")
        started = perf_counter()
        payload = {
            "model": settings.ollama_model,
            "messages": _chat_messages(request, streaming=False),
            "stream": False,
            "format": GeneratedResponse.model_json_schema(),
            "options": {
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "seed": settings.llm_seed,
                "num_predict": settings.llm_max_output_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
        }
        try:
            req = urllib.request.Request(
                settings.ollama_base_url.rstrip("/") + "/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=settings.ollama_generate_timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderGenerationError(_error_text(exc)) from exc
        if raw.get("error"):
            raise ProviderGenerationError(str(raw["error"]))
        content = raw.get("message", {}).get("content", "")
        if not content:
            raise ProviderGenerationError("Ollama returned empty content")
        parsed = parse_generated_response_result(content)
        if not parsed.success or parsed.structured_output is None:
            raise ProviderGenerationError(f"parser_error:{parsed.parse_error or 'invalid_output'}")
        structured_output = _normalize_case_reasoning_output(
            request, parsed.structured_output
        )
        source_ids = extract_source_ids(request.evidence_packet)
        return LLMResult(
            request_id=request.request_id,
            success=True,
            provider=self.name,
            model=settings.ollama_model,
            content=content,
            structured_output=structured_output,
            grounding=LLMGrounding(
                used_evidence_count=evidence_count(request.evidence_packet),
                missing_evidence=evidence_count(request.evidence_packet) == 0 or not source_ids,
                source_ids=source_ids,
            ),
            metadata=LLMMetadata(
                latency_ms=int((perf_counter() - started) * 1000),
                parse_success=parsed.success,
                schema_valid=parsed.schema_valid,
                parser_error=parsed.parse_error,
            ),
        )

    async def stream_text(self, request: LLMRequest, metrics: dict[str, object]) -> AsyncIterator[str]:
        """Stream display-ready text from Ollama without waiting for structured JSON."""
        settings = get_settings()
        if not settings.ollama_base_url or not settings.ollama_model:
            raise ProviderUnavailable("Ollama is not configured")
        payload = {
            "model": settings.ollama_model,
            "messages": _chat_messages(request, streaming=True),
            "stream": True,
            "options": {
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "seed": settings.llm_seed,
                "num_predict": settings.llm_max_output_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
        }
        metrics.update({"provider": self.name, "model": settings.ollama_model})
        timeout = httpx.Timeout(settings.ollama_generate_timeout_seconds, connect=5.0)
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    settings.ollama_base_url.rstrip("/") + "/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        raw = json.loads(line)
                        if raw.get("error"):
                            raise ProviderGenerationError(str(raw["error"]))
                        text = str((raw.get("message") or {}).get("content") or "")
                        if text:
                            yield text
                        if raw.get("done"):
                            metrics.update(
                                {
                                    "provider": self.name,
                                    "model": settings.ollama_model,
                                    "latency_ms": int((perf_counter() - started) * 1000),
                                    "token_usage": {
                                        "input_tokens": int(raw.get("prompt_eval_count") or 0),
                                        "output_tokens": int(raw.get("eval_count") or 0),
                                    },
                                }
                            )
        except (ProviderGenerationError, ProviderUnavailable):
            raise
        except Exception as exc:
            raise ProviderGenerationError(_error_text(exc)) from exc


def _system_prompt(task_type: str, *, case_reasoning: bool = False) -> str:
    if task_type == "doctor_ai_chat" and case_reasoning:
        return DOCTOR_CASE_REASONING_SYSTEM_PROMPT
    return {
        "doctor_review_text": DOCTOR_REVIEW_SYSTEM_PROMPT,
        "doctor_ai_chat": DOCTOR_AI_CHAT_SYSTEM_PROMPT,
        "patient_education": PATIENT_EDUCATION_SYSTEM_PROMPT,
        "patient_diet_guidance": PATIENT_DIET_GUIDANCE_SYSTEM_PROMPT,
        "student_analysis": STUDENT_ANALYSIS_SYSTEM_PROMPT,
        "student_quiz_explanation": STUDENT_QUIZ_EXPLANATION_SYSTEM_PROMPT,
        "student_ai_chat": STUDENT_AI_CHAT_SYSTEM_PROMPT,
        "quiz_generation": QUIZ_GENERATION_SYSTEM_PROMPT,
    }.get(task_type, PATIENT_EDUCATION_SYSTEM_PROMPT)


def _stream_system_prompt(task_type: str, *, case_reasoning: bool = False) -> str:
    base = _system_prompt(task_type, case_reasoning=case_reasoning).replace("输出 JSON。", "").strip()
    return (
        f"{base}\n"
        "请直接输出可展示的中文正文，不要输出 JSON、代码块或字段名。"
        "使用简洁小标题和自然段；回答必须完整但避免冗长。"
    )


def _chat_messages(request: LLMRequest, *, streaming: bool) -> list[dict[str, str]]:
    """Assemble a real multi-turn prompt instead of hiding all history in vector memory.

    The recent SQL window is deterministic and ordered. Mem0/Graphiti data remains in
    the Evidence Packet as selectively retrieved long-term context.
    """
    case_reasoning = _case_reasoning_contract(request) is not None
    system = (
        _stream_system_prompt(request.task_type, case_reasoning=case_reasoning)
        if streaming
        else _system_prompt(request.task_type, case_reasoning=case_reasoning)
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    context = request.evidence_packet.get("conversation_context")
    turns = context.get("recent_turns") if isinstance(context, dict) else []
    include_assistant_answers = bool(context.get("include_assistant_answers")) if isinstance(context, dict) else False
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, dict):
            continue
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if answer and include_assistant_answers and turn.get("answer_trusted") is True:
            messages.append({"role": "assistant", "content": answer})
    current_payload = _stream_user_payload(request) if streaming else _user_payload(request)
    messages.append({"role": "user", "content": json.dumps(current_payload, ensure_ascii=False)})
    return messages


def _user_payload(request: LLMRequest) -> dict:
    patient_general_education = request.role == "patient" and request.task_type == "patient_education"
    case_contract = _case_reasoning_contract(request)
    case_reasoning = case_contract is not None
    output_sections = [{"heading": "", "content": "", "source_ids": []}]
    if case_contract is not None:
        case_source_id = str(case_contract["case_source_id"])
        output_sections = [
            {
                "heading": heading,
                "content": "",
                "source_ids": [case_source_id],
            }
            for heading, _ in _CASE_REASONING_SECTIONS
        ]
    return {
        "role": request.role,
        "task_type": request.task_type,
        "input_text": request.input_text,
        "constraints": request.constraints,
        "evidence_packet": _prompt_evidence_packet(
            request.evidence_packet, case_reasoning=case_reasoning
        ),
        "conversation_context_meta": _conversation_context_meta(request.evidence_packet),
        "hard_constraints": _hard_constraints(
            patient_general_education, case_reasoning=case_reasoning
        ),
        "output_schema": {
            "title": "",
            "summary": "",
            "sections": output_sections,
            "warnings": [],
            "next_actions": [],
            "disclaimer": "",
        },
    }


def _normalize_case_reasoning_output(
    request: LLMRequest, structured_output: dict
) -> dict:
    """Recover six explicitly titled blocks without inventing model content.

    Some OpenAI-compatible decoders treat an array example as a fixed one-item
    schema and place all requested headings in that item's content.  The prompt
    now supplies all six objects, but this deterministic recovery keeps older
    compliant text usable by splitting only when every heading is present in
    the expected order.  Otherwise the downstream fail-closed quality gate sees
    the original output unchanged.
    """

    contract = _case_reasoning_contract(request)
    if contract is None:
        return structured_output
    sections = [
        section
        for section in structured_output.get("sections") or []
        if isinstance(section, dict) and str(section.get("content") or "").strip()
    ]
    if len(sections) >= len(_CASE_REASONING_SECTIONS):
        return structured_output
    if len(sections) != 1:
        return structured_output

    original = sections[0]
    combined = "\n".join(
        part
        for part in (
            str(original.get("heading") or "").strip(),
            str(original.get("content") or "").strip(),
        )
        if part
    )
    matches: list[tuple[int, int]] = []
    cursor = 0
    for _, patterns in _CASE_REASONING_SECTIONS:
        best_span = None
        for pattern in patterns:
            candidate = re.compile(pattern).search(combined, cursor)
            if candidate is None:
                continue
            span = candidate.span()
            if best_span is None or span[0] < best_span[0]:
                best_span = span
        if best_span is None:
            return structured_output
        matches.append(best_span)
        cursor = best_span[1]

    inherited_source_ids = [
        str(source_id)
        for source_id in original.get("source_ids") or []
        if str(source_id).strip()
    ]
    recovered = []
    for index, ((heading, _), span) in enumerate(
        zip(_CASE_REASONING_SECTIONS, matches, strict=True)
    ):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(combined)
        content = combined[span[1] : end].strip(" \t\r\n：:、.-")
        if not content:
            return structured_output
        recovered.append(
            {
                "heading": heading,
                "content": content,
                "source_ids": inherited_source_ids,
            }
        )
    return {**structured_output, "sections": recovered}


def _stream_user_payload(request: LLMRequest) -> dict:
    patient_general_education = request.role == "patient" and request.task_type == "patient_education"
    case_reasoning = _case_reasoning_contract(request) is not None
    constraints = _hard_constraints(
        patient_general_education, case_reasoning=case_reasoning
    )[1:]
    if case_reasoning:
        structure = "按病例要点、候选辨证、病机证据链、治则治法、无剂量方药思路、鉴别复核六部分组织。"
    elif request.role == "doctor":
        structure = "先给知识结论，再说明证候—治法—方剂—药材关系、证据边界和医生复核提示。"
    elif request.role == "student":
        structure = "按照概念解释、辨析思路、知识联系、复习问题组织。"
    else:
        structure = "采用通俗科普表达，先解释概念，再说明安全边界；不能给个体诊疗或用药建议。"
    return {
        "role": request.role,
        "task_type": request.task_type,
        "input_text": request.input_text,
        "current_request": {
            "text": request.input_text,
            "priority": "highest",
            "instruction": "只回答本轮问题；不得把历史问题当成本轮任务。",
        },
        "constraints": request.constraints,
        "evidence_packet": _prompt_evidence_packet(
            request.evidence_packet, case_reasoning=case_reasoning
        ),
        "conversation_context_meta": _conversation_context_meta(request.evidence_packet),
        "hard_constraints": constraints,
        "answer_structure": structure,
    }


def _compact_prompt_text(value: object, *, limit: int = 900) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _compact_prompt_list(value: object, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        text
        for item in value[:limit]
        if (text := _compact_prompt_text(item, limit=240))
    ]


def _compact_case_evidence_item(item: dict, case_source_id: str) -> dict:
    """Project one persisted evidence item into a bounded inference view.

    The full Evidence Packet remains unchanged in the audit record.  This view
    removes repository metadata, keyword inventories and repeated source
    records that do not help case reasoning, while retaining every retrieved
    item's source identity and substantive snippet.
    """

    source_id = str(item.get("source_id") or "").strip()
    if (
        item.get("kind") == "request_bound_case_material"
        and source_id == case_source_id
    ):
        return {
            "source_id": source_id,
            "kind": "request_bound_case_material",
            "content": str(item.get("content") or ""),
            "claim_status": "user_supplied_case_fact",
            "current_request_bound": True,
        }

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    compact: dict[str, object] = {
        "source_id": source_id,
        "source_ids": [
            str(value)
            for value in item.get("source_ids") or []
            if str(value).strip()
        ],
        "title": _compact_prompt_text(item.get("title"), limit=180),
        "snippet": _compact_prompt_text(
            item.get("snippet") or item.get("summary") or item.get("content"),
            limit=700,
        ),
        "retrieved_by": str(item.get("retrieved_by") or ""),
        "confidence": item.get("confidence"),
        "evidence_level": str(metadata.get("evidence_level") or ""),
        "functions": _compact_prompt_list(metadata.get("functions"), limit=6),
        "cautions": _compact_prompt_list(metadata.get("cautions"), limit=4),
        "usage_note": _compact_prompt_text(metadata.get("usage_note"), limit=260),
        "not_clinical_recommendation": bool(
            item.get("not_clinical_recommendation")
            or metadata.get("not_clinical_recommendation")
        ),
    }
    components = item.get("components")
    if isinstance(components, list):
        compact["component_names"] = [
            name
            for component in components[:16]
            if isinstance(component, dict)
            and (
                name := _compact_prompt_text(
                    component.get("herb_name") or component.get("name"), limit=80
                )
            )
        ]
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _case_reasoning_prompt_packet(packet: dict) -> dict:
    structured = (
        packet.get("structured_context")
        if isinstance(packet.get("structured_context"), dict)
        else {}
    )
    contract = (
        structured.get("case_reasoning_contract")
        if isinstance(structured.get("case_reasoning_contract"), dict)
        else {}
    )
    case_source_id = str(contract.get("case_source_id") or "")
    evidence = [
        _compact_case_evidence_item(item, case_source_id)
        for item in packet.get("retrieved_evidence") or []
        if isinstance(item, dict)
    ]
    case_items = [
        item
        for item in evidence
        if item.get("kind") == "request_bound_case_material"
    ]
    if len(case_items) != 1:
        # The validated caller should never reach this branch.  Preserve the
        # unprojected packet so downstream contract checks remain fail-closed.
        return packet

    backend = (
        packet.get("backend_summary")
        if isinstance(packet.get("backend_summary"), dict)
        else {}
    )
    return {
        "packet_id": packet.get("packet_id"),
        "role": packet.get("role"),
        "task_type": packet.get("task_type"),
        "visibility": packet.get("visibility"),
        "request_bound_case_material": case_items[0],
        "retrieved_knowledge": [
            item
            for item in evidence
            if item.get("kind") != "request_bound_case_material"
        ],
        "graph_reasoning_path": packet.get("graph_reasoning_path") or [],
        "memory_used": packet.get("memory_used") or [],
        "structured_context": {"case_reasoning_contract": dict(contract)},
        "safety_decision": packet.get("safety_decision") or {},
        "backend_summary": {
            key: backend.get(key)
            for key in (
                "knowledge_backend",
                "graph_backend",
                "memory_backend",
                "retrieval_mode",
                "fallback_used",
                "degraded_tools",
            )
            if key in backend
        },
        "prompt_projection": {
            "policy": "content_neutral_bounded_evidence_projection_v1",
            "full_packet_retained_for_audit": True,
            "retrieved_item_count": len(evidence),
        },
    }


def _prompt_evidence_packet(packet: dict, *, case_reasoning: bool = False) -> dict:
    """Avoid duplicate turns and bound irrelevant prompt metadata.

    Operational/audit fields stay in the persisted Evidence Packet.  For the
    teaching-case route, only a deterministic content-neutral projection is
    sent to the model so the request-bound case is not buried under repository
    provenance and keyword inventories.
    """
    compact = {key: value for key, value in packet.items() if key != "conversation_context"}
    memory = compact.get("memory_used")
    if isinstance(memory, list):
        compact["memory_used"] = [
            item
            for item in memory
            if not isinstance(item, dict)
            or (
                item.get("source") != "sql_recent_chat"
                and item.get("memory_type") not in {"ai_chat_turn", "short_term_conversation_turn"}
            )
        ]
    if case_reasoning:
        return _case_reasoning_prompt_packet(compact)
    return compact


def _conversation_context_meta(packet: dict) -> dict:
    context = packet.get("conversation_context") if isinstance(packet.get("conversation_context"), dict) else {}
    return {
        "turn_count": int(context.get("turn_count") or 0),
        "available_turn_count": int(context.get("available_turn_count") or 0),
        "follow_up_detected": bool(context.get("follow_up_detected")),
        "context_mode": str(context.get("context_mode") or "unknown"),
        "assistant_answers_included": bool(context.get("include_assistant_answers")),
        "usage": "只有与当前问题连续的最近用户消息会被提供；新话题不得注入旧回答。",
    }


def _case_reasoning_contract(request: LLMRequest) -> dict | None:
    structured = request.evidence_packet.get("structured_context")
    contract = (
        structured.get("case_reasoning_contract")
        if isinstance(structured, dict)
        else None
    )
    if not isinstance(contract, dict):
        return None
    if (
        contract.get("protocol") != "public_anonymized_tcm_case_reasoning_v1"
        or contract.get("current_request_bound") is not True
        or contract.get("complete_material_bound") is not True
        or contract.get("reference_answer_included") is not False
        or contract.get("clinical_use_authorized") is not False
        or contract.get("model_reference_answer_access") is not False
        or contract.get("product_adapters_bypassed") is not False
    ):
        return None
    source_id = str(contract.get("case_source_id") or "")
    expected_sha256 = str(contract.get("case_material_sha256") or "")
    matching = [
        item
        for item in request.evidence_packet.get("retrieved_evidence") or []
        if isinstance(item, dict)
        and item.get("kind") == "request_bound_case_material"
        and item.get("source_id") == source_id
    ]
    if len(matching) != 1:
        return None
    content = str(matching[0].get("content") or "")
    if (
        not content
        or hashlib.sha256(content.encode("utf-8")).hexdigest() != expected_sha256
    ):
        return None
    metadata = matching[0].get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("case_material_sha256") != expected_sha256
        or metadata.get("current_request_bound") is not True
        or metadata.get("reference_answer_included") is not False
    ):
        return None
    return contract


def _hard_constraints(
    patient_general_education: bool, *, case_reasoning: bool = False
) -> list[str]:
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
    if case_reasoning:
        return base + [
            "sections 必须恰好包含六个非空对象，并严格依次使用以下 heading：病例要点；中医辨病与候选辨证；病因病机及“症状/舌脉—判断”证据链；治则治法；方药思路与加减原则；需要鉴别或进一步复核的信息。不得把六部分合并进一个 section；第四、第五节必须分别说清具体治法方向和无剂量方药或组方方向；第六节必须写明至少一项具体鉴别对象，或需要核对、补充的病例要素，不能只写建议复核或补充资料。以语义完整为先，不设凑字目标，不得重复填充。",
            "病例观察事实只能来自 kind=request_bound_case_material 且哈希已核验的本轮证据；禁止补造未提供的症状、舌象、脉象、病史或检查结果。",
            "允许运用中医通用理论，由病例事实形成明确标为候选、倾向或推断的教学判断；推断不是病例原文，也不能表述成已确诊事实。",
            "完整、哈希绑定的病例正文就是本轮病例证据；缺少舌象、脉象或某项检查只降低置信度并进入第六节，不得据此取消对已有阳性表现的候选推理。只有病例正文为空、不可解析或完全没有症状体征时才允许整体降级。",
            "第二节必须直接写出至少一个具体证候或病机候选，优先采用‘首要候选为【具体候选名称】’；‘候选方向：需结合……’‘待补充后判断’‘进一步鉴别’等流程性文字不构成候选。置信不足时仍给出教学候选并标明不确定性。第三节至少写出两条彼此独立的‘病例已知表现……提示/支持……’事实—推断链。",
            "第四节必须用‘若……则可考虑……’等条件表达给出与候选相连的具体治法方向；第五节必须用‘可参考……随证加减’等条件表达给出无剂量且有具体方向的方药或组方思路。只写‘调理’或‘某方’不合格。",
            "六个 sections 必须全部引用 case_reasoning_contract.case_source_id；辨证、病机、治法和方药思路必须逐项回指病例事实，不得用标题、免责声明或待补信息替代实质分析。",
            "方药仅可给出无剂量的教学性思路与加减原则；禁止固定剂量、用法、疗程、现实处方或绕过医生复核。",
            "不得使用或推测标准答案、参考解析、标签或评分信息；Evidence Packet 未提供这些内容。",
        ]
    return base + ["任何事实、风险和建议都只能直接改写 Evidence Packet；证据不足时明确说明，不得补充模型记忆。"]


def _error_text(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__
