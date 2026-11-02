from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.agents.nodes.auth_node import AuthNode
from app.agents.nodes.conversation_context_node import ConversationContextNode
from app.agents.nodes.evidence_packet_node import EvidencePacketNode
from app.agents.nodes.llm_generate_node import LLMGenerateNode
from app.agents.nodes.mcp_tool_node import MCPToolNode
from app.agents.nodes.memory_router_node import MemoryRouterNode
from app.agents.nodes.response_node import ResponseNode
from app.agents.nodes.router_node import RouterNode
from app.agents.nodes.safety_node import SafetyPostNode, SafetyPreNode
from app.agents.nodes.trace_node import TraceNode
from app.agents.state import AgentState, initial_agent_state
from app.core.config import get_settings
from app.db.models import User
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.providers.template_provider import TemplateProvider
from app.llm.safety import sanitize_stream_segment, student_training_reference_response
from app.llm.doctor_response_policy import doctor_direct_response
from app.llm.patient_response_policy import patient_direct_response, response_display_text
from app.llm.service import LLMService
from app.mcp.contracts import new_request_id
from app.services.patient_education_history_service import display_answer, save_education_chat
from app.services.role_ai_chat_service import save_role_ai_chat
from app.services.memory_consolidation_service import MemoryConsolidationAgent


_SENTENCE_END = re.compile(r"[。！？；\n]")


class StreamingAIChatService:
    def __init__(self, db: Session):
        self.db = db

    def start(self, user: User, message: str) -> AgentState:
        task_type = "patient_education" if user.role == "patient" else f"{user.role}_ai_chat"
        state = initial_agent_state(
            role=user.role,
            task_type=task_type,
            user_id=user.id,
            session_id=new_request_id(),
            request_path=f"/api/{user.role}/{'education-chat' if user.role == 'patient' else 'ai-chat'}/stream",
            graph_name="StreamingAIChatGraph",
            input_text=message,
            patient_id=user.id if user.role == "patient" else None,
            thread_id=f"{user.role}:{user.id}:{user.username}:default_{'education' if user.role == 'patient' else 'ai'}_chat",
        )
        state = AuthNode(self.db)(state)
        if state.get("should_block"):
            return state
        state = RouterNode(self.db)(state)
        if user.role == "patient":
            state = SafetyPreNode(self.db)(state)
        if not state.get("should_block"):
            state = ConversationContextNode(self.db)(state)
        return state

    def retrieve(self, state: AgentState) -> AgentState:
        if state.get("should_block"):
            return state
        state = MemoryRouterNode(self.db)(state)
        state = MCPToolNode(self.db)(state)
        if not state.get("critical_error"):
            state = EvidencePacketNode(self.db)(state)
        self.db.commit()
        return state

    def finish_without_generation(self, state: AgentState) -> dict[str, Any]:
        state = ResponseNode(self.db)(state)
        state = TraceNode(self.db)(state)
        return dict(state.get("final_response") or {})

    def finalize(
        self,
        state: AgentState,
        *,
        content: str,
        metrics: dict[str, Any],
        interrupted: bool,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> dict[str, Any]:
        packet = state.get("evidence_packet") or {}
        request = LLMService(self.db).build_chat_request(
            role=state["role"],
            evidence_packet=packet,
            context={
                "trace_id": state.get("trace_id"),
                "user_id": state.get("user_id"),
                "input_text": state.get("input_text", ""),
                "constraints": {"role": state.get("role"), "visibility": packet.get("visibility")},
            },
        )
        result = LLMService(self.db).finalize_streamed(
            request,
            content=content,
            provider=str(metrics.get("provider") or "template"),
            model=str(metrics.get("model") or "guardtcm-template-v1"),
            latency_ms=int(metrics.get("latency_ms") or 0),
            token_usage=metrics.get("token_usage") if isinstance(metrics.get("token_usage"), dict) else None,
            interrupted=interrupted,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        state = LLMGenerateNode(self.db).apply_result(state, result)
        state = SafetyPostNode(self.db)(state)
        state = ResponseNode(self.db)(state)
        state = TraceNode(self.db)(state)
        response = dict(state.get("final_response") or {})
        response["interrupted"] = interrupted
        return response

    def save(self, user: User, message: str, result: dict[str, Any]):
        if user.role == "patient":
            return save_education_chat(self.db, patient_id=user.id, question=message, result=result)
        return save_role_ai_chat(self.db, user_id=user.id, role=user.role, question=message, result=result)


async def stream_ai_chat_events(db: Session, user: User, message: str) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    service = StreamingAIChatService(db)
    state: AgentState | None = None
    content = ""
    metrics: dict[str, Any] = {}
    started = perf_counter()
    saved = False
    fallback_used = False
    fallback_reason: str | None = None

    try:
        yield "status", {"message": "正在进行权限与安全检查…"}
        await asyncio.sleep(0.01)
        acknowledgement = MemoryConsolidationAgent(db).preference_acknowledgement(
            owner_role=user.role,
            user_message=message,
        )
        if acknowledgement and user.role in {"doctor", "student", "patient"}:
            visibility = "patient_safe" if user.role == "patient" else "doctor_internal" if user.role == "doctor" else "student_learning"
            result = {
                "answer": acknowledgement,
                "generated_response": {
                    "title": "已更新学习偏好" if user.role == "student" else "已更新回答偏好",
                    "summary": acknowledgement,
                    "sections": [],
                    "warnings": [],
                    "next_actions": [],
                    "disclaimer": "长期记忆仅用于调整回答方式，不作为医学证据。",
                },
                "llm_result": {
                    "provider": "memory_consolidation_policy",
                    "model": "memory-canonicalizer-v1",
                    "content": acknowledgement,
                    "safety_flags": ["non_medical_preference_instruction"],
                    "metadata": {"fallback_used": False, "degraded": False},
                },
                "evidence_packet": {"visibility": visibility, "structured_context": {"memory_instruction": True}, "backend_summary": {"memory_backend": "persistent_memory"}},
                "trace_id": new_request_id(),
                "graph_node_path": ["MemoryConsolidationAgent", "ResponseNode"],
                "degraded": False,
                "fallback_used": False,
            }
            for character in acknowledgement:
                yield "answer", {"text": character}
            chat = service.save(user, message, result)
            saved = True
            result["chat_id"] = chat.id
            result["created_at"] = chat.created_at.isoformat()
            yield "complete", {"result": result}
            return
        state = service.start(user, message)
        if state.get("should_block"):
            result = service.finish_without_generation(state)
            answer = display_answer(result)
            for character in answer:
                content += character
                yield "answer", {"text": character}
            chat = service.save(user, message, result)
            saved = True
            result["chat_id"] = chat.id
            result["created_at"] = chat.created_at.isoformat()
            yield "complete", {"result": result}
            return

        yield "status", {"message": "正在检索知识、图谱与历史记忆…"}
        await asyncio.sleep(0.01)
        state = service.retrieve(state)
        if state.get("critical_error"):
            result = service.finish_without_generation(state)
            answer = display_answer(result)
            for character in answer:
                content += character
                yield "answer", {"text": character}
            chat = service.save(user, message, result)
            saved = True
            result["chat_id"] = chat.id
            result["created_at"] = chat.created_at.isoformat()
            yield "complete", {"result": result}
            return

        yield "status", {"message": "正在生成回答，可随时停止…"}
        await asyncio.sleep(0.01)
        packet = state.get("evidence_packet") or {}
        request = LLMService(db).build_chat_request(
            role=user.role,
            evidence_packet=packet,
            context={
                "trace_id": state.get("trace_id"),
                "user_id": user.id,
                "input_text": message,
                "constraints": {"role": user.role, "visibility": packet.get("visibility")},
            },
        )
        pending = ""
        emitted = False
        settings = get_settings()
        patient_policy = patient_direct_response(message, packet) if user.role == "patient" else None
        doctor_policy = doctor_direct_response(message, packet) if user.role == "doctor" else None
        training_reference = student_training_reference_response(packet) if user.role == "student" else None
        if patient_policy is not None:
            response, _reason = patient_policy
            pending = response_display_text(response)
            metrics.update(
                {
                    "provider": "patient_safety_policy",
                    "model": "patient-safety-v1",
                    "latency_ms": int((perf_counter() - started) * 1000),
                }
            )
            emitted = True
        elif doctor_policy is not None:
            response, _reason = doctor_policy
            pending = response_display_text(response)
            metrics.update(
                {
                    "provider": "doctor_business_context_policy",
                    "model": "doctor-context-v1",
                    "latency_ms": int((perf_counter() - started) * 1000),
                }
            )
            emitted = True
        elif training_reference is not None:
            pending = _training_reference_text(training_reference)
            metrics.update(
                {
                    "provider": "grounded_qa",
                    "model": "MTCMB-reference",
                    "latency_ms": int((perf_counter() - started) * 1000),
                }
            )
            emitted = True
        elif settings.llm_provider == "ollama":
            try:
                async for chunk in OllamaProvider().stream_text(request, metrics):
                    pending += chunk
                    complete_segments, pending = _take_complete_segments(pending)
                    for segment in complete_segments:
                        safe_segment = sanitize_stream_segment(user.role, segment)
                        content += safe_segment
                        emitted = emitted or bool(safe_segment)
                        yield "answer", {"text": safe_segment}
            except Exception as exc:
                if emitted or not settings.llm_allow_fallback:
                    raise
                fallback_used = True
                fallback_reason = f"ollama:{str(exc) or exc.__class__.__name__}"
        else:
            fallback_used = settings.llm_provider != "template"
            fallback_reason = f"{settings.llm_provider}_stream_not_supported" if fallback_used else None

        if not emitted and (settings.llm_provider != "ollama" or fallback_used):
            template = TemplateProvider().generate(request)
            pending += display_answer({"generated_response": template.structured_output})
            metrics.update(
                {
                    "provider": "template",
                    "model": template.model or "guardtcm-template-v1",
                    "latency_ms": int((perf_counter() - started) * 1000),
                }
            )
        if pending:
            safe_segment = sanitize_stream_segment(user.role, pending)
            content += safe_segment
            yield "answer", {"text": safe_segment}

        metrics.setdefault("latency_ms", int((perf_counter() - started) * 1000))
        result = service.finalize(
            state,
            content=content,
            metrics=metrics,
            interrupted=False,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        chat = service.save(user, message, result)
        saved = True
        result["chat_id"] = chat.id
        result["created_at"] = chat.created_at.isoformat()
        yield "complete", {"result": result}
    except asyncio.CancelledError:
        if state is not None and not saved:
            metrics.setdefault("latency_ms", int((perf_counter() - started) * 1000))
            partial = content or "回答已由用户停止生成。"
            result = service.finalize(
                state,
                content=partial,
                metrics=metrics,
                interrupted=True,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            service.save(user, message, result)
        raise
    except Exception as exc:
        if state is not None and not saved and content:
            metrics.setdefault("latency_ms", int((perf_counter() - started) * 1000))
            result = service.finalize(
                state,
                content=content,
                metrics=metrics,
                interrupted=True,
                fallback_used=True,
                fallback_reason=str(exc) or exc.__class__.__name__,
            )
            service.save(user, message, result)
        yield "error", {"message": "回答生成中断，请稍后重试。"}


def _training_reference_text(response: dict[str, Any]) -> str:
    sections = response.get("sections") if isinstance(response.get("sections"), list) else []
    parts = [str(response.get("summary") or "").strip()]
    parts.extend(
        f"{section.get('heading')}：{section.get('content')}"
        for section in sections
        if isinstance(section, dict) and section.get("heading") and section.get("content")
    )
    disclaimer = str(response.get("disclaimer") or "").strip()
    if disclaimer:
        parts.append(disclaimer)
    return "\n\n".join(part for part in parts if part)


def _take_complete_segments(value: str) -> tuple[list[str], str]:
    segments: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(value):
        end = match.end()
        segments.append(value[start:end])
        start = end
    return segments, value[start:]
