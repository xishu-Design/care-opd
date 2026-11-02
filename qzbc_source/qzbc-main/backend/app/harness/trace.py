from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AgentTrace, AgentTraceEvent, EvidencePacketLog, LLMGenerationLog, MCPToolCall, PendingHumanAction, SafetyLog, utcnow


def new_trace_id() -> str:
    return f"trace_{uuid4().hex}"


def start_trace(
    db: Session,
    *,
    user_id: int | None,
    role: str,
    task_type: str,
    request_path: str,
    session_id: str | None = None,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentTrace:
    trace = AgentTrace(
        trace_id=trace_id or new_trace_id(),
        user_id=user_id,
        role=role,
        session_id=session_id,
        task_type=task_type,
        request_path=request_path,
        status="running",
        metadata_json=metadata or {},
    )
    db.add(trace)
    db.flush()
    record_event(
        db,
        trace.trace_id,
        node_name="AuthNode",
        event_type="auth",
        status="success",
        output_summary={"role": role, "user_id": user_id},
    )
    return trace


def record_event(
    db: Session,
    trace_id: str | None,
    *,
    node_name: str,
    event_type: str,
    status: str = "success",
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    tool_name: str | None = None,
    evidence_packet_id: str | None = None,
    safety_decision: dict[str, Any] | None = None,
    error_code: str | None = None,
    duration_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
    parent_span_id: str | None = None,
) -> AgentTraceEvent | None:
    if not trace_id:
        return None
    event = AgentTraceEvent(
        trace_id=trace_id,
        span_id=f"span_{uuid4().hex}",
        parent_span_id=parent_span_id,
        node_name=node_name,
        event_type=event_type,
        status=status,
        input_summary=_summarize(input_summary or {}),
        output_summary=_summarize(output_summary or {}),
        tool_name=tool_name,
        evidence_packet_id=evidence_packet_id,
        safety_decision=safety_decision or {},
        error_code=error_code,
        duration_ms=duration_ms,
        metadata_json=metadata or {},
    )
    db.add(event)
    db.flush()
    return event


def finish_trace(
    db: Session,
    trace_id: str | None,
    *,
    status: str = "success",
    degraded: bool = False,
    fallback_used: bool = False,
    metadata: dict[str, Any] | None = None,
) -> AgentTrace | None:
    if not trace_id:
        return None
    trace = db.scalar(select(AgentTrace).where(AgentTrace.trace_id == trace_id))
    if not trace:
        return None
    ended_at = utcnow()
    trace.status = status
    trace.degraded = degraded
    trace.fallback_used = fallback_used
    trace.ended_at = ended_at
    started_at = trace.started_at
    comparable_ended_at = ended_at.replace(tzinfo=None) if started_at.tzinfo is None else ended_at
    trace.duration_ms = int((comparable_ended_at - started_at).total_seconds() * 1000)
    if metadata:
        trace.metadata_json = {**(trace.metadata_json or {}), **metadata}
    record_event(
        db,
        trace_id,
        node_name="ResponseNode",
        event_type="response",
        status=status,
        output_summary={"degraded": degraded, "fallback_used": fallback_used},
    )
    db.flush()
    return trace


def timed_event(db: Session, trace_id: str | None, **kwargs) -> None:
    started = perf_counter()
    kwargs.setdefault("duration_ms", int((perf_counter() - started) * 1000))
    record_event(db, trace_id, **kwargs)


def list_traces(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    rows = db.scalars(select(AgentTrace).order_by(AgentTrace.started_at.desc()).limit(limit)).all()
    return [
        {
            "trace_id": row.trace_id,
            "role": row.role,
            "task_type": row.task_type,
            "status": row.status,
            "degraded": row.degraded,
            "fallback_used": row.fallback_used,
            "duration_ms": row.duration_ms,
            "created_at": row.started_at.isoformat(),
        }
        for row in rows
    ]


def get_trace_detail(db: Session, trace_id: str) -> dict[str, Any]:
    trace = db.scalar(select(AgentTrace).where(AgentTrace.trace_id == trace_id))
    if not trace:
        return {}
    events = db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == trace_id).order_by(AgentTraceEvent.created_at)).all()
    tool_calls = db.scalars(select(MCPToolCall).where(MCPToolCall.trace_id == trace_id).order_by(MCPToolCall.created_at)).all()
    packets = db.scalars(select(EvidencePacketLog).where(EvidencePacketLog.trace_id == trace_id).order_by(EvidencePacketLog.created_at)).all()
    pending_actions = db.scalars(
        select(PendingHumanAction).where(PendingHumanAction.trace_id == trace_id).order_by(PendingHumanAction.created_at)
    ).all()
    llm_logs = db.scalars(select(LLMGenerationLog).where(LLMGenerationLog.trace_id == trace_id).order_by(LLMGenerationLog.created_at)).all()
    safety_logs = db.scalars(select(SafetyLog).where(SafetyLog.user_id == trace.user_id).order_by(SafetyLog.created_at.desc()).limit(10)).all()
    return {
        "trace_id": trace.trace_id,
        "summary": {
            "role": trace.role,
            "task_type": trace.task_type,
            "status": trace.status,
            "degraded": trace.degraded,
            "fallback_used": trace.fallback_used,
            "duration_ms": trace.duration_ms,
            "request_path": trace.request_path,
        },
        "events": [_serialize_event(event) for event in events],
        "tool_calls": [
            {
                "tool_name": call.tool_name,
                "allowed": call.allowed,
                "success": call.success,
                "error_code": call.error_code,
                "backend": call.backend,
                "degraded": call.degraded,
                "fallback_used": call.fallback_used,
            }
            for call in tool_calls
        ],
        "evidence_packets": [
            {
                "packet_id": packet.packet_id,
                "visibility": packet.visibility,
                "degraded": packet.degraded,
                "backend_summary": packet.backend_summary_json,
                "graph_node_path": packet.graph_node_path_json,
                "human_confirm": packet.human_confirm_json,
                "evidence_count": len(packet.retrieved_evidence_json),
            }
            for packet in packets
        ],
        "pending_human_actions": [_serialize_pending_action(action) for action in pending_actions],
        "llm_generations": [_serialize_llm_log(log) for log in llm_logs],
        "safety_logs": [
            {
                "allowed": log.allowed,
                "reason": log.reason,
                "safe_reply": log.safe_reply,
                "created_at": log.created_at.isoformat(),
            }
            for log in safety_logs
        ],
    }


def _serialize_llm_log(log: LLMGenerationLog) -> dict[str, Any]:
    return {
        "request_id": log.request_id,
        "trace_id": log.trace_id,
        "role": log.role,
        "task_type": log.task_type,
        "provider": log.provider,
        "model": log.model,
        "success": log.success,
        "fallback_used": log.fallback_used,
        "degraded": log.degraded,
        "source_ids": log.source_ids_json,
        "safety_flags": log.safety_flags_json,
        "latency_ms": log.latency_ms,
        "error_code": log.error_code,
        "created_at": log.created_at.isoformat(),
    }


def _serialize_pending_action(action: PendingHumanAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "trace_id": action.trace_id,
        "thread_id": action.thread_id,
        "graph_name": action.graph_name,
        "node_name": action.node_name,
        "prescription_id": action.prescription_id,
        "patient_id": action.patient_id,
        "doctor_id": action.doctor_id,
        "evidence_packet_id": action.evidence_packet_id,
        "status": action.status,
        "risk_level": action.risk_level,
        "review_summary": action.review_summary_json,
        "interrupt_payload": action.interrupt_payload_json,
        "resume_payload": action.resume_payload_json,
        "resume_comment": action.resume_comment,
        "created_at": action.created_at.isoformat(),
        "resumed_at": action.resumed_at.isoformat() if action.resumed_at else None,
        "expires_at": action.expires_at.isoformat() if action.expires_at else None,
    }


def _serialize_event(event: AgentTraceEvent) -> dict[str, Any]:
    return {
        "span_id": event.span_id,
        "parent_span_id": event.parent_span_id,
        "node_name": event.node_name,
        "event_type": event.event_type,
        "status": event.status,
        "input_summary": event.input_summary,
        "output_summary": event.output_summary,
        "tool_name": event.tool_name,
        "evidence_packet_id": event.evidence_packet_id,
        "safety_decision": event.safety_decision,
        "error_code": event.error_code,
        "duration_ms": event.duration_ms,
        "metadata": event.metadata_json,
        "created_at": event.created_at.isoformat(),
    }


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > 180:
            summary[key] = value[:180] + "..."
        else:
            summary[key] = value
    return summary
