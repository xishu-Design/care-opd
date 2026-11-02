from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.agents.state import AgentState
from app.harness.trace import record_event


def copy_state(state: AgentState) -> AgentState:
    copied = dict(state)
    copied["graph_node_path"] = list(state.get("graph_node_path", []))
    copied["errors"] = list(state.get("errors", []))
    copied["memory_plan"] = list(state.get("memory_plan", []))
    copied["tool_requests"] = list(state.get("tool_requests", []))
    copied["tool_results"] = list(state.get("tool_results", []))
    copied["input_payload"] = dict(state.get("input_payload", {}))
    copied["resource_scope"] = dict(state.get("resource_scope", {}))
    control = state.get("private_eval_evidence_control")
    copied["private_eval_evidence_control"] = dict(control) if isinstance(control, dict) else None
    return copied


def append_node(
    db: Session,
    state: AgentState,
    node_name: str,
    *,
    status: str = "success",
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    safety_decision: dict[str, Any] | None = None,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    path = list(state.get("graph_node_path", []))
    path.append(node_name)
    state["graph_node_path"] = path
    record_event(
        db,
        state.get("trace_id"),
        node_name=node_name,
        event_type="langgraph_node",
        status=status,
        input_summary=input_summary,
        output_summary={
            **(output_summary or {}),
            "graph_node_path": path,
        },
        safety_decision=safety_decision,
        error_code=error_code,
        metadata={
            "graph_name": state.get("graph_name"),
            "graph_version": state.get("graph_version"),
            "thread_id": state.get("thread_id"),
            **(metadata or {}),
        },
    )
    return state


def add_error(state: AgentState, code: str, message: str, *, status_code: int | None = None) -> AgentState:
    errors = list(state.get("errors", []))
    error = {"code": code, "message": message}
    if status_code is not None:
        error["status_code"] = status_code
    errors.append(error)
    state["errors"] = errors
    return state


def block_with_http_error(state: AgentState, exc: HTTPException) -> AgentState:
    state["should_block"] = True
    state["should_continue"] = False
    state["final_response"] = {
        "allowed": False,
        "reason": "auth_or_resource_scope_failed",
        "detail": exc.detail,
        "status_code": exc.status_code,
    }
    return add_error(state, f"HTTP_{exc.status_code}", str(exc.detail), status_code=exc.status_code)


def evidence_packet_id(state: AgentState) -> str | None:
    packet = state.get("evidence_packet") or {}
    return packet.get("packet_id")


def packet_visibility_for_role(role: str) -> str:
    return {
        "doctor": "doctor_internal",
        "patient": "patient_safe",
        "student": "student_learning",
    }.get(role, "doctor_internal")
