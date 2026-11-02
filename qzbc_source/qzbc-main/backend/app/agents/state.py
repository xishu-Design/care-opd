from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    # identity
    role: Literal["doctor", "patient", "student"]
    task_type: str
    user_id: int
    session_id: str

    # resource
    patient_id: int | None
    prescription_id: int | None
    learning_case_id: int | None

    # input
    input_text: str
    input_payload: dict[str, Any]
    retrieval_query: str
    conversation_context: dict[str, Any]
    private_eval_evidence_control: dict[str, Any] | None
    request_path: str

    # runtime
    graph_name: str
    graph_version: str
    route: str
    resource_scope: dict[str, Any]
    memory_plan: list[dict[str, Any]]
    tool_requests: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    # outputs
    evidence_packet: dict[str, Any] | None
    safety_decision: dict[str, Any] | None
    generated_response: dict[str, Any] | None
    llm_result: dict[str, Any] | None
    final_response: dict[str, Any] | None

    # graph control
    auth_passed: bool
    should_continue: bool
    should_block: bool
    critical_error: bool
    requires_human_confirm: bool
    human_confirmed: bool
    interrupted: bool
    resume_action: str | None

    # observability
    trace_id: str
    thread_id: str
    graph_node_path: list[str]
    errors: list[dict[str, Any]]
    degraded: bool
    fallback_used: bool


def initial_agent_state(
    *,
    role: str,
    task_type: str,
    user_id: int,
    session_id: str,
    request_path: str,
    graph_name: str,
    input_text: str = "",
    input_payload: dict[str, Any] | None = None,
    patient_id: int | None = None,
    prescription_id: int | None = None,
    learning_case_id: int | None = None,
    trace_id: str | None = None,
    thread_id: str | None = None,
    graph_version: str = "0.8.0",
    private_eval_evidence_control: dict[str, Any] | None = None,
) -> AgentState:
    thread_id = thread_id or f"{role}:{user_id}:{session_id}"
    return {
        "role": role,
        "task_type": task_type,
        "user_id": user_id,
        "session_id": session_id,
        "thread_id": thread_id,
        "request_path": request_path,
        "graph_name": graph_name,
        "graph_version": graph_version,
        "patient_id": patient_id,
        "prescription_id": prescription_id,
        "learning_case_id": learning_case_id,
        "input_text": input_text,
        "input_payload": input_payload or {},
        "retrieval_query": input_text,
        "conversation_context": {},
        "private_eval_evidence_control": private_eval_evidence_control,
        "route": "",
        "resource_scope": {},
        "memory_plan": [],
        "tool_requests": [],
        "tool_results": [],
        "evidence_packet": None,
        "safety_decision": None,
        "generated_response": None,
        "llm_result": None,
        "final_response": None,
        "auth_passed": False,
        "should_continue": True,
        "should_block": False,
        "critical_error": False,
        "requires_human_confirm": False,
        "human_confirmed": False,
        "interrupted": False,
        "resume_action": None,
        "trace_id": trace_id,
        "graph_node_path": [],
        "errors": [],
        "degraded": False,
        "fallback_used": False,
    }
