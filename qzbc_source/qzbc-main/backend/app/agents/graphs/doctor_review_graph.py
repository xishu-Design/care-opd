from sqlalchemy.orm import Session
from langgraph.types import Command
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.auth_node import AuthNode
from app.agents.nodes.evidence_packet_node import EvidencePacketNode
from app.agents.nodes.human_confirm_node import HumanConfirmNode
from app.agents.nodes.llm_generate_node import LLMGenerateNode
from app.agents.nodes.mcp_tool_node import MCPToolNode
from app.agents.nodes.memory_router_node import MemoryRouterNode
from app.agents.nodes.review_persist_node import ReviewPersistNode
from app.agents.nodes.response_node import ResponseNode
from app.agents.nodes.router_node import RouterNode
from app.agents.nodes.safety_node import SafetyPostNode
from app.agents.nodes.trace_node import TraceNode
from app.agents.services.checkpoint_service import get_langgraph_checkpointer
from app.agents.services.graph_runtime import ensure_langgraph_enabled, graph_config_for_state
from app.agents.services.human_action_service import get_pending_action_by_trace, mark_interrupted, serialize_pending_action
from app.agents.state import AgentState, initial_agent_state
from app.core.config import get_settings
from app.db.models import User
from app.harness.trace import new_trace_id
from app.mcp.contracts import new_request_id


class DoctorReviewGraph:
    graph_name = "DoctorReviewGraph"

    def __init__(self, db: Session):
        self.db = db

    def invoke(self, doctor: User, prescription_id: int) -> dict:
        ensure_langgraph_enabled()
        trace_id = new_trace_id()
        thread_id = f"{get_settings().langgraph_thread_id_prefix}:doctor_review:{prescription_id}:{trace_id}"
        state = initial_agent_state(
            role="doctor",
            task_type="prescription_review",
            user_id=doctor.id,
            session_id=new_request_id(),
            request_path=f"/api/doctor/prescriptions/{prescription_id}/review",
            graph_name=self.graph_name,
            prescription_id=prescription_id,
            trace_id=trace_id,
            thread_id=thread_id,
        )
        result = self._graph().invoke(state, config=graph_config_for_state(state))
        return self._response(result)

    def resume(self, *, thread_id: str, approved: bool, comment: str | None, doctor_id: int) -> dict:
        ensure_langgraph_enabled()
        result = self._graph().invoke(
            Command(resume={"approved": approved, "comment": comment or "", "doctor_id": doctor_id}),
            config={"configurable": {"thread_id": thread_id}},
        )
        return self._response(result)

    def _graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("AuthNode", AuthNode(self.db))
        builder.add_node("RouterNode", RouterNode(self.db))
        builder.add_node("MemoryRouterNode", MemoryRouterNode(self.db))
        builder.add_node("MCPToolNode", MCPToolNode(self.db))
        builder.add_node("EvidencePacketNode", EvidencePacketNode(self.db))
        builder.add_node("LLMGenerateNode", LLMGenerateNode(self.db))
        builder.add_node("ReviewPersistNode", ReviewPersistNode(self.db))
        builder.add_node("SafetyPostNode", SafetyPostNode(self.db))
        builder.add_node("HumanConfirmNode", HumanConfirmNode(self.db))
        builder.add_node("ResponseNode", ResponseNode(self.db))
        builder.add_node("TraceNode", TraceNode(self.db))

        builder.add_edge(START, "AuthNode")
        builder.add_conditional_edges("AuthNode", _auth_branch, {"blocked": "ResponseNode", "continue": "RouterNode"})
        builder.add_edge("RouterNode", "MemoryRouterNode")
        builder.add_edge("MemoryRouterNode", "MCPToolNode")
        builder.add_conditional_edges("MCPToolNode", _mcp_branch, {"critical_error": "ResponseNode", "continue": "EvidencePacketNode"})
        builder.add_edge("EvidencePacketNode", "LLMGenerateNode")
        builder.add_edge("LLMGenerateNode", "ReviewPersistNode")
        builder.add_edge("ReviewPersistNode", "SafetyPostNode")
        builder.add_edge("SafetyPostNode", "HumanConfirmNode")
        builder.add_edge("HumanConfirmNode", "ResponseNode")
        builder.add_edge("ResponseNode", "TraceNode")
        builder.add_edge("TraceNode", END)
        return builder.compile(checkpointer=get_langgraph_checkpointer())

    def _response(self, state: AgentState) -> dict:
        if state.get("__interrupt__"):
            return self._interrupted_response(state)
        response = dict(state.get("final_response") or {})
        response["trace_id"] = state.get("trace_id")
        response["thread_id"] = state.get("thread_id")
        response["graph_node_path"] = state.get("graph_node_path", [])
        response["evidence_packet_id"] = (state.get("evidence_packet") or {}).get("packet_id")
        response["requires_human_confirm"] = bool(state.get("requires_human_confirm"))
        response["degraded"] = bool(state.get("degraded"))
        response["fallback_used"] = bool(state.get("fallback_used"))
        if state.get("evidence_packet"):
            response["evidence_packet"] = state["evidence_packet"]
        response["interrupted"] = bool(state.get("interrupted", False))
        response["human_confirmed"] = bool(state.get("human_confirmed", False))
        response["resume_action"] = state.get("resume_action")
        if state.get("input_payload", {}).get("pending_action_status"):
            response["pending_action_status"] = state["input_payload"]["pending_action_status"]
        if state.get("input_payload", {}).get("prescription_status"):
            response["prescription_status"] = state["input_payload"]["prescription_status"]
        return response

    def _interrupted_response(self, state: AgentState) -> dict:
        action = get_pending_action_by_trace(self.db, state["trace_id"])
        if action:
            mark_interrupted(self.db, action)
            self.db.commit()
        persisted = dict(state.get("input_payload", {}).get("persisted_review_result") or {})
        packet = state.get("evidence_packet") or persisted.get("evidence_packet") or {}
        graph_node_path = list(state.get("graph_node_path", []))
        if "HumanConfirmNode" not in graph_node_path:
            graph_node_path.append("HumanConfirmNode")
        response = {
            **persisted,
            "trace_id": state.get("trace_id"),
            "thread_id": state.get("thread_id"),
            "graph_node_path": graph_node_path,
            "evidence_packet_id": packet.get("packet_id"),
            "requires_human_confirm": True,
            "interrupted": True,
            "human_confirmed": False,
            "resume_action": None,
            "pending_action_status": action.status if action else None,
            "pending_action": serialize_pending_action(action) if action else None,
            "generated_response": state.get("generated_response"),
            "llm_result": state.get("llm_result"),
            "degraded": state.get("degraded", False),
            "fallback_used": state.get("fallback_used", False),
        }
        if packet:
            packet = dict(packet)
            packet["human_confirm"] = {**(packet.get("human_confirm") or {}), "required": True, "interrupted": True, "resumed": False}
            response["evidence_packet"] = packet
        return response


def _auth_branch(state: AgentState) -> str:
    return "blocked" if state.get("should_block") else "continue"


def _mcp_branch(state: AgentState) -> str:
    return "critical_error" if state.get("critical_error") else "continue"
