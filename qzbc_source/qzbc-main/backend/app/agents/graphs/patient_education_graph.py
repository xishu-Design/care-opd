from sqlalchemy.orm import Session
from langgraph.graph import END, START, StateGraph

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
from app.agents.services.checkpoint_service import get_langgraph_checkpointer
from app.agents.services.graph_runtime import ensure_langgraph_enabled, graph_config_for_state
from app.agents.state import AgentState, initial_agent_state
from app.db.models import User
from app.mcp.contracts import new_request_id


class PatientEducationGraph:
    graph_name = "PatientEducationGraph"

    def __init__(self, db: Session):
        self.db = db

    def invoke(
        self,
        patient: User,
        message: str,
        *,
        private_eval_evidence_control: dict | None = None,
    ) -> dict:
        ensure_langgraph_enabled()
        state = initial_agent_state(
            role="patient",
            task_type="patient_education",
            user_id=patient.id,
            session_id=new_request_id(),
            request_path="/api/patient/education-chat",
            graph_name=self.graph_name,
            input_text=message,
            patient_id=patient.id,
            thread_id=f"patient:{patient.id}:{patient.username}:default_education_chat",
            private_eval_evidence_control=private_eval_evidence_control,
        )
        result = self._graph().invoke(state, config=graph_config_for_state(state))
        return self._response(result)

    def _graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("AuthNode", AuthNode(self.db))
        builder.add_node("RouterNode", RouterNode(self.db))
        builder.add_node("SafetyPreNode", SafetyPreNode(self.db))
        builder.add_node("ConversationContextNode", ConversationContextNode(self.db))
        builder.add_node("MemoryRouterNode", MemoryRouterNode(self.db))
        builder.add_node("MCPToolNode", MCPToolNode(self.db))
        builder.add_node("EvidencePacketNode", EvidencePacketNode(self.db))
        builder.add_node("LLMGenerateNode", LLMGenerateNode(self.db))
        builder.add_node("SafetyPostNode", SafetyPostNode(self.db))
        builder.add_node("ResponseNode", ResponseNode(self.db))
        builder.add_node("TraceNode", TraceNode(self.db))

        builder.add_edge(START, "AuthNode")
        builder.add_conditional_edges("AuthNode", _auth_branch, {"blocked": "ResponseNode", "continue": "RouterNode"})
        builder.add_edge("RouterNode", "SafetyPreNode")
        builder.add_conditional_edges("SafetyPreNode", _safety_branch, {"blocked": "ResponseNode", "continue": "ConversationContextNode"})
        builder.add_edge("ConversationContextNode", "MemoryRouterNode")
        builder.add_edge("MemoryRouterNode", "MCPToolNode")
        builder.add_conditional_edges("MCPToolNode", _mcp_branch, {"critical_error": "ResponseNode", "continue": "EvidencePacketNode"})
        builder.add_edge("EvidencePacketNode", "LLMGenerateNode")
        builder.add_edge("LLMGenerateNode", "SafetyPostNode")
        builder.add_edge("SafetyPostNode", "ResponseNode")
        builder.add_edge("ResponseNode", "TraceNode")
        builder.add_edge("TraceNode", END)
        return builder.compile(checkpointer=get_langgraph_checkpointer())

    def _response(self, state: AgentState) -> dict:
        response = dict(state.get("final_response") or {})
        response["trace_id"] = state.get("trace_id")
        response["thread_id"] = state.get("thread_id")
        response["graph_node_path"] = state.get("graph_node_path", [])
        response["evidence_packet_id"] = (state.get("evidence_packet") or {}).get("packet_id")
        response["degraded"] = bool(state.get("degraded"))
        response["fallback_used"] = bool(state.get("fallback_used"))
        if state.get("evidence_packet"):
            response["evidence_packet"] = state["evidence_packet"]
        return response


def _auth_branch(state: AgentState) -> str:
    return "blocked" if state.get("should_block") else "continue"


def _safety_branch(state: AgentState) -> str:
    return "blocked" if state.get("should_block") else "continue"


def _mcp_branch(state: AgentState) -> str:
    return "critical_error" if state.get("critical_error") else "continue"
