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
from app.agents.nodes.safety_node import SafetyPostNode
from app.agents.nodes.trace_node import TraceNode
from app.agents.services.checkpoint_service import get_langgraph_checkpointer
from app.agents.services.graph_runtime import ensure_langgraph_enabled, graph_config_for_state
from app.agents.state import AgentState, initial_agent_state
from app.db.models import User
from app.mcp.contracts import new_request_id
from app.services.memory_consolidation_service import MemoryConsolidationAgent


class RoleAIChatGraph:
    graph_name = "RoleAIChatGraph"

    def __init__(self, db: Session):
        self.db = db

    def invoke(
        self,
        user: User,
        message: str,
        *,
        private_eval_evidence_control: dict | None = None,
        case_reasoning_context: dict | None = None,
    ) -> dict:
        if user.role not in {"doctor", "student"}:
            raise ValueError("RoleAIChatGraph only supports doctor and student")
        acknowledgement = None
        if private_eval_evidence_control is not None and case_reasoning_context is not None:
            raise ValueError("locked evidence control and case reasoning context are mutually exclusive")
        if private_eval_evidence_control is None and case_reasoning_context is None:
            acknowledgement = MemoryConsolidationAgent(self.db).preference_acknowledgement(
                owner_role=user.role,
                user_message=message,
            )
        if acknowledgement:
            return self._memory_preference_response(user, acknowledgement)
        ensure_langgraph_enabled()
        task_type = f"{user.role}_ai_chat"
        state = initial_agent_state(
            role=user.role,
            task_type=task_type,
            user_id=user.id,
            session_id=new_request_id(),
            request_path=(
                "/api/doctor/case-reasoning"
                if isinstance(case_reasoning_context, dict)
                else f"/api/{user.role}/ai-chat"
            ),
            graph_name=self.graph_name,
            input_text=message,
            input_payload=(
                {"case_reasoning_context": dict(case_reasoning_context)}
                if isinstance(case_reasoning_context, dict)
                else None
            ),
            thread_id=f"{user.role}:{user.id}:{user.username}:default_ai_chat",
            private_eval_evidence_control=private_eval_evidence_control,
        )
        result = self._graph().invoke(state, config=graph_config_for_state(state))
        return self._response(result)

    def _memory_preference_response(self, user: User, acknowledgement: str) -> dict:
        """Answer pure memory instructions without pretending they are medical Q&A."""
        visibility = "doctor_internal" if user.role == "doctor" else "student_learning"
        title = "已更新回答偏好" if user.role == "doctor" else "已更新学习偏好"
        generated = {
            "title": title,
            "summary": acknowledgement,
            "sections": [],
            "warnings": [],
            "next_actions": [],
            "disclaimer": "长期记忆仅用于调整回答方式，不作为医学证据。",
        }
        llm_result = {
            "provider": "memory_consolidation_policy",
            "model": "memory-canonicalizer-v1",
            "content": acknowledgement,
            "safety_flags": ["non_medical_preference_instruction"],
            "metadata": {"fallback_used": False, "degraded": False},
        }
        return {
            "answer": acknowledgement,
            "generated_response": generated,
            "llm_result": llm_result,
            "trace_id": new_request_id(),
            "thread_id": f"{user.role}:{user.id}:{user.username}:default_ai_chat",
            "graph_node_path": ["MemoryConsolidationAgent", "ResponseNode"],
            "degraded": False,
            "fallback_used": False,
            "evidence_packet": {
                "visibility": visibility,
                "structured_context": {"memory_instruction": True},
                "backend_summary": {"memory_backend": "persistent_memory"},
            },
        }

    def _graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("AuthNode", AuthNode(self.db))
        builder.add_node("RouterNode", RouterNode(self.db))
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
        builder.add_edge("RouterNode", "ConversationContextNode")
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


def _mcp_branch(state: AgentState) -> str:
    return "critical_error" if state.get("critical_error") else "continue"
