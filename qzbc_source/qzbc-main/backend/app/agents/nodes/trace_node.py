from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, copy_state, evidence_packet_id
from app.agents.state import AgentState
from app.harness.trace import finish_trace


class TraceNode:
    node_name = "TraceNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        state = append_node(
            self.db,
            state,
            self.node_name,
            output_summary={
                "status": "blocked" if state.get("should_block") else "success",
                "evidence_packet_id": evidence_packet_id(state),
            },
        )
        finish_trace(
            self.db,
            state.get("trace_id"),
            status="blocked" if state.get("should_block") else "success",
            degraded=state.get("degraded", False),
            fallback_used=state.get("fallback_used", False),
            metadata={
                "evidence_packet_id": evidence_packet_id(state),
                "graph_node_path": state.get("graph_node_path", []),
                "graph_name": state.get("graph_name"),
                "graph_version": state.get("graph_version"),
                "thread_id": state.get("thread_id"),
            },
        )
        self.db.commit()
        return state
