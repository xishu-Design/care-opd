from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, copy_state
from app.agents.state import AgentState
from app.services.conversation_context_service import ConversationContextService


class ConversationContextNode:
    node_name = "ConversationContextNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        context = ConversationContextService(self.db).build(
            role=state["role"],
            user_id=state["user_id"],
            current_message=state.get("input_text", ""),
        )
        state["conversation_context"] = context
        state["retrieval_query"] = context["retrieval_query"]
        return append_node(
            self.db,
            state,
            self.node_name,
            output_summary={
                "strategy": context["strategy"],
                "turn_count": context["turn_count"],
                "follow_up_detected": context["follow_up_detected"],
            },
        )
