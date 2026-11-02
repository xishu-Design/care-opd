from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, copy_state
from app.agents.state import AgentState
from app.memory.memory_router import MemoryRouteRequest, MemoryRouter


class MemoryRouterNode:
    node_name = "MemoryRouterNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        plan = MemoryRouter().plan(
            MemoryRouteRequest(
                role=state["role"],
                task_type=state["task_type"],
                risk_level="medium" if state["role"] == "doctor" else None,
                patient_id=state.get("patient_id"),
                prescription_id=state.get("prescription_id"),
                learning_case_id=state.get("learning_case_id"),
            )
        )
        state["memory_plan"] = [item.model_dump() for item in plan.memory_plan]
        return append_node(
            self.db,
            state,
            self.node_name,
            output_summary={"tools": [item["tool_name"] for item in state["memory_plan"]]},
        )

