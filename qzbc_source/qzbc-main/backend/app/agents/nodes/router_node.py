from sqlalchemy.orm import Session

from app.agents.nodes.common import append_node, copy_state
from app.agents.state import AgentState


class RouterNode:
    node_name = "RouterNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        if state["role"] == "doctor" and state["task_type"] == "prescription_review":
            route = "doctor_prescription_review"
        elif state["role"] == "patient" and state["task_type"] == "patient_education":
            route = "patient_public_education"
        elif state["role"] == "student" and state["task_type"] == "student_training":
            route = "student_case_training"
        elif state["role"] == "doctor" and state["task_type"] == "doctor_ai_chat":
            route = "doctor_professional_ai_chat"
        elif state["role"] == "student" and state["task_type"] == "student_ai_chat":
            route = "student_tutoring_ai_chat"
        else:
            route = "unsupported_agent_route"
            state["should_block"] = True
            state["should_continue"] = False

        state["route"] = route
        return append_node(
            self.db,
            state,
            self.node_name,
            output_summary={"route": route, "should_block": state.get("should_block", False)},
        )
