from typing import Any

from app.agents.state import AgentState
from app.core.config import get_settings


def graph_config_for_state(state: AgentState) -> dict[str, Any]:
    return {"configurable": {"thread_id": state["thread_id"]}}


def ensure_langgraph_enabled() -> None:
    settings = get_settings()
    if settings.agent_orchestration_backend != "langgraph":
        raise RuntimeError("AGENT_ORCHESTRATION_BACKEND must be langgraph for Phase 6")
