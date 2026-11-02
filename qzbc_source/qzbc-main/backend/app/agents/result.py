from typing import Any

from pydantic import BaseModel, Field


class AgentInvocationResult(BaseModel):
    trace_id: str | None = None
    thread_id: str | None = None
    graph_node_path: list[str] = Field(default_factory=list)
    evidence_packet_id: str | None = None
    requires_human_confirm: bool = False
    degraded: bool = False
    fallback_used: bool = False
    data: dict[str, Any] = Field(default_factory=dict)

