from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LightRAGDoc(BaseModel):
    doc_id: str
    entry_id: str
    entry_type: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LightRAGQueryResult(BaseModel):
    content: str
    source_id: str = ""
    entry_id: str = ""
    entry_type: str = ""
    title: str = ""
    score: float = 0.0
    graph_entities: list[dict[str, Any]] = Field(default_factory=list)
    graph_relations: list[dict[str, Any]] = Field(default_factory=list)
    graph_path: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class LightRAGQueryResponse(BaseModel):
    success: bool
    query: str
    results: list[LightRAGQueryResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class GraphNode(BaseModel):
    id: str
    label: str
    type: str = "unknown"
    description: str = ""
    source_ids: list[str] = Field(default_factory=list)
    visibility: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""
    source_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 1


class GraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
