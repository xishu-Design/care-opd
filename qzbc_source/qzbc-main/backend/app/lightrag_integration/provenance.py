from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_pipeline.validators.source_registry_validator import SourceRegistryValidator
from app.db.models import LightRAGDocManifest
from app.lightrag_integration.document_builder import load_lightrag_docs
from app.lightrag_integration.schemas import GraphEdge, GraphNode, GraphResponse


SOURCE_ID_RE = re.compile(r"SRC_[A-Z0-9_]+")
ALLOWED_RELATIONS = {"CONTAINS", "DERIVED_FROM", "REFERENCES", "DESCRIBES"}


class ProvenanceResolver:
    """Maps LightRAG graph output back to approved GuardTCM sources."""

    def __init__(
        self,
        docs_path: Path | str | None = None,
        *,
        db: Session | None = None,
        snapshot_id: str = "",
    ) -> None:
        sources = SourceRegistryValidator().load_sources()
        self.policies = {str(item["source_id"]): item for item in sources if item.get("status") == "approved"}
        self.docs = load_lightrag_docs(docs_path)
        self.doc_map = {doc.doc_id: doc for doc in self.docs}
        self.entry_map = {doc.entry_id: doc for doc in self.docs}
        self.manifest_doc_map: dict[str, LightRAGDocManifest] = {}
        self.manifest_entry_map: dict[str, LightRAGDocManifest] = {}
        if db is not None:
            statement = select(LightRAGDocManifest).order_by(LightRAGDocManifest.created_at.desc())
            if snapshot_id:
                statement = statement.where(LightRAGDocManifest.snapshot_id == snapshot_id)
            for manifest in db.scalars(statement).all():
                # The newest manifest wins when a development database has no
                # ACTIVE snapshot yet but contains several verified runs.
                self.manifest_doc_map.setdefault(manifest.doc_id, manifest)
                for entry_id in manifest.entry_ids_json or []:
                    self.manifest_entry_map.setdefault(str(entry_id), manifest)

    def resolve_graph(
        self,
        payload: Any,
        *,
        role: str,
        max_depth: int,
        max_nodes: int,
        snapshot_id: str = "",
    ) -> GraphResponse:
        raw_nodes = _list_at(payload, "nodes", "vertices", "entities")
        raw_edges = _list_at(payload, "edges", "links", "relationships")
        nodes: list[GraphNode] = []
        filtered = 0
        for index, raw in enumerate(raw_nodes):
            node = self._node(raw, index)
            if node is None or not self._visible(node.source_ids, node.visibility, role):
                filtered += 1
                continue
            nodes.append(node)
        truncated = len(nodes) > max_nodes
        nodes = nodes[:max_nodes]
        node_ids = {node.id for node in nodes}
        edges: list[GraphEdge] = []
        for index, raw in enumerate(raw_edges):
            edge = self._edge(raw, index)
            if edge and edge.source in node_ids and edge.target in node_ids and self._sources_approved(edge.source_ids):
                edges.append(edge)
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            meta={
                "backend": "lightrag",
                "degraded": False,
                "fallback_used": False,
                "truncated": truncated,
                "filtered_nodes": filtered,
                "max_depth": max_depth,
                "max_nodes": max_nodes,
                "snapshot_id": snapshot_id,
            },
        )

    def resolve_labels(self, payload: Any, *, role: str, query: str, limit: int) -> list[dict[str, Any]]:
        raw_items = _list_at(payload, "items", "labels", "data", "results")
        labels: list[dict[str, Any]] = []
        for raw in raw_items:
            label = str(raw.get("label") or raw.get("name") or raw) if isinstance(raw, dict) else str(raw)
            if query and query.lower() not in label.lower():
                continue
            source_ids, visibility = self._provenance(raw, label)
            if not self._visible(source_ids, visibility, role):
                continue
            labels.append({"label": label, "entity_type": _entity_type(raw, label), "source_ids": source_ids, "visibility": visibility})
            if len(labels) >= limit:
                break
        return labels

    def _node(self, raw: Any, index: int) -> GraphNode | None:
        item = raw if isinstance(raw, dict) else {"label": str(raw)}
        properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        merged = {**properties, **item}
        labels = merged.get("labels") if isinstance(merged.get("labels"), list) else []
        label = str(merged.get("label") or merged.get("name") or merged.get("entity_name") or merged.get("entity_id") or (labels[0] if labels else "")).strip()
        if not label:
            return None
        node_id = str(merged.get("id") or merged.get("entity_id") or label or f"node-{index}")
        source_ids, visibility = self._provenance(merged, label)
        return GraphNode(
            id=node_id,
            label=label,
            type=_entity_type(merged, label),
            description=str(merged.get("description") or merged.get("content") or ""),
            source_ids=source_ids,
            visibility=visibility,
        )

    def _edge(self, raw: Any, index: int) -> GraphEdge | None:
        if not isinstance(raw, dict):
            return None
        properties = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
        merged = {**properties, **raw}
        source = str(merged.get("source") or merged.get("src") or merged.get("from") or merged.get("source_id") or "")
        target = str(merged.get("target") or merged.get("dst") or merged.get("to") or merged.get("target_id") or "")
        if not source or not target:
            return None
        source_ids, _ = self._provenance(merged, "")
        label = str(merged.get("label") or merged.get("relation") or merged.get("type") or "").upper().replace(" ", "_")
        if label not in ALLOWED_RELATIONS:
            return None
        return GraphEdge(
            id=str(merged.get("id") or f"edge-{index}-{source}-{target}"),
            source=source,
            target=target,
            label=label,
            source_ids=source_ids,
        )

    def _provenance(self, raw: Any, label: str) -> tuple[list[str], list[str]]:
        text = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
        source_ids = sorted(source_id for source_id in set(SOURCE_ID_RE.findall(text)) if source_id in self.policies)
        doc = None
        if isinstance(raw, dict):
            keys = [raw.get(key) for key in ("doc_id", "document_id", "entry_id", "source_id")]
            doc = next((self.doc_map.get(str(key)) or self.entry_map.get(str(key)) for key in keys if key), None)
            if doc is None:
                file_paths = str(raw.get("file_path") or "").split("<SEP>")
                doc = next((self.doc_map.get(path.strip()) for path in file_paths if path.strip()), None)
        if doc is None and label:
            doc = next((item for item in self.docs if item.title == label or item.entry_id == label), None)
        manifest = None
        if doc is None and isinstance(raw, dict):
            keys = [raw.get(key) for key in ("doc_id", "document_id", "entry_id", "source_id")]
            manifest = next(
                (
                    self.manifest_doc_map.get(str(key)) or self.manifest_entry_map.get(str(key))
                    for key in keys
                    if key
                ),
                None,
            )
            if manifest is None:
                file_paths = str(raw.get("file_path") or "").split("<SEP>")
                manifest = next(
                    (
                        self.manifest_doc_map.get(path.strip()) or self.manifest_entry_map.get(path.strip())
                        for path in file_paths
                        if path.strip()
                    ),
                    None,
                )
        if doc:
            doc_source = str(doc.metadata.get("source_id", ""))
            if doc_source in self.policies and doc_source not in source_ids:
                source_ids.append(doc_source)
        if manifest:
            for manifest_source in manifest.source_ids_json or []:
                source_id = str(manifest_source)
                if source_id in self.policies and source_id not in source_ids:
                    source_ids.append(source_id)
        visibility = _string_list(raw.get("visibility")) if isinstance(raw, dict) else []
        if not visibility and doc:
            visibility = _string_list(doc.metadata.get("visibility"))
        if not visibility and manifest:
            visibility = _string_list(manifest.visibility_json)
        if not visibility and source_ids:
            visibility = _policy_visibility([self.policies[source_id] for source_id in source_ids])
        return sorted(source_ids), sorted(set(visibility))

    def _sources_approved(self, source_ids: list[str]) -> bool:
        return bool(source_ids) and all(source_id in self.policies for source_id in source_ids)

    def _visible(self, source_ids: list[str], visibility: list[str], role: str) -> bool:
        if not self._sources_approved(source_ids) or not visibility:
            return False
        if role == "doctor":
            return "doctor" in visibility and all(self.policies[source_id].get("doctor_evidence_allowed", True) for source_id in source_ids)
        if role == "student":
            return bool({"student", "student_learning"} & set(visibility)) and all(self.policies[source_id].get("student_visible", True) for source_id in source_ids)
        return False


def _list_at(payload: Any, *keys: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _list_at(value, *keys)
            if nested:
                return nested
    data = payload.get("data")
    return _list_at(data, *keys) if data is not payload else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _policy_visibility(policies: list[dict[str, Any]]) -> list[str]:
    result: set[str] = set()
    if policies and all(policy.get("doctor_evidence_allowed", True) for policy in policies):
        result.add("doctor")
    if policies and all(policy.get("student_visible", True) for policy in policies):
        result.update(("student", "student_learning"))
    return sorted(result)


def _entity_type(raw: Any, label: str) -> str:
    item = raw if isinstance(raw, dict) else {}
    raw_type = str(item.get("entity_type") or item.get("type") or item.get("category") or "").lower()
    aliases = {"herb": "herb", "中药": "herb", "formula": "formula", "方剂": "formula", "source": "source", "来源": "source", "case": "case", "案例": "case", "concept": "concept", "概念": "concept"}
    for token, canonical in aliases.items():
        if token in raw_type:
            return canonical
    if label.startswith("SRC_"):
        return "source"
    return "unknown"
