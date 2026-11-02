from __future__ import annotations

import json
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR
from app.data_pipeline.validators.source_registry_validator import SourceRegistryValidator
from app.knowledge.entity_quality import is_valid_tcm_herb_label
from app.lightrag_integration.schemas import GraphEdge, GraphNode, GraphResponse
from app.services.herb_incompatibility_service import expanded_incompatibility_pairs
from sqlalchemy.orm import Session


def local_graph(*, label: str, role: str, max_depth: int, max_nodes: int, content_dir: Path | str = DEFAULT_CONTENT_DATA_DIR, db: Session | None = None) -> GraphResponse:
    del db
    root = Path(content_dir).resolve()
    raw_nodes, raw_edges = _raw_graph_snapshot(str(root), role, _graph_file_signature(root))
    selected = _select_subgraph(raw_nodes, raw_edges, label, max_depth)
    truncated = len(selected) > max_nodes
    selected = selected[:max_nodes]
    selected_ids = {node.id for node in selected}
    edges = [edge for edge in raw_edges if edge.source in selected_ids and edge.target in selected_ids]
    return GraphResponse(
        nodes=selected,
        edges=edges,
        meta={
            "backend": "local_knowledge",
            "degraded": True,
            "fallback_used": True,
            "degraded_reason": "lightrag_unavailable",
            "truncated": truncated,
            "filtered_nodes": 0,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "snapshot_id": "",
            "query_correction": _query_correction(label),
        },
    )


@lru_cache(maxsize=8)
def _raw_graph_snapshot(root_value: str, role: str, signature: tuple[tuple[str, int, int], ...]) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
    del signature
    root = Path(root_value)
    registry = {
        item["source_id"]: item
        for item in SourceRegistryValidator().load_sources()
        if item.get("status") == "approved"
        or (item.get("status") == "reference_only" and item.get("allowed_use") == "reference_only")
    }
    raw_nodes: dict[str, GraphNode] = {}
    raw_edges: list[GraphEdge] = []
    for file_name, node_type, id_key, label_key in (
        ("herbs.json", "herb", "entry_id", "name"),
        ("formulas.json", "formula", "entry_id", "name"),
        ("sources.json", "source", "source_id", "source_title"),
    ):
        for item in _items(root / file_name):
            source_ids = item.get("source_ids") or [item.get("source_id", "")]
            source_ids = [str(value) for value in source_ids if value in registry]
            visibility = _visibility(item, source_ids, registry)
            if not _visible(source_ids, visibility, role, registry):
                continue
            node_id = str(item.get(id_key, ""))
            if not node_id:
                continue
            label_value = str(item.get("display_name") or item.get(label_key) or node_id)
            if node_type == "herb" and not is_valid_tcm_herb_label(label_value):
                continue
            # DragonTCM includes numbered condition labels (for example
            # "Cervical Cancer #18") alongside actual formula names. They may
            # remain traceable in the catalog but must not become clinical-claim
            # nodes in the visual knowledge graph.
            if node_type == "formula" and (
                "SRC_DRAGON_TCM" in source_ids
                or not _contains_chinese(str(item.get(label_key) or ""))
                or _is_numbered_condition_label(str(item.get(label_key) or ""))
            ):
                continue
            raw_nodes[node_id] = GraphNode(
                id=node_id,
                label=label_value,
                type=node_type,
                description=_description(item),
                source_ids=source_ids,
                visibility=visibility,
            )
            if node_type == "herb":
                _append_static_function_graph(
                    raw_nodes=raw_nodes,
                    raw_edges=raw_edges,
                    herb_id=node_id,
                    item=item,
                    source_ids=source_ids,
                    visibility=visibility,
                )
                _append_chpa_relation_graph(
                    raw_nodes=raw_nodes,
                    raw_edges=raw_edges,
                    herb_id=node_id,
                    item=item,
                    registry=registry,
                    role=role,
                )
            if node_type == "formula":
                _append_static_formula_graph(
                    raw_nodes=raw_nodes,
                    raw_edges=raw_edges,
                    formula_id=node_id,
                    item=item,
                    source_ids=source_ids,
                    visibility=visibility,
                )
    _append_incompatibility_graph(
        raw_nodes=raw_nodes,
        raw_edges=raw_edges,
        data_file=root / "herb_incompatibilities.json",
        registry=registry,
        role=role,
    )
    for item in _items(root / "relationships.json"):
        source_ids = [str(value) for value in item.get("source_ids", []) if value in registry]
        visibility = [str(value) for value in item.get("visibility", [])]
        if not _visible(source_ids, visibility, role, registry):
            continue
        raw_edges.append(
            GraphEdge(
                id=str(item.get("relation_id", "")),
                source=str(item.get("source_entity_id", "")),
                target=str(item.get("target_entity_id", "")),
                label=str(item.get("relation_type", "RELATED_TO")),
                source_ids=source_ids,
            )
        )
    return raw_nodes, _merge_parallel_edges(raw_edges)


def _graph_file_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    paths = [
        root / name
        for name in ("herbs.json", "formulas.json", "sources.json", "relationships.json", "herb_incompatibilities.json")
    ]
    registry_dir = SourceRegistryValidator().registry_dir
    paths.extend(
        registry_dir / name
        for name in ("data_sources_registry.yaml", "approved_sources.yaml", "rejected_sources.yaml")
    )
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
        if path.exists()
    )


def local_labels(*, query: str, role: str, limit: int, content_dir: Path | str = DEFAULT_CONTENT_DATA_DIR, db: Session | None = None) -> list[dict[str, Any]]:
    del db
    root = Path(content_dir).resolve()
    raw_nodes, _ = _raw_graph_snapshot(str(root), role, _graph_file_signature(root))
    search_terms = _graph_search_terms(query)
    return [
        {"label": node.label, "entity_type": node.type, "source_ids": node.source_ids, "visibility": node.visibility}
        for node in sorted(raw_nodes.values(), key=_node_sort_key)
        if not query or any(term.lower() in node.label.lower() for term in search_terms)
    ][:limit]


def _select_subgraph(nodes: dict[str, GraphNode], edges: list[GraphEdge], label: str, max_depth: int) -> list[GraphNode]:
    if not label:
        tcm_node_ids = {
            node_id
            for edge in edges
            if edge.label in {
                "HAS_TCM_FUNCTION",
                "ASSOCIATED_WITH_TCM_SYMPTOM",
                "ASSOCIATED_WITH_TCM_SYNDROME",
                "HAS_CHPA_EFFICACY",
                "ASSOCIATED_WITH_CHPA_SYMPTOM",
                "TRADITIONAL_INCOMPATIBILITY_18",
                "TRADITIONAL_INCOMPATIBILITY_19",
            }
            for node_id in (edge.source, edge.target)
        }
        degree: dict[str, int] = {}
        for edge in edges:
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1
        ranked = sorted(nodes.values(), key=lambda node: (node.id not in tcm_node_ids, -degree.get(node.id, 0), _node_sort_key(node)))
        # Keep the overview graph connected.  A plain degree sort previously
        # sliced the first N isolated nodes, which is why the UI showed dots
        # without lines.
        ordered_ids: list[str] = []
        selected: set[str] = set()
        for node in ranked:
            if node.id in selected:
                continue
            selected.add(node.id)
            ordered_ids.append(node.id)
            neighbors = {
                edge.target if edge.source == node.id else edge.source
                for edge in edges
                if edge.source == node.id or edge.target == node.id
            }
            for neighbor_id in sorted((value for value in neighbors if value in nodes), key=lambda value: _node_sort_key(nodes[value]))[:3]:
                if neighbor_id not in selected:
                    selected.add(neighbor_id)
                    ordered_ids.append(neighbor_id)
        return [nodes[node_id] for node_id in ordered_ids]
    search_terms = _graph_search_terms(label)
    exact = {node.id for node in nodes.values() if any(node.label.lower() == term.lower() for term in search_terms)}
    seeds = exact or {
        node.id
        for node in nodes.values()
        if any(term.lower() in node.label.lower() for term in search_terms)
    }
    selected = set(seeds)
    frontier = set(seeds)
    ordered_ids = sorted(seeds, key=lambda node_id: _node_sort_key(nodes[node_id]))
    for _ in range(max_depth):
        expandable = {node_id for node_id in frontier if node_id in nodes and nodes[node_id].type != "source"}
        adjacent = {edge.target for edge in edges if edge.source in expandable} | {edge.source for edge in edges if edge.target in expandable}
        frontier = adjacent - selected
        ordered_ids.extend(sorted((node_id for node_id in frontier if node_id in nodes), key=lambda node_id: _node_sort_key(nodes[node_id])))
        selected.update(frontier)
    return [nodes[node_id] for node_id in ordered_ids if node_id in nodes]


def _node_sort_key(node: GraphNode) -> tuple[int, str]:
    priority = {"herb": 0, "efficacy": 1, "formula": 2, "syndrome": 3, "symptom": 4, "source": 5, "case": 6, "concept": 7}
    return priority.get(node.type, 99), node.label


def _graph_search_terms(label: str) -> list[str]:
    terms = [label.strip()]
    expansions = {
        "眼睛疲劳": ("眼疲劳", "视疲劳"),
        "眼疲劳": ("视疲劳",),
        "视觉疲劳": ("视疲劳",),
        "肠胃虚弱": ("胃肠虚弱", "脾胃虚弱"),
        # “麝虫” is a common character-input mistake for the catalogued herb
        # “麝香”. Keep the original query visible in the UI and expose the
        # correction in graph metadata; do not silently invent a new herb.
        "麝虫": ("麝香",),
    }
    for term, values in expansions.items():
        if term in label:
            terms.extend(values)
    return list(dict.fromkeys(value for value in terms if value))


def _query_correction(label: str) -> str:
    return "麝香" if label.strip() == "麝虫" else ""


def _is_numbered_condition_label(value: str) -> bool:
    return bool(re.search(r"#\s*\d+\s*$", value))


def _contains_chinese(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def _visibility(item: dict[str, Any], source_ids: list[str], registry: dict[str, dict[str, Any]]) -> list[str]:
    value = item.get("visibility")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value:
        return [str(entry) for entry in value]
    result = []
    if source_ids and all(registry[source_id].get("doctor_evidence_allowed", True) for source_id in source_ids):
        result.append("doctor")
    if source_ids and all(registry[source_id].get("student_visible", True) for source_id in source_ids):
        result.extend(("student", "student_learning"))
    return result


def _visible(source_ids: list[str], visibility: list[str], role: str, registry: dict[str, dict[str, Any]]) -> bool:
    if not source_ids or not visibility:
        return False
    if role == "doctor":
        return "doctor" in visibility and all(registry[source_id].get("doctor_evidence_allowed", True) for source_id in source_ids)
    return bool({"student", "student_learning"} & set(visibility)) and all(registry[source_id].get("student_visible", True) for source_id in source_ids)


def _description(item: dict[str, Any]) -> str:
    for key in ("description", "standard_analysis", "usage_note", "license_note"):
        if item.get(key):
            return str(item[key])
    return ""


def _append_static_function_graph(
    *,
    raw_nodes: dict[str, GraphNode],
    raw_edges: list[GraphEdge],
    herb_id: str,
    item: dict[str, Any],
    source_ids: list[str],
    visibility: list[str],
) -> None:
    """Expose only structured catalog functions, never generated case/Q&A text."""
    for function in item.get("display_functions") or item.get("functions") or []:
        label = str(function).strip()
        if not label:
            continue
        node_id = f"EFF_{hashlib.sha256(label.encode('utf-8')).hexdigest()[:20].upper()}"
        raw_nodes.setdefault(
            node_id,
            GraphNode(
                id=node_id,
                label=label,
                type="efficacy",
                description="药材目录的来源原文功效字段。",
                source_ids=list(source_ids),
                visibility=list(visibility),
            ),
        )
        raw_edges.append(
            GraphEdge(
                id=f"REL_{hashlib.sha256(f'{herb_id}|{node_id}|HAS_TCM_FUNCTION'.encode('utf-8')).hexdigest()[:20].upper()}",
                source=herb_id,
                target=node_id,
                label="HAS_TCM_FUNCTION",
                source_ids=list(source_ids),
            )
        )


def _append_chpa_relation_graph(
    *,
    raw_nodes: dict[str, GraphNode],
    raw_edges: list[GraphEdge],
    herb_id: str,
    item: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    role: str,
) -> None:
    """Project CHPA statistical associations with source-specific provenance."""
    source_ids = ["SRC_DRYAD_TCM_NETWORK"] if "SRC_DRYAD_TCM_NETWORK" in registry else []
    visibility = _visibility(item, source_ids, registry)
    if not _visible(source_ids, visibility, role, registry):
        return
    for values, node_type, relation_type, prefix, description in (
        (
            item.get("chpa_efficacies") or [],
            "efficacy",
            "HAS_CHPA_EFFICACY",
            "EFF",
            "CHPA/Dryad 草药—功效统计关联；中文术语由本地模型辅助翻译，待医学复核。",
        ),
        (
            item.get("chpa_symptoms") or [],
            "symptom",
            "ASSOCIATED_WITH_CHPA_SYMPTOM",
            "SYM",
            "CHPA/Dryad 草药—症状统计关联（P<0.05）；中文术语由本地模型辅助翻译，待医学复核。",
        ),
    ):
        for value in values:
            label = str(value).strip()
            if not label or not _contains_chinese(label):
                continue
            node_id = f"{prefix}_{hashlib.sha256(label.encode('utf-8')).hexdigest()[:20].upper()}"
            current = raw_nodes.get(node_id)
            if current is None:
                raw_nodes[node_id] = GraphNode(
                    id=node_id,
                    label=label,
                    type=node_type,
                    description=description,
                    source_ids=list(source_ids),
                    visibility=list(visibility),
                )
            else:
                current.source_ids = sorted(set(current.source_ids) | set(source_ids))
                current.visibility = sorted(set(current.visibility) | set(visibility))
            raw_edges.append(
                GraphEdge(
                    id=f"REL_{hashlib.sha256(f'{herb_id}|{node_id}|{relation_type}'.encode('utf-8')).hexdigest()[:20].upper()}",
                    source=herb_id,
                    target=node_id,
                    label=relation_type,
                    source_ids=list(source_ids),
                )
            )


def _append_static_formula_graph(
    *,
    raw_nodes: dict[str, GraphNode],
    raw_edges: list[GraphEdge],
    formula_id: str,
    item: dict[str, Any],
    source_ids: list[str],
    visibility: list[str],
) -> None:
    """Project source-backed formula efficacy and syndrome fields into graph edges."""
    herb_ids_by_label = {node.label: node_id for node_id, node in raw_nodes.items() if node.type == "herb"}
    for component in item.get("components") or []:
        if not isinstance(component, dict):
            continue
        label = str(component.get("normalized_name") or component.get("herb_name") or "").strip()
        if not label or not _contains_chinese(label):
            continue
        herb_id = herb_ids_by_label.get(label)
        if herb_id is None:
            herb_id = f"HERB_COMPONENT_{hashlib.sha256(label.encode('utf-8')).hexdigest()[:20].upper()}"
            raw_nodes.setdefault(
                herb_id,
                GraphNode(
                    id=herb_id,
                    label=label,
                    type="herb",
                    description="方剂目录中记录的组成药材。",
                    source_ids=list(source_ids),
                    visibility=list(visibility),
                ),
            )
        raw_edges.append(
            GraphEdge(
                id=f"REL_{hashlib.sha256(f'{formula_id}|{herb_id}|CONTAINS'.encode('utf-8')).hexdigest()[:20].upper()}",
                source=formula_id,
                target=herb_id,
                label="CONTAINS",
                source_ids=list(source_ids),
            )
        )
    for values, node_type, relation_type, prefix, description in (
        (item.get("display_functions") or item.get("functions") or [], "efficacy", "HAS_TCM_FUNCTION", "EFF", "方剂目录的来源功效字段。"),
        (item.get("syndromes") or [], "syndrome", "ASSOCIATED_WITH_TCM_SYNDROME", "SYN", "方剂目录的来源中医证候字段。"),
        (item.get("symptoms") or [], "symptom", "ASSOCIATED_WITH_TCM_SYMPTOM", "SYM", "方剂目录的来源症状或疾病术语字段。"),
    ):
        for value in values:
            label = str(value).strip()
            if not label or not _contains_chinese(label):
                continue
            node_id = f"{prefix}_{hashlib.sha256(label.encode('utf-8')).hexdigest()[:20].upper()}"
            raw_nodes.setdefault(
                node_id,
                GraphNode(
                    id=node_id,
                    label=label,
                    type=node_type,
                    description=description,
                    source_ids=list(source_ids),
                    visibility=list(visibility),
                ),
            )
            raw_edges.append(
                GraphEdge(
                    id=f"REL_{hashlib.sha256(f'{formula_id}|{node_id}|{relation_type}'.encode('utf-8')).hexdigest()[:20].upper()}",
                    source=formula_id,
                    target=node_id,
                    label=relation_type,
                    source_ids=list(source_ids),
                )
            )


def _append_incompatibility_graph(
    *,
    raw_nodes: dict[str, GraphNode],
    raw_edges: list[GraphEdge],
    data_file: Path,
    registry: dict[str, dict[str, Any]],
    role: str,
) -> None:
    """Project source-bound traditional incompatibilities as herb-herb edges."""

    if not data_file.exists():
        return
    source_id = "SRC_DB4403_T142_2021"
    if source_id not in registry:
        return
    source_ids = [source_id]
    visibility = ["doctor", "student", "student_learning"]
    if not _visible(source_ids, visibility, role, registry):
        return

    for pair in expanded_incompatibility_pairs(data_file=data_file):
        herb_a = str(pair.get("herb_a") or "").strip()
        herb_b = str(pair.get("herb_b") or "").strip()
        if not herb_a or not herb_b:
            continue
        source_node = _herb_node_for_label(raw_nodes, herb_a, source_ids, visibility)
        target_node = _herb_node_for_label(raw_nodes, herb_b, source_ids, visibility)
        relation_type = str(pair.get("relation_type") or "TRADITIONAL_INCOMPATIBILITY")
        identity = f"{source_node}|{target_node}|{relation_type}|{pair.get('constraint_id', '')}"
        raw_edges.append(
            GraphEdge(
                id=f"REL_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20].upper()}",
                source=source_node,
                target=target_node,
                label=relation_type,
                source_ids=source_ids,
            )
        )


def _herb_node_for_label(
    raw_nodes: dict[str, GraphNode],
    label: str,
    source_ids: list[str],
    visibility: list[str],
) -> str:
    matches = [node for node in raw_nodes.values() if node.type == "herb" and node.label == label]
    if matches:
        return sorted(matches, key=lambda node: node.id)[0].id
    node_id = f"HERB_STANDARD_{hashlib.sha256(label.encode('utf-8')).hexdigest()[:20].upper()}"
    raw_nodes[node_id] = GraphNode(
        id=node_id,
        label=label,
        type="herb",
        description="DB4403/T 142—2021 附录 C 传统配伍禁忌表中的药材术语。",
        source_ids=list(source_ids),
        visibility=list(visibility),
    )
    return node_id


def _merge_parallel_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    """Expose one visual relation while retaining the number of supporting records."""
    merged: dict[tuple[str, str, str], GraphEdge] = {}
    for edge in edges:
        key = (edge.source, edge.target, edge.label)
        current = merged.get(key)
        if current is None:
            merged[key] = GraphEdge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                label=edge.label,
                source_ids=list(edge.source_ids),
                evidence_count=edge.evidence_count,
            )
            continue
        current.source_ids = sorted(set(current.source_ids) | set(edge.source_ids))
        current.evidence_count += edge.evidence_count
    return list(merged.values())


def _items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("items", []) if isinstance(data, dict) and isinstance(data.get("items"), list) else []
