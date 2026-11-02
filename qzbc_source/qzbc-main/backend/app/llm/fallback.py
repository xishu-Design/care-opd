from __future__ import annotations

from typing import Any


def extract_source_ids(evidence_packet: dict[str, Any]) -> list[str]:
    source_ids: list[str] = []
    for item in evidence_packet.get("retrieved_evidence", []) or []:
        source_id = item.get("source_id") or item.get("source") or item.get("id")
        if source_id:
            source_ids.append(str(source_id))
    for item in evidence_packet.get("graph_reasoning_path", []) or []:
        source_id = item.get("source_id") or item.get("source")
        if source_id:
            source_ids.append(str(source_id))
    return sorted(set(source_ids))


def evidence_count(evidence_packet: dict[str, Any]) -> int:
    return len(evidence_packet.get("retrieved_evidence", []) or []) + len(evidence_packet.get("rule_findings", []) or [])


def compact_text(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."
