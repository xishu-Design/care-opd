from __future__ import annotations

from typing import Any


def validate_source_catalog_traceability(processed_data: dict[str, list[dict[str, Any]]], source_catalog: dict[str, Any]) -> list[dict[str, str]]:
    source_ids = {source.get("source_id") for source in source_catalog.get("sources", [])}
    issues: list[dict[str, str]] = []
    for section, items in processed_data.items():
        for item in items:
            source_id = item.get("source_id")
            if not source_id:
                issues.append({"section": section, "code": "MISSING_SOURCE_ID", "message": "Item has no source_id"})
            elif source_id not in source_ids:
                issues.append({"section": section, "code": "SOURCE_NOT_IN_CATALOG", "message": f"{source_id} not found in source_catalog"})
    return issues

