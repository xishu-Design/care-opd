from __future__ import annotations

from typing import Any


def merge_by_key(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        item_key = str(item.get(key, ""))
        if not item_key:
            continue
        if item_key not in merged:
            merged[item_key] = dict(item)
            continue
        _merge_item(merged[item_key], item)
    return [merged[key] for key in sorted(merged)]


def _merge_item(target: dict[str, Any], item: dict[str, Any]) -> None:
    for field in ("source_ids", "aliases", "keywords"):
        target[field] = _unique([*target.get(field, []), *item.get(field, [])])
    for field in ("source_records", "components"):
        target[field] = _unique_dicts([*target.get(field, []), *item.get(field, [])])


def _unique(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _unique_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for value in values:
        marker = str(sorted(value.items()))
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result

