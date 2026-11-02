from __future__ import annotations

from app.data_pipeline.normalizers.text_normalizer import normalize_text


def normalize_entity_name(value: object) -> str:
    text = normalize_text(value)
    return text.replace(" ", "")


def unique_normalized_names(values: list[object]) -> list[dict[str, str]]:
    seen: set[str] = set()
    items: list[dict[str, str]] = []
    for value in values:
        original = normalize_text(value)
        normalized = normalize_entity_name(original)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        items.append({"original_name": original, "normalized_name": normalized, "name": normalized})
    return items
