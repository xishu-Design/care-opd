from __future__ import annotations

import re
from collections.abc import Iterable


_NEGATIVE_ALLERGY_SUMMARIES = (
    "无过敏史",
    "无明确过敏史",
    "无药物过敏史",
    "无明确药物过敏史",
    "未发现过敏",
    "否认过敏史",
)


def allergy_conflicts_for_herbs(allergy_summary: str | None, herb_names: Iterable[str]) -> list[str]:
    summary = _compact(allergy_summary or "")
    negative_summaries = {_compact(value) for value in _NEGATIVE_ALLERGY_SUMMARIES}
    if not summary or summary in negative_summaries:
        return []

    conflicts: list[str] = []
    for raw_name in herb_names:
        herb_name = str(raw_name or "").strip()
        normalized = _compact(herb_name)
        if not normalized or normalized not in summary:
            continue
        if _is_explicitly_negated(summary, normalized):
            continue
        if herb_name not in conflicts:
            conflicts.append(herb_name)
    return conflicts


def _compact(value: str) -> str:
    return re.sub(r"[\s，,、；;。.!！：:（）()\[\]【】]", "", value).lower()


def _is_explicitly_negated(summary: str, herb: str) -> bool:
    escaped = re.escape(herb)
    patterns = (
        rf"无(?:明确)?(?:对)?{escaped}(?:药物)?过敏",
        rf"(?:对)?{escaped}(?:不|无)过敏",
        rf"否认(?:对)?{escaped}过敏",
    )
    return any(re.search(pattern, summary) for pattern in patterns)
