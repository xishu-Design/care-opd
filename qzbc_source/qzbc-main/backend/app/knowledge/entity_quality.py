from __future__ import annotations

import re


_CHINESE_NAME = re.compile(r"^[\u3400-\u9fff]{2,8}$")
_INVALID_HERB_NAMES = {"未知", "未知药", "不详", "未详"}
_NON_ENTITY_MARKERS = (
    "什么",
    "哪些",
    "为什么",
    "怎么",
    "如何",
    "我想",
    "我担心",
    "但是",
    "会不会",
    "推荐",
    "病人",
    "治疗",
    "中医",
    "中药",
    "药材中",
    "中的",
    "请问",
    "听说",
    "有没",
    "有没有",
    "能不能",
    "是否",
    "方剂",
    "具",
    "可以",
    "对于",
    "具备",
    "服用",
    "长期",
    "短期",
    "止疼",
    "药物",
)


def is_valid_tcm_herb_label(value: str) -> bool:
    """Reject question fragments before they enter the herb catalog or graph."""
    label = str(value or "").strip()
    return (
        bool(_CHINESE_NAME.fullmatch(label))
        and label not in _INVALID_HERB_NAMES
        and not any(marker in label for marker in _NON_ENTITY_MARKERS)
    )
