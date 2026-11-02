from __future__ import annotations

import re


ENCOUNTER_MARKER_RE = re.compile(r"【诊疗记录(?:\s*RX-\d+|\s*\d+)】", re.IGNORECASE)
EMPTY_HISTORY_VALUES = {
    "",
    "-",
    "无",
    "暂无",
    "未记录",
    "无特殊",
    "无特殊病史",
    "无明确既往史",
    "无明确病史",
}


def normalize_medical_history(summary: str | None) -> str:
    """Use patient-local numbering for legacy and current encounter records."""
    text = str(summary or "").strip()
    if not ENCOUNTER_MARKER_RE.search(text):
        return text

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if _placeholder_key(line) not in EMPTY_HISTORY_VALUES]
    text = "\n".join(line for line in lines if line)
    sequence = 0

    def replace_marker(_: re.Match[str]) -> str:
        nonlocal sequence
        sequence += 1
        return f"【诊疗记录{sequence}】"

    return ENCOUNTER_MARKER_RE.sub(replace_marker, text)


def append_confirmed_encounter(
    summary: str | None,
    *,
    visit_date: str,
    prescription_title: str,
    symptoms: str,
) -> str:
    normalized = normalize_medical_history(summary)
    next_number = len(ENCOUNTER_MARKER_RE.findall(normalized)) + 1
    entry = (
        f"【诊疗记录{next_number}】{visit_date}；处方：{prescription_title}；"
        f"医生确认症状/诊疗背景：{symptoms}"
    )
    return "\n".join(part for part in (normalized, entry) if part)


def _placeholder_key(value: str) -> str:
    return value.strip().rstrip("。；; ")
