from __future__ import annotations

import re
import unicodedata


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARS.sub("", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()


def split_candidate_text(value: object) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    parts = re.split(r"[,，;；、\s]+", text)
    return [part for part in (normalize_text(part) for part in parts) if part]

