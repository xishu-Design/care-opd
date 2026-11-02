from __future__ import annotations

import json
import re
from typing import Any

from app.llm.schemas import GeneratedResponse, ParserResult


def parse_generated_response_result(content: str) -> ParserResult:
    raw_text = content or ""
    last_error = "empty_output" if not raw_text.strip() else "invalid_json"
    for candidate in _json_candidates(raw_text):
        parsed, parse_error = _load_json_candidate(candidate)
        if parse_error is not None:
            last_error = parse_error
            continue
        if not _has_generated_response_shape(parsed):
            last_error = "schema_invalid:generated_response_shape_missing"
            continue
        try:
            structured = GeneratedResponse.model_validate(parsed).model_dump(mode="json")
        except Exception as exc:
            # A response can contain an explanatory JSON object before the
            # fenced schema-conformant answer.  Keep trying later candidates
            # instead of failing on the first syntactically valid object.
            last_error = f"schema_invalid:{exc}"
            continue
        return ParserResult(success=True, structured_output=structured, raw_text=raw_text, parse_error=None, schema_valid=True)
    return ParserResult(success=False, structured_output=None, raw_text=raw_text, parse_error=last_error, schema_valid=False)


def parse_generated_response(content: str) -> dict[str, Any]:
    result = parse_generated_response_result(content)
    if result.success and result.structured_output is not None:
        return result.structured_output
    return GeneratedResponse(
        title="LLM 生成内容",
        summary=content,
        disclaimer="本内容由生成层整理，必须以 Evidence Packet 和后端安全判断为准。",
    ).model_dump(mode="json")


def _json_candidates(content: str) -> list[str]:
    stripped = content.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", content, re.DOTALL))
    candidates.extend(_extract_json_objects(content))
    candidates.extend(
        repaired
        for start in _json_object_starts(content)
        if (repaired := _close_unterminated_json_object(content[start:]))
    )

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _load_json_candidate(candidate: str) -> tuple[Any | None, str | None]:
    """Decode common, content-preserving JSON deviations from local models.

    The recovery steps never invent fields or text.  They only accept literal
    control characters inside strings, remove trailing collection commas, or
    close containers that ended cleanly at EOF.  Pydantic validation remains
    the authority for the response schema after decoding.
    """

    last_error = "invalid_json"
    variants = [candidate]
    without_trailing_commas = _remove_trailing_collection_commas(candidate)
    if without_trailing_commas != candidate:
        variants.append(without_trailing_commas)
    for variant in variants:
        for strict in (True, False):
            try:
                return json.loads(variant, strict=strict), None
            except Exception as exc:
                last_error = f"invalid_json:{exc}"
    return None, last_error


def _has_generated_response_shape(parsed: Any) -> bool:
    """Reject nested section/prose objects before permissive Pydantic defaults."""

    if not isinstance(parsed, dict):
        return False
    response_keys = {
        "title",
        "summary",
        "sections",
        "warnings",
        "next_actions",
        "disclaimer",
    }
    present = response_keys.intersection(parsed)
    return len(present) >= 2 and bool(
        {"title", "summary", "disclaimer"}.intersection(present)
    )


def _json_object_starts(content: str) -> list[int]:
    """Return possible object starts, excluding braces inside quoted text."""

    starts: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(content):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            starts.append(index)
    return starts


def _extract_json_objects(content: str) -> list[str]:
    """Extract every balanced object, not only the first one in the answer."""

    objects: list[str] = []
    for start in _json_object_starts(content):
        extracted = _extract_balanced_json_object(content, start)
        if extracted:
            objects.append(extracted)
    return objects


def _extract_balanced_json_object(content: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return None


def _close_unterminated_json_object(content: str) -> str | None:
    """Close only cleanly terminated JSON containers at end-of-output.

    A response ending inside a string is not repaired because doing so could
    silently accept truncated clinical text.  Mismatched containers are also
    rejected.  This is deliberately narrower than a general JSON-repair tool.
    """

    candidate = content.strip()
    if not candidate.startswith("{"):
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            expected = "{" if char == "}" else "["
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
    if in_string or not stack:
        return None
    closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return candidate + closers


def _remove_trailing_collection_commas(content: str) -> str:
    """Remove commas immediately before `}` or `]`, outside JSON strings."""

    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(content):
        char = content[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(content) and content[lookahead].isspace():
                lookahead += 1
            if lookahead < len(content) and content[lookahead] in "}]":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)
