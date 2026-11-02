from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR


DEFAULT_INCOMPATIBILITY_FILE = DEFAULT_CONTENT_DATA_DIR / "herb_incompatibilities.json"
_SPACE_RE = re.compile(r"[\s\u3000]+")


def find_herb_incompatibilities(
    herb_names: Iterable[str],
    *,
    data_file: Path | str = DEFAULT_INCOMPATIBILITY_FILE,
) -> list[dict[str, str]]:
    """Return deterministic, source-bound traditional incompatibility matches.

    Matching is intentionally exact after whitespace normalization.  It does
    not use embeddings or an LLM, and it does not infer that processed herbs
    are equivalent to an unlisted source term.
    """

    present: dict[str, str] = {}
    for raw_name in herb_names:
        display_name = str(raw_name or "").strip()
        normalized = _normalize(display_name)
        if normalized:
            present.setdefault(normalized, display_name)

    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for constraint in load_herb_incompatibilities(data_file).get("constraints", []):
        left_matches = _matching_terms(constraint.get("left", []), present)
        right_matches = _matching_terms(constraint.get("right", []), present)
        for left_term, left_actual in left_matches:
            for right_term, right_actual in right_matches:
                if _normalize(left_actual) == _normalize(right_actual):
                    continue
                pair_key = tuple(sorted((_normalize(left_actual), _normalize(right_actual))))
                key = (str(constraint.get("constraint_id", "")), *pair_key)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    {
                        "constraint_id": str(constraint.get("constraint_id", "")),
                        "category": str(constraint.get("category", "传统配伍禁忌")),
                        "herb_a": left_actual,
                        "herb_b": right_actual,
                        "source_term_a": str(left_term.get("source_term") or left_term.get("name") or ""),
                        "source_term_b": str(right_term.get("source_term") or right_term.get("name") or ""),
                        "relation_type": str(constraint.get("relation_type", "TRADITIONAL_INCOMPATIBILITY")),
                        "source_id": str(constraint.get("source_id") or ""),
                        "source_record_id": str(constraint.get("source_record_id") or ""),
                    }
                )
    return sorted(matches, key=lambda item: (item["category"], item["herb_a"], item["herb_b"]))


def expanded_incompatibility_pairs(
    *,
    data_file: Path | str = DEFAULT_INCOMPATIBILITY_FILE,
) -> list[dict[str, str]]:
    """Expand grouped source rules into graph-ready canonical herb pairs."""

    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for constraint in load_herb_incompatibilities(data_file).get("constraints", []):
        for left in constraint.get("left", []):
            for right in constraint.get("right", []):
                herb_a = str(left.get("name") or "").strip()
                herb_b = str(right.get("name") or "").strip()
                if not herb_a or not herb_b or herb_a == herb_b:
                    continue
                key = (str(constraint.get("constraint_id", "")), *sorted((herb_a, herb_b)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "constraint_id": str(constraint.get("constraint_id", "")),
                        "category": str(constraint.get("category", "传统配伍禁忌")),
                        "herb_a": herb_a,
                        "herb_b": herb_b,
                        "relation_type": str(constraint.get("relation_type", "TRADITIONAL_INCOMPATIBILITY")),
                        "source_id": str(constraint.get("source_id") or ""),
                        "source_record_id": str(constraint.get("source_record_id") or ""),
                    }
                )
    return pairs


def incompatibilities_for_herb(
    herb_name: str,
    *,
    data_file: Path | str = DEFAULT_INCOMPATIBILITY_FILE,
) -> list[dict[str, str]]:
    """Return the source-listed herbs that should trigger a pairing warning.

    This catalog helper follows the same exact-name/explicit-alias policy as
    prescription review.  An empty list means only that no relationship is
    present in the currently imported standard; it must not be presented as a
    general claim that the herb is safe with every other medicine.
    """

    normalized_name = _normalize(str(herb_name or ""))
    if not normalized_name:
        return []

    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for constraint in load_herb_incompatibilities(data_file).get("constraints", []):
        left = constraint.get("left", [])
        right = constraint.get("right", [])
        partner_terms: list[dict] = []
        if any(_term_matches_name(term, normalized_name) for term in left):
            partner_terms.extend(right)
        if any(_term_matches_name(term, normalized_name) for term in right):
            partner_terms.extend(left)

        for partner in partner_terms:
            partner_name = str(partner.get("name") or "").strip()
            key = (str(constraint.get("constraint_id", "")), _normalize(partner_name))
            if not partner_name or key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "herb_name": partner_name,
                    "category": str(constraint.get("category", "传统配伍禁忌")),
                    "relation_type": str(constraint.get("relation_type", "TRADITIONAL_INCOMPATIBILITY")),
                    "source_id": str(constraint.get("source_id") or ""),
                    "source_record_id": str(constraint.get("source_record_id") or ""),
                }
            )
    return sorted(results, key=lambda item: (item["category"], item["herb_name"]))


def load_herb_incompatibilities(data_file: Path | str = DEFAULT_INCOMPATIBILITY_FILE) -> dict:
    path = Path(data_file)
    return _load_cached(str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)


@lru_cache(maxsize=8)
def _load_cached(path_value: str, _mtime_ns: int, _size: int) -> dict:
    with Path(path_value).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("constraints"), list):
        raise ValueError("herb_incompatibilities.json must contain a constraints list")
    return payload


def _matching_terms(terms: list[dict], present: dict[str, str]) -> list[tuple[dict, str]]:
    matches: list[tuple[dict, str]] = []
    seen_actual: set[str] = set()
    for term in terms:
        labels = [term.get("name"), term.get("source_term"), *(term.get("aliases") or [])]
        for label in labels:
            normalized = _normalize(str(label or ""))
            if normalized not in present or normalized in seen_actual:
                continue
            seen_actual.add(normalized)
            matches.append((term, present[normalized]))
    return matches


def _term_matches_name(term: dict, normalized_name: str) -> bool:
    labels = [term.get("name"), term.get("source_term"), *(term.get("aliases") or [])]
    return any(_normalize(str(label or "")) == normalized_name for label in labels)


def _normalize(value: str) -> str:
    return _SPACE_RE.sub("", value).strip().lower()
