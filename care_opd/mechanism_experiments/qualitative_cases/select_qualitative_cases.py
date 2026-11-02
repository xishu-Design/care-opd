#!/usr/bin/env python3
"""Select a fixed, auditable six-slot CARE-OPD qualitative case panel.

The selector never asks a human to rank response quality. Four representative
slots are defined in advance and choose the example closest to the within-slot
median CARE full-response recovery margin, with a seeded hash tie-break. A
CARE-still-failed slot uses the same median rule inside that predeclared
outcome stratum. A CARE-regression slot is selected by seeded hash because
recovery margins are intentionally measured only on Clean-SFT failures.

If a slot has no eligible unused request, the output records that absence; it
never substitutes a different category or fabricates a case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "care-opd-qualitative-cases-v1"
MODEL_LABELS = ("clean_sft", "gkd_full", "care_opd")
SLOT_SPECS = (
    {
        "slot": "representative_bad_history",
        "title": "Bad-history representative",
        "kind": "representative",
        "field": "variant",
        "value": "bad_history",
    },
    {
        "slot": "representative_source_outage",
        "title": "Source-outage representative",
        "kind": "representative",
        "field": "variant",
        "value": "source_outage",
    },
    {
        "slot": "representative_evidence_conflict",
        "title": "Evidence-conflict representative",
        "kind": "representative",
        "field": "variant",
        "value": "evidence_conflict",
    },
    {
        "slot": "representative_invented_source_id",
        "title": "Invented-source-ID representative",
        "kind": "representative",
        "field": "failure_type",
        "value": "invented_source_id",
    },
    {
        "slot": "care_still_failed",
        "title": "CARE-OPD remaining failure",
        "kind": "care_still_failed",
    },
    {
        "slot": "care_regression",
        "title": "CARE-OPD regression from a Clean-SFT success",
        "kind": "care_regression",
    },
)

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
NATIONAL_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\w)")
LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{8,16}(?!\d)")


class InputError(ValueError):
    """Raised when a source archive violates the paired selection contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise InputError(f"{path}:{line_number}: row must be an object")
            value["__line__"] = line_number
            rows.append(value)
    if not rows:
        raise InputError(f"{path}: no records")
    return rows


def _nonempty(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}: expected a non-empty string")
    return value.strip()


def _string_list(value: Any, *, where: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise InputError(f"{where}: expected a list of non-empty strings")
    return [item.strip() for item in value]


def _index_rollouts(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = _nonempty(row.get("id"), where=f"{where}.id")
        if item_id in indexed:
            raise InputError(f"{where}: duplicate id {item_id!r}")
        passed = row.get("passed")
        if not isinstance(passed, bool):
            raise InputError(f"{where}.passed: expected a JSON boolean")
        indexed[item_id] = {
            "id": item_id,
            "variant": _nonempty(row.get("variant"), where=f"{where}.variant"),
            "passed": passed,
            "failure_types": _string_list(
                row.get("failure_types"), where=f"{where}.failure_types"
            ),
            "selected_components": _string_list(
                row.get("selected_components"), where=f"{where}.selected_components"
            ),
            "student_messages": row.get("student_messages"),
            "rollout": str(row.get("rollout") or ""),
        }
    return indexed


def _index_failure_bank(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = _nonempty(row.get("id"), where=f"{where}.id")
        if item_id in indexed:
            raise InputError(f"{where}: duplicate id {item_id!r}")
        indexed[item_id] = {
            "id": item_id,
            "variant": _nonempty(row.get("variant"), where=f"{where}.variant"),
            "failure_types": _string_list(
                row.get("failure_types"), where=f"{where}.failure_types"
            ),
            "selected_components": _string_list(
                row.get("selected_components"), where=f"{where}.selected_components"
            ),
            "student_messages": row.get("student_messages"),
            "oracle_response": str(row.get("oracle_response") or ""),
        }
    return indexed


def _finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{where}: expected a finite number")
    return result


def _index_margins(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        line = row.pop("__line__")
        where = f"{path}:{line}"
        item_id = _nonempty(row.get("id"), where=f"{where}.id")
        label = _nonempty(row.get("model_label"), where=f"{where}.model_label")
        if label not in MODEL_LABELS:
            continue
        full = row.get("full")
        selected = row.get("selected")
        outside = row.get("outside")
        if not all(isinstance(value, Mapping) for value in (full, selected, outside)):
            raise InputError(f"{where}: full/selected/outside objects are required")
        by_label = indexed.setdefault(item_id, {})
        if label in by_label:
            raise InputError(f"{where}: duplicate ({item_id!r}, {label!r})")
        by_label[label] = {
            "full": _finite(full.get("margin"), where=f"{where}.full.margin"),
            "localized": (
                _finite(
                    selected.get("localized_margin"),
                    where=f"{where}.selected.localized_margin",
                )
                if selected.get("localized_complete")
                else None
            ),
            "outside": (
                _finite(outside.get("margin"), where=f"{where}.outside.margin")
                if selected.get("localized_complete")
                else None
            ),
            "classification": selected.get("classification"),
        }
    for item_id, models in indexed.items():
        missing = sorted(set(MODEL_LABELS) - set(models))
        if missing:
            raise InputError(f"{path}: id {item_id!r} is missing models {missing}")
    if not indexed:
        raise InputError(f"{path}: no supported margin records")
    return indexed


def _stable_hash(seed: int, slot: str, item_id: str) -> str:
    return _sha256_text(f"{seed}:{slot}:{item_id}")


def _redact(text: str) -> str:
    text = NATIONAL_ID_RE.sub("[NATIONAL_ID]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    return LONG_DIGITS_RE.sub("[LONG_NUMBER]", text)


def _preview(value: Any, *, max_characters: int) -> dict[str, Any]:
    if isinstance(value, str):
        raw = value
    else:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    redacted = _redact(raw)
    normalized = " ".join(redacted.split())
    truncated = len(normalized) > max_characters
    return {
        "sha256": _sha256_text(raw),
        "characters": len(raw),
        "preview": normalized[:max_characters] + (" …[TRUNCATED]" if truncated else ""),
        "truncated": truncated,
        "redaction": "email, mainland-mobile, national-ID, and long-number regex",
    }


def _representative_candidates(
    spec: Mapping[str, Any],
    failure_bank: Mapping[str, Mapping[str, Any]],
    margins: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[str]:
    result: list[str] = []
    for item_id in sorted(set(failure_bank) & set(margins)):
        row = failure_bank[item_id]
        if spec["field"] == "variant" and row["variant"] == spec["value"]:
            result.append(item_id)
        elif (
            spec["field"] == "failure_type"
            and spec["value"] in row["failure_types"]
        ):
            result.append(item_id)
    return result


def _choose_nearest_median(
    candidates: Sequence[str],
    *,
    margins: Mapping[str, Mapping[str, Mapping[str, Any]]],
    excluded: set[str],
    seed: int,
    slot: str,
) -> tuple[str | None, dict[str, Any]]:
    all_scores = [float(margins[item_id]["care_opd"]["full"]) for item_id in candidates]
    median = statistics.median(all_scores) if all_scores else None
    available = [item_id for item_id in candidates if item_id not in excluded]
    audit = {
        "rule": "nearest within-slot median CARE full margin; seeded SHA-256 tie-break",
        "eligible_before_distinctness": len(candidates),
        "eligible_after_distinctness": len(available),
        "within_slot_median_care_full_margin": median,
        "seed": seed,
    }
    if not available or median is None:
        return None, audit
    chosen = min(
        available,
        key=lambda item_id: (
            abs(float(margins[item_id]["care_opd"]["full"]) - median),
            _stable_hash(seed, slot, item_id),
        ),
    )
    audit["chosen_distance_to_median"] = abs(
        float(margins[chosen]["care_opd"]["full"]) - median
    )
    return chosen, audit


def _choose_hash(
    candidates: Sequence[str], *, excluded: set[str], seed: int, slot: str
) -> tuple[str | None, dict[str, Any]]:
    available = [item_id for item_id in candidates if item_id not in excluded]
    audit = {
        "rule": "minimum seeded SHA-256 within the predeclared outcome stratum",
        "eligible_before_distinctness": len(candidates),
        "eligible_after_distinctness": len(available),
        "seed": seed,
    }
    if not available:
        return None, audit
    return min(available, key=lambda item_id: _stable_hash(seed, slot, item_id)), audit


def _case_payload(
    *,
    spec: Mapping[str, Any],
    item_id: str,
    audit: Mapping[str, Any],
    failure_bank: Mapping[str, Mapping[str, Any]],
    margins: Mapping[str, Mapping[str, Mapping[str, Any]]],
    rollouts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    max_characters: int,
) -> dict[str, Any]:
    bank = failure_bank.get(item_id)
    clean = rollouts["clean_sft"][item_id]
    input_value = bank.get("student_messages") if bank else clean.get("student_messages")
    response_payload: dict[str, Any] = {}
    for label in MODEL_LABELS:
        row = rollouts[label][item_id]
        response_payload[label] = {
            "passed": row["passed"],
            "failure_types": row["failure_types"],
            "selected_components": row["selected_components"],
            "response": _preview(row["rollout"], max_characters=max_characters),
        }
    return {
        "slot": spec["slot"],
        "title": spec["title"],
        "status": "SELECTED",
        "id": item_id,
        "id_sha256": _sha256_text(item_id),
        "selection_audit": dict(audit),
        "variant": clean["variant"],
        "reference_failure_types": bank["failure_types"] if bank else [],
        "reference_selected_components": bank["selected_components"] if bank else [],
        "margins": margins.get(item_id),
        "input": _preview(input_value, max_characters=max_characters),
        "oracle": (
            _preview(bank["oracle_response"], max_characters=max_characters)
            if bank and bank["oracle_response"]
            else None
        ),
        "models": response_payload,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CARE-OPD result-blind qualitative case panel",
        "",
        (
            f"Status: **{payload['status']}**. Selected slots: "
            f"`{payload['selected_slots']}` / `{payload['requested_slots']}`."
        ),
        "",
        "Cases are selected by preregistered numeric/hash rules that never inspect response "
        "text; previews are regex-redacted and length-limited.",
        "",
    ]
    for index, case in enumerate(payload["cases"], 1):
        lines.extend([f"## {index}. {case['title']}", ""])
        if case["status"] != "SELECTED":
            lines.extend(
                [
                    "- Status: **MISSING**",
                    f"- Reason: {case['missing_reason']}",
                    f"- Selection rule: {case['selection_audit']['rule']}",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"- Request ID: `{case['id']}`",
                f"- ID SHA-256: `{case['id_sha256']}`",
                f"- Variant: `{case['variant']}`",
                f"- Reference failures: `{', '.join(case['reference_failure_types']) or 'N/A'}`",
                f"- Verifier-selected fields: `{', '.join(case['reference_selected_components']) or 'N/A'}`",
                f"- Selection rule: {case['selection_audit']['rule']}",
                "",
                "### Input preview",
                "",
                case["input"]["preview"] or "*(empty)*",
                "",
            ]
        )
        if case.get("oracle"):
            lines.extend(["### Verifier-valid oracle preview", "", case["oracle"]["preview"], ""])
        for label in MODEL_LABELS:
            model = case["models"][label]
            margin = (case.get("margins") or {}).get(label, {}).get("full")
            lines.extend(
                [
                    f"### {label}",
                    "",
                    f"- Passed: `{model['passed']}`",
                    f"- Failure types: `{', '.join(model['failure_types']) or 'none'}`",
                    f"- Full recovery margin: `{margin if margin is not None else 'N/A'}`",
                    "",
                    model["response"]["preview"] or "*(empty)*",
                    "",
                ]
            )
    lines.extend(
        [
            "## Claim boundary",
            "",
            "- The four representative slots target a typical CARE full-margin example within each fixed category, not the best-looking example.",
            "- The remaining-failure and regression slots are intentionally outcome-conditioned diagnostic strata; selection within each stratum remains automatic.",
            "- Qualitative cases illustrate verifier behavior and failure modes. Aggregate effectiveness claims must come from the paired quantitative evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def select_cases(
    *,
    clean_rollouts_path: Path,
    gkd_rollouts_path: Path,
    care_rollouts_path: Path,
    failure_bank_path: Path,
    margin_path: Path,
    output_dir: Path,
    selection_seed: int = 20260813,
    max_preview_characters: int = 1200,
    overwrite: bool = False,
) -> dict[str, Any]:
    if max_preview_characters < 100:
        raise InputError("max_preview_characters must be at least 100")
    json_output = output_dir / "qualitative_cases.json"
    md_output = output_dir / "qualitative_cases.md"
    if not overwrite and (json_output.exists() or md_output.exists()):
        raise FileExistsError("refusing to overwrite existing qualitative-case outputs")

    paths = {
        "clean_sft": clean_rollouts_path,
        "gkd_full": gkd_rollouts_path,
        "care_opd": care_rollouts_path,
    }
    rollouts = {label: _index_rollouts(path) for label, path in paths.items()}
    common_ids = set.intersection(*(set(rows) for rows in rollouts.values()))
    if not common_ids:
        raise InputError("formal rollout archives have no common request IDs")
    failure_bank = _index_failure_bank(failure_bank_path)
    margins = _index_margins(margin_path)
    missing_bank = sorted(set(margins) - set(failure_bank))
    if missing_bank:
        raise InputError(f"margin IDs absent from failure bank: {missing_bank[:10]}")
    missing_formal = sorted((set(margins) | set(failure_bank)) - common_ids)
    if missing_formal:
        raise InputError(f"analysis IDs absent from paired formal rollouts: {missing_formal[:10]}")
    invalid_failure_bank = sorted(
        item_id for item_id in failure_bank if rollouts["clean_sft"][item_id]["passed"]
    )
    if invalid_failure_bank:
        raise InputError(f"failure-bank IDs pass under formal Clean-SFT: {invalid_failure_bank[:10]}")

    selected: set[str] = set()
    cases: list[dict[str, Any]] = []
    for spec in SLOT_SPECS:
        if spec["kind"] == "representative":
            candidates = _representative_candidates(spec, failure_bank, margins)
            chosen, audit = _choose_nearest_median(
                candidates,
                margins=margins,
                excluded=selected,
                seed=selection_seed,
                slot=spec["slot"],
            )
        elif spec["kind"] == "care_still_failed":
            candidates = [
                item_id
                for item_id in sorted(set(margins) & common_ids)
                if not rollouts["care_opd"][item_id]["passed"]
            ]
            chosen, audit = _choose_nearest_median(
                candidates,
                margins=margins,
                excluded=selected,
                seed=selection_seed,
                slot=spec["slot"],
            )
        else:
            candidates = [
                item_id
                for item_id in sorted(common_ids)
                if rollouts["clean_sft"][item_id]["passed"]
                and not rollouts["care_opd"][item_id]["passed"]
            ]
            chosen, audit = _choose_hash(
                candidates,
                excluded=selected,
                seed=selection_seed,
                slot=spec["slot"],
            )
        if chosen is None:
            cases.append(
                {
                    "slot": spec["slot"],
                    "title": spec["title"],
                    "status": "MISSING",
                    "missing_reason": (
                        "no eligible unused request satisfies this predeclared slot"
                    ),
                    "selection_audit": audit,
                }
            )
            continue
        selected.add(chosen)
        cases.append(
            _case_payload(
                spec=spec,
                item_id=chosen,
                audit=audit,
                failure_bank=failure_bank,
                margins=margins,
                rollouts=rollouts,
                max_characters=max_preview_characters,
            )
        )

    selected_count = sum(case["status"] == "SELECTED" for case in cases)
    payload = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if selected_count == len(SLOT_SPECS) else "COMPLETE_WITH_MISSING_SLOTS",
        "requested_slots": len(SLOT_SPECS),
        "selected_slots": selected_count,
        "selection_seed": selection_seed,
        "max_preview_characters": max_preview_characters,
        "selection_contract": {
            "representatives": (
                "fixed category; nearest within-category median CARE full margin; "
                "seeded SHA-256 tie-break"
            ),
            "care_still_failed": (
                "CARE verifier failure within fixed margin cohort; nearest stratum median"
            ),
            "care_regression": (
                "Clean-SFT pass and CARE failure; seeded SHA-256 (no margin exists by design)"
            ),
            "distinctness": "slot priority order; one request ID may appear at most once",
            "missing_policy": "emit explicit MISSING slot; never substitute or fabricate",
            "content_blinding": "response text is not used for selection",
        },
        "inputs": {
            **{
                f"{label}_rollouts": str(path.resolve())
                for label, path in paths.items()
            },
            **{
                f"{label}_rollouts_sha256": _sha256_file(path)
                for label, path in paths.items()
            },
            "failure_bank": str(failure_bank_path.resolve()),
            "failure_bank_sha256": _sha256_file(failure_bank_path),
            "margin_jsonl": str(margin_path.resolve()),
            "margin_jsonl_sha256": _sha256_file(margin_path),
        },
        "cases": cases,
        "claim_boundary": (
            "Automatically selected illustration panel; not statistical effectiveness "
            "evidence and not causal attribution"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        json_output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(md_output, _markdown(payload))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-rollouts", type=Path, required=True)
    parser.add_argument("--gkd-rollouts", type=Path, required=True)
    parser.add_argument("--care-rollouts", type=Path, required=True)
    parser.add_argument("--failure-bank", type=Path, required=True)
    parser.add_argument("--margin-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260813)
    parser.add_argument("--max-preview-characters", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = select_cases(
        clean_rollouts_path=args.clean_rollouts,
        gkd_rollouts_path=args.gkd_rollouts,
        care_rollouts_path=args.care_rollouts,
        failure_bank_path=args.failure_bank,
        margin_path=args.margin_jsonl,
        output_dir=args.output_dir,
        selection_seed=args.selection_seed,
        max_preview_characters=args.max_preview_characters,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_slots": payload["selected_slots"],
                "requested_slots": payload["requested_slots"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
