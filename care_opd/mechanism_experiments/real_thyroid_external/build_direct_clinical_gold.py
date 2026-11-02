#!/usr/bin/env python3
"""Materialize spreadsheet-provided clinical gold without an LLM annotator.

The private agent queue already contains the deidentified report text and the
structured spreadsheet candidates.  This utility treats every non-empty
candidate as *provided clinical gold* and performs only deterministic literal
alignment against its own report text.  Alignment is auxiliary metadata: a
gold field remains gold when its value is repeated, one character long, or not
present verbatim in the free-text report.

No clinical value is inferred, normalized, corrected, or printed.  Blank
spreadsheet cells remain unknown/not-reported and never become negative gold.
All row-level artifacts and hash-bearing manifests are written atomically with
private permissions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "care_opd.real_thyroid_direct_clinical_gold.v1"
CLINICAL_GOLD_NAME = "clinical_gold.private.jsonl"
EXACT_SPAN_NAME = "exact_verified_fields.private.jsonl"
REVIEW_NAME = "human_review_queue.private.jsonl"
SUMMARY_NAME = "direct_annotation_summary.json"
MANIFEST_NAME = "direct_annotation_manifest.private.json"


class DirectClinicalGoldError(ValueError):
    """Raised when the private queue violates the direct-gold contract."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DirectClinicalGoldError(
                    f"Invalid JSONL object at line {line_number}."
                ) from error
            if not isinstance(value, dict):
                raise DirectClinicalGoldError(
                    f"Queue row {line_number} must be an object."
                )
            rows.append(value)
    if not rows:
        raise DirectClinicalGoldError("Private agent queue contains no rows.")
    return rows


def _atomic_text(path: Path, text: str, *, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if private:
            try:
                path.chmod(0o600)
            except OSError:
                pass
    finally:
        temporary.unlink(missing_ok=True)


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise DirectClinicalGoldError(f"Queue row is missing {key}.")
    return value


def _literal_alignment(source: str, value: str) -> tuple[str, list[dict[str, Any]]]:
    """Return conservative alignment metadata for a provided gold value."""

    first = source.find(value)
    if first < 0:
        return "not_verbatim", []
    second = source.find(value, first + 1)
    if len(value) < 2:
        return "ambiguous_or_short", []
    if second >= 0:
        return "duplicate_verbatim", []
    end = first + len(value)
    return (
        "exact_unique_verbatim",
        [{"start": first, "end": end, "text_sha256": _sha256_text(source[first:end])}],
    )


def _base_identity(task: Mapping[str, Any], field: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": task["case_id"],
        "report_group_hmac": task["report_group_hmac"],
        "modality": task["modality"],
        "entity_id": field["entity_id"],
        "field_path": field["field_path"],
        "source_record_sha256": task["source_record_sha256"],
        "annotation_protocol": PROTOCOL_VERSION,
    }


def materialize_direct_clinical_gold(
    queue_path: Path,
    output_dir: Path,
    *,
    acknowledge_private_clinical_data: bool = False,
) -> dict[str, Any]:
    """Create full provided gold plus an optional exact-span alignment layer."""

    if not acknowledge_private_clinical_data:
        raise PermissionError(
            "Direct gold materialization requires "
            "--acknowledge-private-clinical-data."
        )

    tasks = _read_jsonl(queue_path)
    case_ids: set[str] = set()
    clinical_gold: list[dict[str, Any]] = []
    exact_spans: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    alignment_counts: Counter[str] = Counter()
    modality_reports: Counter[str] = Counter()
    modality_gold: Counter[str] = Counter()
    modality_localized: Counter[str] = Counter()
    total_slots = 0
    empty_slots = 0

    for task in tasks:
        case_id = _required_text(task, "case_id")
        if case_id in case_ids:
            raise DirectClinicalGoldError("Queue contains duplicate case_id values.")
        case_ids.add(case_id)
        _required_text(task, "report_group_hmac")
        modality = _required_text(task, "modality")
        source = str(task.get("source_text_private") or "")
        if not source:
            raise DirectClinicalGoldError("Queue row has empty private source text.")
        source_sha256 = _required_text(task, "source_record_sha256")
        if _sha256_text(source) != source_sha256:
            raise DirectClinicalGoldError(
                "Queue source hash mismatch; refusing to materialize gold."
            )
        fields = task.get("candidate_fields")
        if not isinstance(fields, list):
            raise DirectClinicalGoldError("candidate_fields must be a list.")
        modality_reports[modality] += 1
        observed_slots: set[tuple[str, str]] = set()

        for field in fields:
            total_slots += 1
            if not isinstance(field, Mapping):
                raise DirectClinicalGoldError("Candidate field must be an object.")
            entity_id = _required_text(field, "entity_id")
            field_path = _required_text(field, "field_path")
            slot = (entity_id, field_path)
            if slot in observed_slots:
                raise DirectClinicalGoldError(
                    "Queue contains a duplicate entity/field slot within a case."
                )
            observed_slots.add(slot)
            identity = _base_identity(task, field)
            raw_value = field.get("raw_value")
            value = "" if raw_value is None else str(raw_value)
            if not value.strip():
                empty_slots += 1
                alignment_counts["not_reported_unknown"] += 1
                review_rows.append(
                    {
                        **identity,
                        "candidate_value_private": raw_value,
                        "reference_status": "not_reported_unknown",
                        "support_status": "unresolved",
                        "alignment_status": "not_applicable_empty_candidate",
                        "clinical_gold_retained": False,
                        "evidence_spans": [],
                        "reason_codes": ["EMPTY_CELL_IS_UNKNOWN_NOT_NEGATIVE"],
                    }
                )
                continue

            alignment_status, spans = _literal_alignment(source, value)
            localized = alignment_status == "exact_unique_verbatim"
            alignment_counts[alignment_status] += 1
            modality_gold[modality] += 1
            if localized:
                modality_localized[modality] += 1
            row = {
                **identity,
                "candidate_value_private": value,
                "normalized_value": value,
                "unit": "",
                "reference_status": "clinical_gold_provided",
                "support_status": (
                    "exact_supported" if localized else "clinical_gold_provided"
                ),
                "alignment_status": alignment_status,
                "clinical_gold_retained": True,
                "evidence_spans": spans,
                "reason_codes": [
                    (
                        "DETERMINISTIC_UNIQUE_VERBATIM_SPAN"
                        if localized
                        else f"CLINICAL_GOLD_WITH_{alignment_status.upper()}"
                    )
                ],
            }
            clinical_gold.append(row)
            if localized:
                exact_spans.append(row)
            else:
                review_rows.append(
                    {
                        **row,
                        "review_status": "optional_alignment_review",
                        "clinical_gold_retained": True,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / ".gitignore", "*\n!.gitignore\n")
    _atomic_text(output_dir / CLINICAL_GOLD_NAME, _jsonl(clinical_gold))
    _atomic_text(output_dir / EXACT_SPAN_NAME, _jsonl(exact_spans))
    _atomic_text(output_dir / REVIEW_NAME, _jsonl(review_rows))

    summary: dict[str, Any] = {
        "schema_version": f"{PROTOCOL_VERSION}.summary",
        "status": "COMPLETE",
        "annotation_mode": "provided_clinical_gold_plus_deterministic_literal_alignment",
        "counts": {
            "report_groups": len(tasks),
            "candidate_field_slots": total_slots,
            "clinical_gold_fields": len(clinical_gold),
            "span_localized_clinical_gold_fields": len(exact_spans),
            "clinical_gold_without_exact_span": len(clinical_gold) - len(exact_spans),
            "empty_unknown_slots": empty_slots,
            "human_review_queue_rows": len(review_rows),
        },
        "alignment_counts": dict(sorted(alignment_counts.items())),
        "by_modality": {
            modality: {
                "report_groups": modality_reports[modality],
                "clinical_gold_fields": modality_gold[modality],
                "span_localized_clinical_gold_fields": modality_localized[modality],
            }
            for modality in sorted(modality_reports)
        },
        "policy": {
            "nonempty_structured_candidates_are_provided_clinical_gold": True,
            "exact_span_is_an_auxiliary_alignment_not_an_inclusion_gate": True,
            "minimum_unique_span_characters": 2,
            "empty_candidate_semantics": "unknown_not_negative",
            "clinical_inference_or_correction_performed": False,
            "source_text_or_values_in_summary": False,
        },
        "claim_boundary": (
            "Spreadsheet-provided clinical gold with deterministic literal alignment; "
            "not newly inferred clinical labels and not independent expert re-adjudication."
        ),
    }
    _atomic_text(
        output_dir / SUMMARY_NAME,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    manifest = {
        "schema_version": f"{PROTOCOL_VERSION}.private_manifest",
        "created_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "queue": {
            "sha256": _sha256_file(queue_path),
            "rows": len(tasks),
            "case_ids_sha256": _sha256_text("\n".join(sorted(case_ids))),
        },
        "outputs": {
            name: {"sha256": _sha256_file(output_dir / name), "rows": rows}
            for name, rows in (
                (CLINICAL_GOLD_NAME, len(clinical_gold)),
                (EXACT_SPAN_NAME, len(exact_spans)),
                (REVIEW_NAME, len(review_rows)),
            )
        },
        "protocol": {
            "clinical_gold_source": "nonempty_structured_spreadsheet_candidate",
            "literal_alignment": "unique_exact_substring_with_minimum_length_2",
            "blank_semantics": "unknown_not_negative",
            "model_calls": 0,
        },
        "privacy": "PRIVATE_DO_NOT_PUBLISH",
    }
    _atomic_text(
        output_dir / MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize provided thyroid clinical gold and deterministic exact spans."
        )
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-private-clinical-data", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = materialize_direct_clinical_gold(
        args.queue,
        args.output_dir,
        acknowledge_private_clinical_data=args.acknowledge_private_clinical_data,
    )
    counts = summary["counts"]
    print(
        "THYROID_DIRECT_CLINICAL_GOLD=COMPLETE "
        f"reports={counts['report_groups']} "
        f"gold={counts['clinical_gold_fields']} "
        f"localized={counts['span_localized_clinical_gold_fields']} "
        f"empty_unknown={counts['empty_unknown_slots']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
