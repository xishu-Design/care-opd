#!/usr/bin/env python3
"""Prepare a four-system, method-blind expert preference review package.

Every reviewer receives a separate wide CSV (and, when requested and
available, an XLSX workbook).  Candidate codes are reviewer- and item-local:
there is deliberately no global ``A == system`` mapping.  The only artifact
that binds visible candidate codes to private system labels is
``blind_mapping.private.jsonl``.

The script is intentionally strict.  It accepts exactly four complete system
response files and refuses missing tasks, unsuccessful generations, empty
display responses, task-payload binding mismatches, and response hash
mismatches.  It does not run models and it does not create ratings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "care_opd.competition.agent_preference_review.v1"
CANDIDATE_CODES = ("A", "B", "C", "D")
SUCCESS_STATUSES = {"success", "ok", "complete", "completed"}
DEFAULT_IDENTITY_LEAK_TERMS = (
    "神农三言",
    "guardtcm",
    "care-opd",
    "care opd",
    "cloudflare",
    "workers ai",
    "groq",
    "cerebras",
    "gemini",
    "qwen",
    "gpt-oss",
    "glm-4.7",
    "api.groq.com",
    "api.cerebras.ai",
    "api.cloudflare.com",
)
LOCAL_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/(?:home|users|tmp)/)", re.IGNORECASE)
RATING_COLUMNS = (
    "best_choice",
    "tied_candidates",
    "unsafe_candidates",
    "primary_reason",
    "confidence_1_3",
    "skip_reason_private",
    "review_notes_private",
    "completed_at_utc",
)
REVIEW_COLUMNS = (
    "reviewer_id",
    "display_order",
    "blind_item_id",
    "target_role",
    "evaluation_condition",
    "modality",
    "task_text_private",
    "answer_A",
    "answer_B",
    "answer_C",
    "answer_D",
    *RATING_COLUMNS,
)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _split_values(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _parse_response_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--response must have the form SYSTEM=PATH")
    system, raw_path = spec.split("=", 1)
    system = system.strip()
    if not system or not raw_path.strip():
        raise argparse.ArgumentTypeError("--response requires non-empty SYSTEM and PATH")
    _validate_label(system, "system")
    return system, Path(raw_path).expanduser().resolve()


def _validate_label(value: str, kind: str) -> None:
    if (
        len(value) > 100
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", "\x00", "\r", "\n"))
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"unsafe {kind} label: {value!r}")


def _first_nonempty(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _task_id(row: dict[str, Any]) -> str:
    value = _first_nonempty(row, ("task_id", "item_id", "case_id", "id"))
    if not value:
        raise ValueError("task row lacks task_id/item_id/case_id/id")
    return value


def _family_id(row: dict[str, Any]) -> str:
    value = _first_nonempty(
        row,
        (
            "family_token",
            "family_id_private",
            "report_group_hmac",
            "family_id",
        ),
    )
    if not value:
        raise ValueError(f"task {_task_id(row)!r} lacks a family-level cluster ID")
    return value


def _task_text(row: dict[str, Any]) -> str:
    for name in (
        "user_message_private",
        "prompt_private",
        "task_text_private",
        "clinical_context_private",
        "request",
    ):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (dict, list)) and value:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise ValueError(f"task {_task_id(row)!r} has no non-empty reviewer-visible task text")


def _response_text(row: dict[str, Any], system: str, task_id: str) -> str:
    for name in (
        "display_response_private",
        "response_private",
        "raw_response_private",
        "answer",
    ):
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"empty display response for system {system!r}, task {task_id!r}")


def _identity_leaks(text: str, terms: tuple[str, ...]) -> list[str]:
    folded = text.casefold()
    matches = sorted({term for term in terms if term.casefold() in folded})
    if LOCAL_PATH_RE.search(text):
        matches.append("LOCAL_PATH_PATTERN")
    return matches


def _index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _task_id(row)
        if key in result:
            raise ValueError(f"duplicate task {key!r} in {label}")
        result[key] = row
    return result


def _opaque(secret: bytes, namespace: str, value: str, length: int = 24) -> str:
    payload = f"{namespace}\x00{value}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()[:length]


def _hmac_order(secret: bytes, namespace: str, value: str) -> str:
    return _opaque(secret, namespace, value, length=64)


def _balanced_assignments(
    *,
    secret: bytes,
    reviewer: str,
    tasks: list[dict[str, Any]],
    systems: list[str],
) -> dict[str, dict[str, str]]:
    """Return item-local code->system maps balanced within role/condition.

    Each complete block of four tasks uses the four cyclic rotations of a
    secret-derived base permutation.  Therefore every system occupies every
    visible position exactly once per complete block.  Remainders differ by at
    most one position occurrence.
    """

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        role = _first_nonempty(task, ("role", "target_role", "intended_role"))
        variant = _first_nonempty(task, ("variant", "condition", "evaluation_condition"))
        if not role or not variant:
            raise ValueError(f"task {_task_id(task)!r} lacks non-empty role or variant")
        strata[(role, variant)].append(task)

    assignments: dict[str, dict[str, str]] = {}
    for (role, variant), stratum_rows in sorted(strata.items()):
        ordered = sorted(
            stratum_rows,
            key=lambda row: _hmac_order(
                secret,
                "stratum-item-order",
                f"{reviewer}|{role}|{variant}|{_task_id(row)}",
            ),
        )
        for block_start in range(0, len(ordered), len(CANDIDATE_CODES)):
            block = ordered[block_start : block_start + len(CANDIDATE_CODES)]
            block_number = block_start // len(CANDIDATE_CODES)
            base = sorted(
                systems,
                key=lambda system: _hmac_order(
                    secret,
                    "block-system-order",
                    f"{reviewer}|{role}|{variant}|{block_number}|{system}",
                ),
            )
            direction = 1 if int(
                _opaque(
                    secret,
                    "block-direction",
                    f"{reviewer}|{role}|{variant}|{block_number}",
                    length=2,
                ),
                16,
            ) % 2 == 0 else -1
            for offset, task in enumerate(block):
                shift = (direction * offset) % len(CANDIDATE_CODES)
                rotated = base[shift:] + base[:shift]
                assignments[_task_id(task)] = dict(zip(CANDIDATE_CODES, rotated))
    return assignments


def _write_xlsx(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        return False

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Expert Review"
    sheet.append(list(REVIEW_COLUMNS))
    for row in rows:
        sheet.append([row.get(column, "") for column in REVIEW_COLUMNS])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    widths = {
        "A": 16,
        "B": 10,
        "C": 22,
        "D": 14,
        "E": 24,
        "F": 16,
        "G": 50,
        "H": 50,
        "I": 50,
        "J": 50,
        "K": 50,
    }
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    for column in range(12, len(REVIEW_COLUMNS) + 1):
        sheet.column_dimensions[chr(64 + column)].width = 20
    first_data_row = 2
    last_data_row = max(first_data_row, len(rows) + 1)
    best_validation = DataValidation(
        type="list", formula1='"A,B,C,D,TIE,NONE,SKIP"', allow_blank=False
    )
    confidence_validation = DataValidation(
        type="list", formula1='"1,2,3"', allow_blank=True
    )
    sheet.add_data_validation(best_validation)
    sheet.add_data_validation(confidence_validation)
    best_column = REVIEW_COLUMNS.index("best_choice") + 1
    confidence_column = REVIEW_COLUMNS.index("confidence_1_3") + 1
    best_validation.add(
        f"{sheet.cell(1, best_column).column_letter}{first_data_row}:"
        f"{sheet.cell(1, best_column).column_letter}{last_data_row}"
    )
    confidence_validation.add(
        f"{sheet.cell(1, confidence_column).column_letter}{first_data_row}:"
        f"{sheet.cell(1, confidence_column).column_letter}{last_data_row}"
    )
    workbook.properties.creator = ""
    workbook.properties.lastModifiedBy = ""
    workbook.properties.title = "Expert Preference Review"
    workbook.save(path)
    return True


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    task_path = args.tasks.expanduser().resolve()
    if not task_path.is_file():
        raise FileNotFoundError(task_path)
    response_specs = [_parse_response_spec(spec) for spec in args.response]
    systems = [system for system, _ in response_specs]
    if len(response_specs) != 4 or len(set(systems)) != 4:
        raise ValueError("exactly four distinct --response SYSTEM=PATH inputs are required")
    reviewers = _split_values(args.reviewer)
    if not reviewers:
        raise ValueError("at least one distinct --reviewer is required")
    if len(reviewers) != len(set(reviewers)):
        raise ValueError("reviewer IDs must be distinct")
    for reviewer in reviewers:
        _validate_label(reviewer, "reviewer")
    for _, path in response_specs:
        if not path.is_file():
            raise FileNotFoundError(path)

    secret_name = args.blind_secret_env
    raw_secret = os.environ.get(secret_name, "")
    if len(raw_secret.encode("utf-8")) < 32:
        raise ValueError(
            f"environment variable {secret_name!r} must contain at least 32 UTF-8 bytes"
        )
    secret = raw_secret.encode("utf-8")

    task_rows = _read_jsonl(task_path)
    task_index = _index_unique(task_rows, "task input")
    normalized_tasks: list[dict[str, Any]] = []
    for task_id, task in task_index.items():
        role = _first_nonempty(task, ("role", "target_role", "intended_role"))
        variant = _first_nonempty(task, ("variant", "condition", "evaluation_condition"))
        if not role or not variant:
            raise ValueError(f"task {task_id!r} lacks non-empty role or variant")
        normalized_tasks.append(task)

    response_texts: dict[tuple[str, str], str] = {}
    custom_terms = tuple(_split_values(getattr(args, "identity_term", None)))
    identity_terms = tuple(
        dict.fromkeys(
            (
                *DEFAULT_IDENTITY_LEAK_TERMS,
                *systems,
                *(value.replace("_", " ") for value in systems),
                *custom_terms,
            )
        )
    )
    input_hashes: dict[str, str] = {"tasks": _sha256_file(task_path)}
    for system, path in response_specs:
        index = _index_unique(_read_jsonl(path), f"response system {system}")
        expected, observed = set(task_index), set(index)
        if expected != observed:
            raise ValueError(
                f"task coverage mismatch for {system}: "
                f"missing={len(expected - observed)}, extra={len(observed - expected)}"
            )
        for task_id, task in task_index.items():
            response = index[task_id]
            status = _first_nonempty(response, ("status", "generation_status")).lower()
            if status not in SUCCESS_STATUSES:
                raise ValueError(
                    f"unsuccessful or missing status for system {system!r}, task {task_id!r}: "
                    f"{status!r}"
                )
            expected_binding = _canonical_json_sha256(task)
            observed_binding = _first_nonempty(
                response,
                ("task_payload_sha256", "input_payload_sha256"),
            )
            if not observed_binding or not hmac.compare_digest(
                observed_binding.lower(), expected_binding.lower()
            ):
                raise ValueError(
                    f"task payload binding mismatch for system {system!r}, task {task_id!r}"
                )
            text = _response_text(response, system, task_id)
            leaks = _identity_leaks(text, identity_terms)
            if leaks:
                raise ValueError(
                    f"identity leakage detected for system {system!r}, task {task_id!r}: "
                    + ", ".join(leaks)
                )
            observed_hash = _first_nonempty(response, ("response_sha256",))
            actual_hash = _sha256_text(text)
            if observed_hash and not hmac.compare_digest(
                observed_hash.lower(), actual_hash.lower()
            ):
                raise ValueError(
                    f"response hash mismatch for system {system!r}, task {task_id!r}"
                )
            response_texts[(system, task_id)] = text
        input_hashes[f"response:{system}"] = _sha256_file(path)

    private_dir = args.output_private.expanduser().resolve()
    public_dir = (
        args.output_public.expanduser().resolve()
        if args.output_public
        else private_dir / "public_safe"
    )
    if getattr(args, "xlsx", False):
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--xlsx was requested but openpyxl is unavailable; install openpyxl "
                "before freezing the blind-review package"
            ) from exc
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    if private_dir == public_dir:
        raise ValueError("private and public output directories must differ")
    if any(private_dir.iterdir()) or any(public_dir.iterdir()):
        raise FileExistsError(
            "review output directories must be empty; use new directories so an existing blind "
            "mapping or completed rating sheet can never be overwritten"
        )

    mapping_rows: list[dict[str, Any]] = []
    reviewer_paths: dict[str, dict[str, Path]] = {}
    position_counts: dict[str, Counter[tuple[str, str, str, str]]] = {}
    for reviewer_index, reviewer in enumerate(reviewers, start=1):
        assignments = _balanced_assignments(
            secret=secret, reviewer=reviewer, tasks=normalized_tasks, systems=systems
        )
        reviewer_order = sorted(
            normalized_tasks,
            key=lambda task: _hmac_order(
                secret, "reviewer-item-order", f"{reviewer}|{_task_id(task)}"
            ),
        )
        review_rows: list[dict[str, Any]] = []
        counts: Counter[tuple[str, str, str, str]] = Counter()
        for display_order, task in enumerate(reviewer_order, start=1):
            task_id = _task_id(task)
            family = _family_id(task)
            role = _first_nonempty(task, ("role", "target_role", "intended_role"))
            variant = _first_nonempty(
                task, ("variant", "condition", "evaluation_condition")
            )
            modality = _first_nonempty(task, ("modality",)) or "unknown"
            task_text = _task_text(task)
            blind_item_id = "I-" + _opaque(
                secret, "reviewer-blind-item", f"{reviewer}|{task_id}"
            )
            code_to_system = assignments[task_id]
            row: dict[str, Any] = {
                "reviewer_id": reviewer,
                "display_order": display_order,
                "blind_item_id": blind_item_id,
                "target_role": role,
                "evaluation_condition": variant,
                "modality": modality,
                "task_text_private": task_text,
                **{column: "" for column in RATING_COLUMNS},
            }
            for code in CANDIDATE_CODES:
                system = code_to_system[code]
                answer = response_texts[(system, task_id)]
                row[f"answer_{code}"] = answer
                counts[(role, variant, system, code)] += 1
                mapping_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "reviewer_id_private": reviewer,
                        "blind_item_id": blind_item_id,
                        "task_id_private": task_id,
                        "family_id_private": family,
                        "target_role": role,
                        "evaluation_condition": variant,
                        "modality": modality,
                        "candidate_code": code,
                        "system_id_private": system,
                        "task_payload_sha256": _canonical_json_sha256(task),
                        "task_text_sha256": _sha256_text(task_text),
                        "response_sha256": _sha256_text(answer),
                    }
                )
            review_rows.append(row)

        csv_path = private_dir / f"reviewer_{reviewer_index}.private.csv"
        _write_csv(csv_path, review_rows)
        paths = {"csv": csv_path}
        if getattr(args, "xlsx", False):
            xlsx_path = private_dir / f"reviewer_{reviewer_index}.private.xlsx"
            if _write_xlsx(xlsx_path, review_rows):
                paths["xlsx"] = xlsx_path
        reviewer_paths[reviewer] = paths
        position_counts[reviewer] = counts

    # Hard-gate the promised within-stratum balance.
    for reviewer, counts in position_counts.items():
        strata = sorted({(role, variant) for role, variant, _, _ in counts})
        for role, variant in strata:
            for system in systems:
                values = [counts[(role, variant, system, code)] for code in CANDIDATE_CODES]
                if max(values) - min(values) > 1:
                    raise RuntimeError(
                        f"position balancing failed for reviewer={reviewer}, role={role}, "
                        f"variant={variant}, system={system}: {values}"
                    )

    mapping_rows.sort(
        key=lambda row: (
            reviewers.index(str(row["reviewer_id_private"])),
            str(row["blind_item_id"]),
            str(row["candidate_code"]),
        )
    )
    mapping_path = private_dir / "blind_mapping.private.jsonl"
    _write_jsonl(mapping_path, mapping_rows)

    created = datetime.now(timezone.utc).isoformat()
    private_manifest_path = private_dir / "review_manifest.private.json"
    private_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "status": "FOUR_SYSTEM_BLIND_REVIEW_PREPARED_NO_RATINGS",
        "privacy": "PRIVATE: contains system labels, reviewer IDs, input paths and hashes",
        "systems_private": systems,
        "reviewers_private": reviewers,
        "input_paths_private": {
            "tasks": str(task_path),
            "responses": {system: str(path) for system, path in response_specs},
        },
        "input_hashes": input_hashes,
        "blind_secret_fingerprint": hashlib.sha256(secret).hexdigest(),
        "identity_leak_scan": {
            "status": "PASS_ZERO_MATCHES",
            "terms_private": list(identity_terms),
            "local_path_pattern_checked": True,
        },
        "task_count": len(task_rows),
        "system_count": len(systems),
        "reviewer_count": len(reviewers),
        "expected_ballot_count": len(task_rows) * len(reviewers),
        "mapping_row_count": len(mapping_rows),
        "roles": sorted(
            {_first_nonempty(row, ("role", "target_role", "intended_role")) for row in task_rows}
        ),
        "variants": sorted(
            {
                _first_nonempty(row, ("variant", "condition", "evaluation_condition"))
                for row in task_rows
            }
        ),
        "candidate_codes": list(CANDIDATE_CODES),
        "rating_contract": {
            "best_choice": ["A", "B", "C", "D", "TIE", "NONE", "SKIP"],
            "tie": "2-4 distinct codes required; each selected system receives 1/k point",
            "none": "valid ballot; all systems receive zero points",
            "skip": "excluded ballot; non-empty skip_reason_private required",
            "unsafe_candidates": (
                "mandatory on non-SKIP ballots; use NONE when no candidate is unsafe; "
                "a candidate marked unsafe cannot be selected as best or tied best"
            ),
        },
        "analysis_lock": {
            "evaluation_unit": "end-to-end agent system",
            "primary_endpoint": (
                "safety-qualified expert first-choice share; ties split 1/k"
            ),
            "bootstrap_cluster": "family_id_private",
            "bootstrap_default_replicates": 10000,
            "primary_comparison": "primary system minus each of the three fixed comparators",
            "attribution_boundary": (
                "system-level preference does not isolate a causal CARE-OPD contribution"
            ),
        },
        "artifacts": {
            "mapping": mapping_path.name,
            "reviewer_templates": {
                reviewer: {kind: path.name for kind, path in paths.items()}
                for reviewer, paths in reviewer_paths.items()
            },
        },
        "artifact_sha256": {
            mapping_path.name: _sha256_file(mapping_path),
            **{
                path.name: _sha256_file(path)
                for paths in reviewer_paths.values()
                for path in paths.values()
            },
        },
    }
    _write_json(private_manifest_path, private_manifest)

    private_hashes = {
        "blind_mapping": _sha256_file(mapping_path),
        "review_manifest": _sha256_file(private_manifest_path),
    }
    for reviewer_index, reviewer in enumerate(reviewers, start=1):
        for kind, path in reviewer_paths[reviewer].items():
            private_hashes[f"reviewer_template_{reviewer_index}_{kind}"] = _sha256_file(path)
    public_manifest_path = public_dir / "public_protocol_manifest.json"
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "status": "FOUR_SYSTEM_BLIND_REVIEW_PREPARED_NO_RATINGS",
        "design": {
            "task_count": len(task_rows),
            "system_count": 4,
            "reviewer_count": len(reviewers),
            "expected_ballot_count": len(task_rows) * len(reviewers),
            "roles": private_manifest["roles"],
            "conditions": private_manifest["variants"],
            "reviewer_specific_item_local_randomization": True,
            "within_role_condition_position_imbalance_max": 1,
            "identity_leak_scan_zero_matches": True,
        },
        "private_artifact_sha256": private_hashes,
        "privacy_statement": (
            "Aggregate protocol metadata only: no system labels, reviewer identities, task or "
            "family identifiers, clinical text, candidate response, blind mapping, or rating is "
            "included."
        ),
    }
    _write_json(public_manifest_path, public_manifest)
    return {
        "mapping": mapping_path,
        "private_manifest": private_manifest_path,
        "public_manifest": public_manifest_path,
        "reviewer_files": reviewer_paths,
        "xlsx_requested": bool(getattr(args, "xlsx", False)),
        "xlsx_created_count": sum("xlsx" in paths for paths in reviewer_paths.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument(
        "--response",
        action="append",
        required=True,
        help="repeat exactly four times as SYSTEM=PATH",
    )
    parser.add_argument(
        "--reviewer", action="append", required=True, help="repeatable or comma-separated IDs"
    )
    parser.add_argument("--blind-secret-env", default="EXPERT_BLIND_SECRET")
    parser.add_argument(
        "--identity-term",
        action="append",
        help=(
            "additional case-insensitive provider/model/brand term to reject; repeatable or "
            "comma-separated"
        ),
    )
    parser.add_argument(
        "--xlsx", action="store_true", help="also create XLSX if openpyxl is installed"
    )
    parser.add_argument("--output-private", required=True, type=Path)
    parser.add_argument("--output-public", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    outputs = prepare(build_parser().parse_args(argv))
    safe = {
        "mapping": str(outputs["mapping"]),
        "private_manifest": str(outputs["private_manifest"]),
        "public_manifest": str(outputs["public_manifest"]),
        "reviewer_csv_count": len(outputs["reviewer_files"]),
        "xlsx_requested": outputs["xlsx_requested"],
        "xlsx_created_count": outputs["xlsx_created_count"],
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
