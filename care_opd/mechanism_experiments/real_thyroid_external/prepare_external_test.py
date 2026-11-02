#!/usr/bin/env python3
"""Prepare a private thyroid XLSX as a locked report-level external test.

The default mode is aggregate-only audit.  Private row-level JSONL is emitted
only when ``--private-export`` and the explicit acknowledgement flag are both
provided.  The program never logs cell values, stores the HMAC key, or invents
patient/encounter/lesion linkage.

Only the Python standard library is required; the small XLSX reader handles
shared-string and inline-string cells used by the source workbook.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN_NS, "r": DOC_REL_NS}

SCHEMA_VERSION = "care_opd.real_thyroid_external.v1"
DEFAULT_HMAC_ENV = "CARE_OPD_THYROID_HMAC_KEY"

ULTRASOUND_SHEET = "超声"
PATHOLOGY_SHEET = "病理"
ULTRASOUND_SOURCE = "检查报告单-检查结论（原始）"
PATHOLOGY_SOURCE = "检查报告单-检查结论（原始）"

ULTRASOUND_FIELDS: tuple[tuple[str, str], ...] = (
    ("结节部位", "location"),
    ("结节回声", "echo_classification_raw"),
    ("尺寸", "size.raw"),
    ("生长", "growth_descriptor_raw"),
    ("形状", "shape"),
    ("边缘", "margin"),
    ("内部回声", "internal_echo"),
    ("强回声", "echogenic_foci"),
    ("后方回声改变", "posterior_acoustic_features"),
    ("血流信号", "vascularity"),
)

PATHOLOGY_FIELDS: tuple[tuple[str, str], ...] = (
    ("标本部位", "specimen.site"),
    ("标本尺寸", "specimen.size"),
    ("结节大小", "gross.nodule_size"),
    ("甲状腺结节颜色", "gross.nodule_color"),
    ("钙化", "gross.calcification_reported"),
    ("病理诊断", "diagnostic_events.primary_pathology"),
    ("病理肿瘤直径", "diagnostic_events.primary_tumor_diameter"),
    ("冰冻报告诊断", "diagnostic_events.frozen_section"),
    ("甲状腺补充报告诊断", "diagnostic_events.thyroid_supplement"),
    ("补充病理诊断", "diagnostic_events.pathology_supplement"),
    ("补充病理肿瘤直径", "diagnostic_events.supplemental_tumor_diameter"),
    ("CK19", "ihc.CK19.raw"),
    ("TPO", "ihc.TPO.raw"),
    ("CgA", "ihc.CgA.raw"),
    ("TTF-1", "ihc.TTF-1.raw"),
    ("Ki-67", "ihc.Ki-67.raw"),
    ("Bcl-2", "ihc.Bcl-2.raw"),
    ("Tg", "ihc.Tg.raw"),
    ("Galectin3", "ihc.Galectin3.raw"),
)

QUALITY_NOTE = "结构化提取备注"
QUALITY_GRADE = "数据质量等级"


class WorkbookContractError(ValueError):
    """Raised for workbook structure problems without echoing cell values."""


@dataclass(frozen=True)
class SheetRow:
    row_number: int
    values: Mapping[str, str]


@dataclass(frozen=True)
class SheetData:
    name: str
    headers: tuple[str, ...]
    rows: tuple[SheetRow, ...]


@dataclass(frozen=True)
class CandidateField:
    field_path: str
    candidate_value_private: str | None


@dataclass(frozen=True)
class CandidateEntity:
    entity_id: str
    fields: tuple[CandidateField, ...]


@dataclass(frozen=True)
class ReportGroup:
    modality: str
    ordinal: int
    source_text_private: str
    worksheet_rows: tuple[int, ...]
    entities: tuple[CandidateEntity, ...]
    quality_grade_private: str | None = None
    extraction_note_private: str | None = None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _cell_column(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha()).upper()
    if not letters:
        raise WorkbookContractError("A worksheet cell is missing its column reference.")
    number = 0
    for character in letters:
        number = number * 26 + ord(character) - 64
    return number


def _normalize_archive_target(target: str) -> str:
    path = PurePosixPath("xl") / PurePosixPath(target)
    parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise WorkbookContractError("Workbook relationship escapes the XLSX archive root.")
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.iter() if _local_name(node.tag) == "t")
        for item in root.findall("m:si", NS)
    ]


def _sheet_inventory(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except KeyError as error:
        raise WorkbookContractError("The XLSX workbook structure is incomplete.") from error
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relations.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    inventory: list[tuple[str, str]] = []
    for sheet in workbook.findall("m:sheets/m:sheet", NS):
        relation_id = sheet.attrib.get(f"{{{DOC_REL_NS}}}id")
        if not relation_id or relation_id not in targets:
            raise WorkbookContractError("A worksheet relationship is missing.")
        inventory.append(
            (sheet.attrib.get("name", ""), _normalize_archive_target(targets[relation_id]))
        )
    return inventory


def _decode_cell(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    kind = cell.attrib.get("t", "n")
    if kind == "inlineStr":
        value = "".join(
            node.text or "" for node in cell.iter() if _local_name(node.tag) == "t"
        )
    else:
        node = cell.find("m:v", NS)
        value = "" if node is None or node.text is None else node.text
        if kind == "s" and value:
            try:
                value = shared_strings[int(value)]
            except (IndexError, ValueError) as error:
                raise WorkbookContractError("A shared-string cell has an invalid index.") from error
        elif kind == "b":
            value = "TRUE" if value == "1" else "FALSE"
    return value.strip()


def _parse_sheet(xml_bytes: bytes, shared_strings: Sequence[str], name: str) -> SheetData:
    root = ET.fromstring(xml_bytes)
    decoded_rows: list[tuple[int, dict[int, str]]] = []
    for row_index, row in enumerate(root.findall("m:sheetData/m:row", NS), start=1):
        row_number = int(row.attrib.get("r", row_index))
        values: dict[int, str] = {}
        for cell in row.findall("m:c", NS):
            values[_cell_column(cell.attrib.get("r", ""))] = _decode_cell(
                cell, shared_strings
            )
        if any(values.values()):
            decoded_rows.append((row_number, values))
    if not decoded_rows:
        raise WorkbookContractError(f"Required worksheet '{name}' is empty.")

    header_row_number, raw_headers = decoded_rows[0]
    max_column = max(raw_headers)
    headers = tuple(raw_headers.get(column, "") for column in range(1, max_column + 1))
    if any(not header for header in headers):
        raise WorkbookContractError(f"Worksheet '{name}' contains an unnamed required column.")
    if len(set(headers)) != len(headers):
        raise WorkbookContractError(f"Worksheet '{name}' contains duplicate headers.")

    rows: list[SheetRow] = []
    for row_number, raw_values in decoded_rows[1:]:
        if row_number <= header_row_number:
            continue
        values = {
            header: raw_values.get(column, "")
            for column, header in enumerate(headers, start=1)
        }
        if any(values.values()):
            rows.append(SheetRow(row_number=row_number, values=values))
    return SheetData(name=name, headers=headers, rows=tuple(rows))


def read_xlsx(path: Path) -> dict[str, SheetData]:
    """Read the workbook without third-party dependencies."""

    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _read_shared_strings(archive)
            sheets = {
                name: _parse_sheet(archive.read(target), shared_strings, name)
                for name, target in _sheet_inventory(archive)
            }
    except (OSError, zipfile.BadZipFile, ET.ParseError, KeyError) as error:
        raise WorkbookContractError("The input is not a readable XLSX workbook.") from error
    return sheets


def _require_headers(sheet: SheetData, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(sheet.headers))
    if missing:
        raise WorkbookContractError(
            f"Worksheet '{sheet.name}' is missing required headers: {', '.join(missing)}"
        )


def _field_candidates(
    values: Mapping[str, str], mapping: Sequence[tuple[str, str]]
) -> tuple[CandidateField, ...]:
    return tuple(
        CandidateField(field_path=field_path, candidate_value_private=values[header] or None)
        for header, field_path in mapping
    )


def _group_ultrasound(sheet: SheetData) -> tuple[ReportGroup, ...]:
    required = [ULTRASOUND_SOURCE, "结节编号", *(header for header, _ in ULTRASOUND_FIELDS)]
    _require_headers(sheet, required)

    groups: list[ReportGroup] = []
    source_text: str | None = None
    worksheet_rows: list[int] = []
    entities: list[CandidateEntity] = []

    def finish_group() -> None:
        nonlocal source_text, worksheet_rows, entities
        if source_text is None:
            return
        groups.append(
            ReportGroup(
                modality="thyroid_ultrasound",
                ordinal=len(groups) + 1,
                source_text_private=source_text,
                worksheet_rows=tuple(worksheet_rows),
                entities=tuple(entities),
            )
        )

    structured_headers = ["结节编号", *(header for header, _ in ULTRASOUND_FIELDS)]
    for row in sheet.rows:
        new_source = row.values[ULTRASOUND_SOURCE]
        if new_source:
            finish_group()
            source_text = new_source
            worksheet_rows = []
            entities = []
        elif source_text is None:
            raise WorkbookContractError(
                f"Ultrasound continuation row {row.row_number} appears before any source report."
            )
        worksheet_rows.append(row.row_number)
        if any(row.values[header] for header in structured_headers):
            entities.append(
                CandidateEntity(
                    entity_id=f"N{len(entities) + 1:02d}",
                    fields=_field_candidates(row.values, ULTRASOUND_FIELDS),
                )
            )
    finish_group()
    return tuple(groups)


def _group_pathology(sheet: SheetData) -> tuple[ReportGroup, ...]:
    required = [
        PATHOLOGY_SOURCE,
        *(header for header, _ in PATHOLOGY_FIELDS),
        QUALITY_NOTE,
        QUALITY_GRADE,
    ]
    _require_headers(sheet, required)
    groups: list[ReportGroup] = []
    for row in sheet.rows:
        if not row.values[PATHOLOGY_SOURCE]:
            raise WorkbookContractError(
                f"Pathology data row {row.row_number} has no source report; linkage is not inferred."
            )
        groups.append(
            ReportGroup(
                modality="thyroid_pathology",
                ordinal=len(groups) + 1,
                source_text_private=row.values[PATHOLOGY_SOURCE],
                worksheet_rows=(row.row_number,),
                entities=(
                    CandidateEntity(
                        entity_id="REPORT",
                        fields=_field_candidates(row.values, PATHOLOGY_FIELDS),
                    ),
                ),
                quality_grade_private=row.values[QUALITY_GRADE] or None,
                extraction_note_private=row.values[QUALITY_NOTE] or None,
            )
        )
    return tuple(groups)


def load_report_groups(path: Path) -> tuple[ReportGroup, ...]:
    """Load report-level groups without creating cross-report linkage."""

    sheets = read_xlsx(path)
    missing_sheets = sorted({ULTRASOUND_SHEET, PATHOLOGY_SHEET} - set(sheets))
    if missing_sheets:
        raise WorkbookContractError(
            f"Workbook is missing required worksheets: {', '.join(missing_sheets)}"
        )
    return _group_ultrasound(sheets[ULTRASOUND_SHEET]) + _group_pathology(
        sheets[PATHOLOGY_SHEET]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hmac_hex(secret: bytes, *parts: str) -> str:
    message = "\x1f".join(parts).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _load_hmac_secret(
    *,
    environment_name: str,
    key_file: Path | None,
    generate_key_file: bool,
) -> bytes:
    """Load a private HMAC secret without printing or persisting it implicitly."""

    environment_value = os.environ.get(environment_name, "")
    if environment_value and key_file is not None:
        raise PermissionError("Use either the HMAC environment variable or --hmac-key-file.")
    if generate_key_file and key_file is None:
        raise PermissionError("--generate-hmac-key-file requires --hmac-key-file.")
    if environment_value:
        secret = environment_value.encode("utf-8")
    elif key_file is not None:
        key_file = key_file.expanduser().resolve()
        if generate_key_file:
            if key_file.exists():
                raise FileExistsError("Refusing to overwrite the existing HMAC key file.")
            key_file.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
                    handle.write(secrets.token_hex(32) + "\n")
            except BaseException:
                key_file.unlink(missing_ok=True)
                raise
        if not key_file.is_file():
            raise PermissionError("The HMAC key file does not exist.")
        secret = key_file.read_text(encoding="ascii").strip().encode("ascii")
    else:
        secret = b""
    if len(secret) < 32:
        raise PermissionError(
            f"Provide at least 32 secret bytes via {environment_name} or --hmac-key-file."
        )
    return secret


def _identifiers(group: ReportGroup, workbook_sha256: str, secret: bytes) -> dict[str, str]:
    source_hmac = _hmac_hex(
        secret,
        SCHEMA_VERSION,
        workbook_sha256,
        group.modality,
        str(group.ordinal),
        group.source_text_private,
    )
    prefix = "US" if group.modality == "thyroid_ultrasound" else "PATH"
    case_id = f"THY-{prefix}-{source_hmac[:24]}"
    return {
        "case_id": case_id,
        "report_group_hmac": f"rg_{source_hmac}",
        "source_id": f"{case_id}-SRC",
        "source_record_hmac": f"src_{_hmac_hex(secret, 'source', group.source_text_private)}",
    }


def _field_record(field: CandidateField) -> dict[str, Any]:
    has_candidate = field.candidate_value_private is not None
    return {
        "field_path": field.field_path,
        "candidate_value_private": field.candidate_value_private,
        "candidate_origin": "xlsx_structured_column",
        "candidate_status": "reported_candidate" if has_candidate else "unresolved_missingness",
        "support_status": "unreviewed",
        "evidence_spans": [],
        "normalized_value": None,
        "uncertainty_reason": None,
    }


def build_private_records(
    groups: Sequence[ReportGroup],
    *,
    workbook_sha256: str,
    secret: bytes,
    deidentification_status: str = "unreviewed",
    authorization_reference: str | None = None,
) -> list[dict[str, Any]]:
    """Build private canonical records; patient-level keys intentionally stay null."""

    if len(secret) < 32:
        raise ValueError("The HMAC secret must contain at least 32 bytes.")
    if deidentification_status not in {"unreviewed", "reviewed_private_use"}:
        raise ValueError("Unsupported deidentification status.")

    records: list[dict[str, Any]] = []
    for group in groups:
        ids = _identifiers(group, workbook_sha256, secret)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                **ids,
                "modality": group.modality,
                "split": "locked_external_test",
                "linkage": {
                    "status": "unavailable_do_not_infer",
                    "patient_group_hmac": None,
                    "encounter_group_hmac": None,
                    "lesion_group_hmac": None,
                    "site_code": None,
                    "time_bucket": None,
                },
                "source": {
                    "source_text_private": group.source_text_private,
                    "deidentification_status": deidentification_status,
                    "model_input_allowed": deidentification_status == "reviewed_private_use",
                },
                "entities": [
                    {
                        "entity_id": entity.entity_id,
                        "entity_status": "candidate_unreviewed",
                        "fields": [_field_record(field) for field in entity.fields],
                    }
                    for entity in group.entities
                ],
                "quality_private": {
                    "grade": group.quality_grade_private,
                    "extraction_note": group.extraction_note_private,
                    "excluded_from_model_input": True,
                },
                "annotation": {
                    "status": "agent_pending_then_clinician_review",
                    "critical_fact_keys": [],
                    "must_not_claim": [
                        "patient_or_encounter_linkage",
                        "cross_modal_lesion_match",
                        "diagnosis_not_explicitly_supported_by_source",
                        "negative_finding_from_blank_structured_cell",
                    ],
                },
                "governance": {
                    "contains_real_patient_data": True,
                    "private_only": True,
                    "redistribution_allowed": False,
                    "authorization_reference": authorization_reference,
                },
                "private_provenance": {
                    "worksheet_rows": list(group.worksheet_rows),
                    "workbook_sha256": workbook_sha256,
                },
            }
        )
    return records


def _agent_field_task(entity_id: str, field: CandidateField) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "field_path": field.field_path,
        "raw_value": field.candidate_value_private,
        "required_outputs": {
            "support_status": "pending",
            "value_status": "pending",
            "evidence_spans": [],
            "span_match_type": "pending_exact_or_approved_alias",
            "uncertainty_reason": None,
            "needs_clinician_review": True,
        },
        "instructions": [
            "Locate exact source-text spans for every supported non-null value.",
            "Treat a blank candidate as unresolved, not as a negative finding.",
            "If the source is ambiguous, preserve the candidate and flag uncertainty.",
            "Do not convert a descriptive phrase into a new diagnosis.",
        ],
    }


def build_agent_queue(
    groups: Sequence[ReportGroup],
    *,
    workbook_sha256: str,
    secret: bytes,
    agent_mode: str = "candidate_auditor",
) -> list[dict[str, Any]]:
    """Create stable work packets for an approved private/local annotation agent.

    ``blind_extractor`` deliberately receives no sheet candidates.  The
    ``candidate_auditor`` receives the flattened candidate fields.  Both modes
    use the same queue-row schema so a local runner can dispatch either file.
    """

    if len(secret) < 32:
        raise ValueError("The HMAC secret must contain at least 32 bytes.")
    if agent_mode not in {"blind_extractor", "candidate_auditor"}:
        raise ValueError("Unsupported agent mode.")
    queue: list[dict[str, Any]] = []
    for group in groups:
        ids = _identifiers(group, workbook_sha256, secret)
        candidate_fields = [
            _agent_field_task(entity.entity_id, field)
            for entity in group.entities
            for field in entity.fields
        ]
        if agent_mode == "blind_extractor":
            candidate_fields = []
        template_paths = (
            [path for _, path in ULTRASOUND_FIELDS]
            if group.modality == "thyroid_ultrasound"
            else [path for _, path in PATHOLOGY_FIELDS]
        )
        queue.append(
            {
                "schema_version": f"{SCHEMA_VERSION}.agent_queue",
                "task_id": f"ann_{_hmac_hex(secret, 'annotation', ids['case_id'])[:24]}",
                "case_id": ids["case_id"],
                "report_group_hmac": ids["report_group_hmac"],
                "modality": group.modality,
                "source_text_private": group.source_text_private,
                "source_record_sha256": hashlib.sha256(
                    group.source_text_private.encode("utf-8")
                ).hexdigest(),
                "agent_mode": agent_mode,
                "candidate_visibility": (
                    "hidden" if agent_mode == "blind_extractor" else "visible"
                ),
                "candidate_entities": [
                    {
                        "entity_id": entity.entity_id,
                        "field_paths": [field.field_path for field in entity.fields],
                    }
                    for entity in group.entities
                ],
                "entity_review": {
                    "candidate_entity_count": len(group.entities),
                    "status": "pending",
                    "allowed_actions": [
                        "confirm_candidate_entities",
                        "split_or_merge_candidates_with_exact_span_evidence",
                        "add_source_supported_entity_with_exact_span_evidence",
                    ],
                    "field_template_paths": template_paths,
                },
                "candidate_fields": candidate_fields,
                "agent_boundary": {
                    "execution": "approved_private_local_agent_only",
                    "may": [
                        "locate_exact_evidence_spans",
                        "confirm_source_supported_extraction",
                        "resolve_missingness_semantics_when_explicit",
                        "flag_ambiguity_and_uncertainty",
                    ],
                    "must_not": [
                        "infer_patient_encounter_or_lesion_linkage",
                        "infer_cross_modal_correspondence",
                        "invent_clinical_labels_or_final_diagnosis",
                        "infer_negative_findings_from_empty_cells",
                        "make_treatment_recommendations",
                    ],
                    "clinical_gold_requires_human_adjudication": True,
                },
                "acceptance_policy": {
                    "primary_gold_requires": [
                        "sheet_candidate_present",
                        "candidate_auditor_exact_or_approved_alias_span",
                        "independent_blind_extractor_agreement",
                    ],
                    "disagreement_action": "clinician_adjudication",
                    "blank_sheet_cell_is_negative": False,
                },
            }
        )
    return queue


def _atomic_write(path: Path, text: str, *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
        if private:
            try:
                path.chmod(0o600)
            except OSError:
                pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _aggregate_summary(
    path: Path, groups: Sequence[ReportGroup], *, private_export: bool
) -> dict[str, Any]:
    ultrasound = [group for group in groups if group.modality == "thyroid_ultrasound"]
    pathology = [group for group in groups if group.modality == "thyroid_pathology"]
    ultrasound_rows = sum(len(group.worksheet_rows) for group in ultrasound)
    ultrasound_entities = sum(len(group.entities) for group in ultrasound)
    populated_fields = sum(
        field.candidate_value_private is not None
        for group in groups
        for entity in group.entities
        for field in entity.fields
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}.audit",
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "absolute_path_recorded": False,
            "file_name_recorded": False,
        },
        "counts": {
            "report_groups_total": len(groups),
            "ultrasound_report_groups": len(ultrasound),
            "ultrasound_worksheet_rows": ultrasound_rows,
            "ultrasound_continuation_rows": ultrasound_rows - len(ultrasound),
            "ultrasound_candidate_entities": ultrasound_entities,
            "pathology_report_groups": len(pathology),
            "candidate_fields_populated": populated_fields,
        },
        "mode": "private_export" if private_export else "audit_only",
        "privacy": {
            "raw_cell_values_in_audit": False,
            "raw_cell_values_in_console": False,
            "hmac_secret_persisted": False,
            "patient_linkage_inferred": False,
            "clinical_labels_inferred": False,
        },
        "claim_boundary": (
            "Report-level external extraction preparation only; no patient-level independence, "
            "cross-modal linkage, diagnosis accuracy, or clinical validity is established."
        ),
    }


def prepare_external_test(
    xlsx_path: Path,
    output_dir: Path,
    *,
    private_export: bool = False,
    acknowledge_private_clinical_data: bool = False,
    hmac_key_env: str = DEFAULT_HMAC_ENV,
    hmac_key_file: Path | None = None,
    generate_hmac_key_file: bool = False,
    expected_ultrasound_reports: int | None = 23,
    expected_pathology_reports: int | None = 49,
    deidentification_status: str = "unreviewed",
    authorization_reference: str | None = None,
) -> dict[str, Any]:
    """Audit the workbook and optionally emit private canonical/agent JSONL."""

    xlsx_path = xlsx_path.resolve()
    output_dir = output_dir.resolve()
    groups = load_report_groups(xlsx_path)
    ultrasound_count = sum(group.modality == "thyroid_ultrasound" for group in groups)
    pathology_count = sum(group.modality == "thyroid_pathology" for group in groups)
    if expected_ultrasound_reports is not None and ultrasound_count != expected_ultrasound_reports:
        raise WorkbookContractError(
            "Ultrasound report-group count does not match the locked workbook contract: "
            f"expected {expected_ultrasound_reports}, observed {ultrasound_count}."
        )
    if expected_pathology_reports is not None and pathology_count != expected_pathology_reports:
        raise WorkbookContractError(
            "Pathology report-group count does not match the locked workbook contract: "
            f"expected {expected_pathology_reports}, observed {pathology_count}."
        )

    summary = _aggregate_summary(xlsx_path, groups, private_export=private_export)
    _atomic_write(
        output_dir / "audit_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        private=False,
    )
    if not private_export:
        return summary

    if not acknowledge_private_clinical_data:
        raise PermissionError(
            "Private export requires --acknowledge-private-clinical-data."
        )
    secret = _load_hmac_secret(
        environment_name=hmac_key_env,
        key_file=hmac_key_file,
        generate_key_file=generate_hmac_key_file,
    )

    workbook_sha256 = summary["source"]["sha256"]
    records = build_private_records(
        groups,
        workbook_sha256=workbook_sha256,
        secret=secret,
        deidentification_status=deidentification_status,
        authorization_reference=authorization_reference,
    )
    queue = build_agent_queue(
        groups,
        workbook_sha256=workbook_sha256,
        secret=secret,
        agent_mode="candidate_auditor",
    )
    _atomic_write(output_dir / ".gitignore", "*\n!.gitignore\n", private=False)
    _atomic_write(
        output_dir / "canonical_records.private.jsonl", _jsonl(records), private=True
    )
    _atomic_write(output_dir / "agent_work_queue.private.jsonl", _jsonl(queue), private=True)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.private_manifest",
        "created_utc": summary["created_utc"],
        "workbook_sha256": workbook_sha256,
        "record_count": len(records),
        "agent_task_count": len(queue),
        "canonical_sha256": hashlib.sha256(_jsonl(records).encode("utf-8")).hexdigest(),
        "agent_queue_sha256": hashlib.sha256(_jsonl(queue).encode("utf-8")).hexdigest(),
        "contains_raw_clinical_text": False,
        "describes_private_files_that_contain_raw_clinical_text": True,
        "patient_linkage_inferred": False,
        "clinical_labels_inferred": False,
    }
    _atomic_write(
        output_dir / "private_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        private=True,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or privately canonicalize the locked thyroid external-test XLSX."
    )
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--private-export",
        action="store_true",
        help="Emit private canonical and agent work-queue JSONL (default: audit only).",
    )
    parser.add_argument("--acknowledge-private-clinical-data", action="store_true")
    parser.add_argument("--hmac-key-env", default=DEFAULT_HMAC_ENV)
    parser.add_argument(
        "--hmac-key-file",
        type=Path,
        help="Private file containing the stable HMAC key (never logged).",
    )
    parser.add_argument(
        "--generate-hmac-key-file",
        action="store_true",
        help="Create --hmac-key-file once with a cryptographically random key.",
    )
    parser.add_argument("--expect-ultrasound-reports", type=int, default=23)
    parser.add_argument("--expect-pathology-reports", type=int, default=49)
    parser.add_argument(
        "--deidentification-status",
        choices=("unreviewed", "reviewed_private_use"),
        default="unreviewed",
    )
    parser.add_argument("--authorization-reference")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = prepare_external_test(
        args.xlsx,
        args.output_dir,
        private_export=args.private_export,
        acknowledge_private_clinical_data=args.acknowledge_private_clinical_data,
        hmac_key_env=args.hmac_key_env,
        hmac_key_file=args.hmac_key_file,
        generate_hmac_key_file=args.generate_hmac_key_file,
        expected_ultrasound_reports=args.expect_ultrasound_reports,
        expected_pathology_reports=args.expect_pathology_reports,
        deidentification_status=args.deidentification_status,
        authorization_reference=args.authorization_reference,
    )
    counts = summary["counts"]
    print(
        "THYROID_EXTERNAL_PREP=PASS "
        f"mode={summary['mode']} reports={counts['report_groups_total']} "
        f"ultrasound={counts['ultrasound_report_groups']} "
        f"pathology={counts['pathology_report_groups']} raw_values_logged=0"
    )


if __name__ == "__main__":
    main()
