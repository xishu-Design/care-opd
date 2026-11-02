#!/usr/bin/env python3
"""Privacy-preserving XLSX schema audit using only the Python standard library.

The report contains headers and aggregate statistics, never cell-level values.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import statistics
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN, "r": REL}

DIRECT_ID_TERMS = (
    "姓名", "身份证", "手机号", "手机", "电话", "住址", "地址", "邮箱", "微信",
    "病历号", "门诊号", "住院号", "医保号", "患者id", "patient_id", "patientid",
    "medical_record", "mrn", "出生日期", "生日", "name", "email", "phone",
)
LINKAGE_TERMS = (
    "患者", "病例", "就诊", "检查号", "报告号", "case_id", "encounter", "record_id",
)
DATE_RE = re.compile(r"^(?:19|20)\d{2}[-/.年](?:0?[1-9]|1[0-2])[-/.月](?:0?[1-9]|[12]\d|3[01])日?$")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_RE = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def cell_column(reference: str) -> int:
    letters = "".join(ch for ch in reference if ch.isalpha()).upper()
    result = 0
    for char in letters:
        result = result * 26 + ord(char) - 64
    return result


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for item in root.findall("m:si", NS):
        values.append("".join(node.text or "" for node in item.iter() if local_name(node.tag) == "t"))
    return values


def sheet_inventory(archive: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in rels.findall(f"{{{PKG_REL}}}Relationship")
    }
    sheets = []
    for sheet in workbook.findall("m:sheets/m:sheet", NS):
        rid = sheet.attrib[f"{{{REL}}}id"]
        target = PurePosixPath("xl") / PurePosixPath(targets[rid])
        normalized = str(target)
        while "/../" in normalized:
            parts = []
            for part in PurePosixPath(normalized).parts:
                if part == "..":
                    parts.pop()
                else:
                    parts.append(part)
            normalized = "/".join(parts)
        sheets.append((sheet.attrib.get("name", ""), normalized, sheet.attrib.get("state", "visible")))
    return sheets


def decode_cell(cell: ET.Element, shared: list[str]) -> tuple[str, str, bool]:
    kind = cell.attrib.get("t", "n")
    formula = cell.find("m:f", NS) is not None
    if kind == "inlineStr":
        value = "".join(node.text or "" for node in cell.iter() if local_name(node.tag) == "t")
    else:
        node = cell.find("m:v", NS)
        value = "" if node is None or node.text is None else node.text
        if kind == "s" and value:
            index = int(value)
            value = shared[index] if 0 <= index < len(shared) else ""
        elif kind == "b":
            value = "TRUE" if value == "1" else "FALSE"
    return value.strip(), kind, formula


def value_kind(value: str) -> str:
    if not value:
        return "empty"
    if value.upper() in {"TRUE", "FALSE"}:
        return "boolean"
    try:
        float(value)
        return "numeric"
    except ValueError:
        pass
    if DATE_RE.match(value):
        return "date_string"
    if (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]")):
        try:
            json.loads(value)
            return "json"
        except json.JSONDecodeError:
            pass
    length = len(value)
    if length <= 32:
        return "short_text"
    if length <= 200:
        return "medium_text"
    return "long_text"


def header_risk(header: str) -> list[str]:
    lowered = header.lower().replace(" ", "")
    flags = []
    if any(term in lowered for term in DIRECT_ID_TERMS):
        flags.append("direct_identifier_column")
    if any(term in lowered for term in LINKAGE_TERMS):
        flags.append("linkage_key_column")
    return flags


def audit_sheet(xml_bytes: bytes, shared: list[str], name: str, state: str) -> dict:
    root = ET.fromstring(xml_bytes)
    rows: list[tuple[int, dict[int, tuple[str, str, bool]]]] = []
    merged_ranges = [node.attrib.get("ref", "") for node in root.findall("m:mergeCells/m:mergeCell", NS)]
    max_column = 0
    for row in root.findall("m:sheetData/m:row", NS):
        row_number = int(row.attrib.get("r", len(rows) + 1))
        decoded: dict[int, tuple[str, str, bool]] = {}
        for cell in row.findall("m:c", NS):
            column = cell_column(cell.attrib.get("r", ""))
            max_column = max(max_column, column)
            decoded[column] = decode_cell(cell, shared)
        rows.append((row_number, decoded))

    nonempty_rows = [(number, row) for number, row in rows if any(value[0] for value in row.values())]
    if not nonempty_rows:
        return {
            "sheet": name,
            "state": state,
            "rows": 0,
            "columns": 0,
            "merged_range_count": len(merged_ranges),
            "column_audit": [],
        }

    header_row_number, header_row = nonempty_rows[0]
    headers = {}
    for column in range(1, max_column + 1):
        raw = header_row.get(column, ("", "", False))[0]
        headers[column] = raw if raw else f"__unnamed_col_{column}"

    data_rows = [(number, row) for number, row in nonempty_rows if number > header_row_number]
    columns = []
    for column in range(1, max_column + 1):
        header = headers[column]
        values = [row.get(column, ("", "", False))[0] for _, row in data_rows]
        nonempty = [value for value in values if value]
        hashes = {hashlib.sha256(value.encode("utf-8")).hexdigest() for value in nonempty}
        lengths = [len(value) for value in nonempty]
        kinds = collections.Counter(value_kind(value) for value in nonempty)
        formula_count = sum(bool(row.get(column, ("", "", False))[2]) for _, row in data_rows)
        pattern_hits = {
            "phone_like": sum(bool(PHONE_RE.search(value)) for value in nonempty),
            "national_id_like": sum(bool(ID_RE.search(value)) for value in nonempty),
            "email_like": sum(bool(EMAIL_RE.search(value)) for value in nonempty),
        }
        columns.append(
            {
                "column_index": column,
                "header": header,
                "header_risk_flags": header_risk(header),
                "rows": len(data_rows),
                "nonempty": len(nonempty),
                "missing": len(data_rows) - len(nonempty),
                "missing_rate": round((len(data_rows) - len(nonempty)) / max(len(data_rows), 1), 6),
                "unique_nonempty": len(hashes),
                "unique_ratio": round(len(hashes) / max(len(nonempty), 1), 6),
                "type_counts": dict(sorted(kinds.items())),
                "length_min": min(lengths) if lengths else 0,
                "length_median": statistics.median(lengths) if lengths else 0,
                "length_max": max(lengths) if lengths else 0,
                "formula_count": formula_count,
                "pattern_hits": pattern_hits,
            }
        )

    return {
        "sheet": name,
        "state": state,
        "xml_rows": len(rows),
        "header_row": header_row_number,
        "data_rows": len(data_rows),
        "columns": max_column,
        "merged_range_count": len(merged_ranges),
        "column_audit": columns,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Real medical XLSX schema and privacy audit",
        "",
        "This report contains headers and aggregate statistics only; no cell-level values are reproduced.",
        "",
        "## Workbook",
        "",
        f"- File: `{report['source']['file_name']}`",
        f"- Bytes: `{report['source']['bytes']}`",
        f"- SHA-256: `{report['source']['sha256']}`",
        f"- Sheets: `{len(report['sheets'])}`",
        "",
    ]
    for sheet in report["sheets"]:
        lines.extend(
            [
                f"## Sheet: {sheet['sheet']}",
                "",
                f"- State: `{sheet['state']}`",
                f"- Header row: `{sheet.get('header_row', 'N/A')}`",
                f"- Data rows: `{sheet.get('data_rows', 0)}`",
                f"- Columns: `{sheet.get('columns', 0)}`",
                f"- Merged ranges: `{sheet.get('merged_range_count', 0)}`",
                "",
                "| # | Header | Nonempty | Missing | Unique | Types | Identifier flags | Pattern hits |",
                "|---:|---|---:|---:|---:|---|---|---|",
            ]
        )
        for column in sheet.get("column_audit", []):
            kinds = ", ".join(f"{key}:{value}" for key, value in column["type_counts"].items()) or "-"
            flags = ", ".join(column["header_risk_flags"]) or "-"
            hits = ", ".join(f"{key}:{value}" for key, value in column["pattern_hits"].items() if value) or "-"
            safe_header = column["header"].replace("|", "\\|")
            lines.append(
                f"| {column['column_index']} | {safe_header} | {column['nonempty']} | "
                f"{column['missing']} ({column['missing_rate']:.1%}) | {column['unique_nonempty']} | "
                f"{kinds} | {flags} | {hits} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Claim boundary",
            "",
            "This audit establishes workbook shape and potential privacy risks only. It does not validate clinical correctness, consent, de-identification quality, annotation quality, or redistribution rights.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.xlsx.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source) as archive:
        shared = read_shared_strings(archive)
        sheets = [
            audit_sheet(archive.read(target), shared, name, state)
            for name, target, state in sheet_inventory(archive)
        ]
        core_props_present = "docProps/core.xml" in archive.namelist()

    report = {
        "schema_version": "care_opd.real_xlsx_schema_audit.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "file_name": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "absolute_path_recorded": False,
        },
        "workbook_metadata": {
            "core_properties_present": core_props_present,
            "core_property_values_suppressed": True,
        },
        "sheets": sheets,
        "privacy": {
            "raw_values_emitted": False,
            "absolute_source_path_emitted": False,
            "header_and_aggregate_only": True,
        },
    }
    (output / "xlsx_schema_privacy_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "xlsx_schema_privacy_audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        "XLSX_AUDIT=COMPLETE "
        f"sheets={len(sheets)} rows={sum(sheet.get('data_rows', 0) for sheet in sheets)} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
