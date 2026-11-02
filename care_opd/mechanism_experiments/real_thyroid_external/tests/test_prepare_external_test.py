from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from mechanism_experiments.real_thyroid_external.prepare_external_test import (
    PATHOLOGY_FIELDS,
    PATHOLOGY_SHEET,
    PATHOLOGY_SOURCE,
    QUALITY_GRADE,
    QUALITY_NOTE,
    ULTRASOUND_FIELDS,
    ULTRASOUND_SHEET,
    ULTRASOUND_SOURCE,
    build_private_records,
    load_report_groups,
    prepare_external_test,
)


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _sheet_xml(rows: list[list[str]]) -> str:
    xml_rows = []
    for row_number, values in enumerate(rows, start=1):
        cells = []
        for column, value in enumerate(values, start=1):
            if value == "":
                continue
            reference = f"{_column_name(column)}{row_number}"
            cells.append(
                f'<c r="{reference}" t="inlineStr"><is><t>{_xml_escape(value)}</t></is></c>'
            )
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def _write_synthetic_workbook(path: Path) -> None:
    ultrasound_headers = [
        ULTRASOUND_SOURCE,
        "结节编号",
        *(header for header, _ in ULTRASOUND_FIELDS),
    ]
    pathology_headers = [
        PATHOLOGY_SOURCE,
        *(header for header, _ in PATHOLOGY_FIELDS),
        QUALITY_NOTE,
        QUALITY_GRADE,
    ]
    us_width = len(ultrasound_headers)
    path_width = len(pathology_headers)
    ultrasound_rows = [
        ultrasound_headers,
        ["PRIVATE_US_ALPHA", "1", "左叶", *([""] * (us_width - 3))],
        ["", "2", "右叶", *([""] * (us_width - 3))],
        ["PRIVATE_US_BETA", *([""] * (us_width - 1))],
        ["PRIVATE_US_GAMMA", "1", "峡部", *([""] * (us_width - 3))],
    ]
    pathology_rows = [
        pathology_headers,
        ["PRIVATE_PATH_ALPHA", "甲状腺", *([""] * (path_width - 4)), "复核", "A"],
        ["PRIVATE_PATH_BETA", "", *([""] * (path_width - 4)), "", "B"],
    ]
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        f'<sheet name="{ULTRASOUND_SHEET}" sheetId="1" r:id="rId1"/>'
        f'<sheet name="{PATHOLOGY_SHEET}" sheetId="2" r:id="rId2"/>'
        '</sheets></workbook>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", _sheet_xml(ultrasound_rows))
        archive.writestr("xl/worksheets/sheet2.xml", _sheet_xml(pathology_rows))


class RealThyroidExternalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workbook = self.root / "synthetic.xlsx"
        _write_synthetic_workbook(self.workbook)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_report_level_grouping_does_not_invent_linkage(self) -> None:
        groups = load_report_groups(self.workbook)
        ultrasound = [group for group in groups if group.modality == "thyroid_ultrasound"]
        pathology = [group for group in groups if group.modality == "thyroid_pathology"]

        self.assertEqual(len(ultrasound), 3)
        self.assertEqual(len(pathology), 2)
        self.assertEqual(len(ultrasound[0].entities), 2)
        self.assertEqual(ultrasound[0].entities[0].entity_id, "N01")
        self.assertEqual(ultrasound[0].entities[1].entity_id, "N02")
        self.assertEqual(len(ultrasound[1].entities), 0)

        records = build_private_records(
            groups,
            workbook_sha256="a" * 64,
            secret=b"synthetic-secret-material-32-bytes-minimum",
        )
        self.assertEqual(len(records), 5)
        for record in records:
            self.assertIsNone(record["linkage"]["patient_group_hmac"])
            self.assertIsNone(record["linkage"]["encounter_group_hmac"])
            self.assertIsNone(record["linkage"]["lesion_group_hmac"])
            self.assertEqual(record["linkage"]["status"], "unavailable_do_not_infer")

    def test_audit_default_never_emits_raw_values(self) -> None:
        output = self.root / "audit"
        summary = prepare_external_test(
            self.workbook,
            output,
            expected_ultrasound_reports=3,
            expected_pathology_reports=2,
        )
        serialized = (output / "audit_summary.json").read_text(encoding="utf-8")
        self.assertEqual(summary["mode"], "audit_only")
        self.assertEqual(summary["counts"]["report_groups_total"], 5)
        self.assertEqual(summary["counts"]["ultrasound_candidate_entities"], 3)
        for sentinel in ("PRIVATE_US_ALPHA", "PRIVATE_PATH_ALPHA", "左叶", "甲状腺"):
            self.assertNotIn(sentinel, serialized)
        self.assertFalse((output / "canonical_records.private.jsonl").exists())
        self.assertFalse((output / "agent_work_queue.private.jsonl").exists())

    def test_private_export_requires_acknowledgement_and_env_secret(self) -> None:
        with self.assertRaises(PermissionError):
            prepare_external_test(
                self.workbook,
                self.root / "private",
                private_export=True,
                expected_ultrasound_reports=3,
                expected_pathology_reports=2,
            )

        with mock.patch.dict(os.environ, {"SYNTHETIC_HMAC_KEY": "x" * 40}, clear=False):
            summary = prepare_external_test(
                self.workbook,
                self.root / "private",
                private_export=True,
                acknowledge_private_clinical_data=True,
                hmac_key_env="SYNTHETIC_HMAC_KEY",
                expected_ultrasound_reports=3,
                expected_pathology_reports=2,
            )
        self.assertEqual(summary["mode"], "private_export")
        records = [
            json.loads(line)
            for line in (self.root / "private" / "canonical_records.private.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        queue = [
            json.loads(line)
            for line in (self.root / "private" / "agent_work_queue.private.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(records), 5)
        self.assertEqual(len(queue), 5)
        self.assertTrue(all(item["case_id"].startswith("THY-") for item in records))
        self.assertTrue(
            all(
                item["agent_boundary"]["clinical_gold_requires_human_adjudication"]
                for item in queue
            )
        )
        self.assertTrue(queue[0]["candidate_fields"])
        self.assertEqual(
            set(queue[0]["candidate_fields"][0])
            & {"entity_id", "field_path", "raw_value"},
            {"entity_id", "field_path", "raw_value"},
        )
        self.assertEqual(len(queue[0]["source_record_sha256"]), 64)
        self.assertEqual(queue[0]["agent_mode"], "candidate_auditor")
        self.assertIn("PRIVATE_US_ALPHA", queue[0]["source_text_private"])
        self.assertEqual((self.root / "private" / ".gitignore").read_text(), "*\n!.gitignore\n")

    def test_hmac_ids_are_deterministic_and_keyed(self) -> None:
        groups = load_report_groups(self.workbook)
        first = build_private_records(
            groups, workbook_sha256="b" * 64, secret=b"a" * 32
        )
        repeated = build_private_records(
            groups, workbook_sha256="b" * 64, secret=b"a" * 32
        )
        different_key = build_private_records(
            groups, workbook_sha256="b" * 64, secret=b"c" * 32
        )
        self.assertEqual(
            [record["case_id"] for record in first],
            [record["case_id"] for record in repeated],
        )
        self.assertNotEqual(first[0]["case_id"], different_key[0]["case_id"])


if __name__ == "__main__":
    unittest.main()
