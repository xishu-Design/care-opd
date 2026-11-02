from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_DIR = HERE.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from build_direct_clinical_gold import (  # noqa: E402
    CLINICAL_GOLD_NAME,
    EXACT_SPAN_NAME,
    MANIFEST_NAME,
    REVIEW_NAME,
    DirectClinicalGoldError,
    materialize_direct_clinical_gold,
)
from build_locked_benchmark import build_benchmark  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fixture() -> tuple[dict, dict]:
    source = "甲状腺结节10mm，边缘清晰；复核清晰，代号X。"
    fields = [
        ("size.raw", "10mm"),
        ("margin", "清晰"),
        ("code", "X"),
        ("note", " 保留空格 "),
        ("unknown", None),
    ]
    case_id = "THY-US-0123456789abcdef01234567"
    report_hmac = "rg_" + "1" * 64
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    queue = {
        "case_id": case_id,
        "report_group_hmac": report_hmac,
        "modality": "thyroid_ultrasound",
        "source_text_private": source,
        "source_record_sha256": source_hash,
        "candidate_fields": [
            {"entity_id": "N01", "field_path": path, "raw_value": value}
            for path, value in fields
        ],
    }
    canonical = {
        "case_id": case_id,
        "report_group_hmac": report_hmac,
        "source_id": "SRC_PRIVATE_001",
        "source_record_hmac": "src_" + "2" * 64,
        "modality": "thyroid_ultrasound",
        "source": {
            "source_text_private": source,
            "deidentification_status": "reviewed_private_use",
            "model_input_allowed": True,
        },
        "entities": [
            {
                "entity_id": "N01",
                "fields": [
                    {
                        "field_path": path,
                        "candidate_value_private": value,
                        "candidate_status": (
                            "reported_candidate"
                            if value is not None
                            else "unresolved_missingness"
                        ),
                    }
                    for path, value in fields
                ],
            }
        ],
    }
    return queue, canonical


class DirectClinicalGoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.queue_path = self.root / "queue.private.jsonl"
        self.canonical_path = self.root / "canonical.private.jsonl"
        self.annotation_dir = self.root / "direct"
        self.benchmark_dir = self.root / "benchmark"
        queue, canonical = _fixture()
        _write_jsonl(self.queue_path, [queue])
        _write_jsonl(self.canonical_path, [canonical])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_all_nonempty_gold_is_retained_and_span_is_auxiliary(self) -> None:
        summary = materialize_direct_clinical_gold(
            self.queue_path,
            self.annotation_dir,
            acknowledge_private_clinical_data=True,
        )
        self.assertEqual(summary["counts"]["clinical_gold_fields"], 4)
        self.assertEqual(summary["counts"]["span_localized_clinical_gold_fields"], 1)
        self.assertEqual(summary["counts"]["empty_unknown_slots"], 1)

        gold = _read_jsonl(self.annotation_dir / CLINICAL_GOLD_NAME)
        exact = _read_jsonl(self.annotation_dir / EXACT_SPAN_NAME)
        review = _read_jsonl(self.annotation_dir / REVIEW_NAME)
        self.assertEqual(len(gold), 4)
        self.assertEqual(len(exact), 1)
        self.assertEqual(len(review), 4)
        by_path = {row["field_path"]: row for row in gold}
        self.assertEqual(by_path["size.raw"]["alignment_status"], "exact_unique_verbatim")
        self.assertEqual(by_path["margin"]["alignment_status"], "duplicate_verbatim")
        self.assertEqual(by_path["code"]["alignment_status"], "ambiguous_or_short")
        self.assertEqual(by_path["note"]["candidate_value_private"], " 保留空格 ")
        self.assertEqual(by_path["note"]["alignment_status"], "not_verbatim")
        self.assertFalse(by_path["note"]["evidence_spans"])

        manifest = json.loads(
            (self.annotation_dir / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["protocol"]["model_calls"], 0)
        self.assertEqual(manifest["outputs"][CLINICAL_GOLD_NAME]["rows"], 4)

        benchmark = build_benchmark(
            self.canonical_path,
            self.annotation_dir / CLINICAL_GOLD_NAME,
            self.benchmark_dir,
            task_mode="structured_evidence_faithfulness",
            acknowledge_private_clinical_data=True,
        )
        self.assertEqual(benchmark["counts"]["primary_report_families"], 1)
        self.assertEqual(benchmark["counts"]["primary_qzbc_seed_rows"], 4)
        self.assertEqual(benchmark["counts"]["clinical_gold_fields"], 4)
        self.assertEqual(benchmark["counts"]["exact_verified_fields"], 1)
        self.assertEqual(
            benchmark["gold_completeness"][
                "clinical_gold_coverage_of_populated_candidates"
            ],
            1.0,
        )
        private_gold = _read_jsonl(self.benchmark_dir / "field_gold.private.jsonl")
        clean = next(row for row in private_gold if row["variant"] == "clean")
        self.assertEqual(clean["reference_scope"], "all_nonempty_structured_clinical_gold")
        self.assertEqual(len(clean["gold_fields"]), 4)
        self.assertIn(" 保留空格 ", [field["value"] for field in clean["gold_fields"]])

    def test_source_hash_mismatch_is_rejected(self) -> None:
        queue, _ = _fixture()
        queue["source_record_sha256"] = "0" * 64
        _write_jsonl(self.queue_path, [queue])
        with self.assertRaises(DirectClinicalGoldError):
            materialize_direct_clinical_gold(
                self.queue_path,
                self.annotation_dir,
                acknowledge_private_clinical_data=True,
            )


if __name__ == "__main__":
    unittest.main()
