from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from care_opd.evaluate import build_report


def _paired_rows(
    method: str,
    split: str,
    *,
    critical_success: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(2):
        group = f"{split}-{index}::eval_seed=42"
        rows.extend(
            [
                {
                    "method": method,
                    "split": split,
                    "world_group_id": group,
                    "condition": "critical",
                    "resolved": critical_success,
                    "unsafe_green": False,
                    "risk_weight": 2.0,
                    "total_steps": 2,
                },
                {
                    "method": method,
                    "split": split,
                    "world_group_id": group,
                    "condition": "nuisance",
                    "resolved": True,
                    "unsafe_green": False,
                    "risk_weight": 2.0,
                    "total_steps": 2,
                },
            ]
        )
    return rows


class AutoBaselineSelectionTest(unittest.TestCase):
    def test_auto_selects_on_dev_and_only_tests_on_test(self) -> None:
        rows: list[dict[str, object]] = []
        for split in ("dev", "test"):
            rows.extend(
                _paired_rows(
                    "baseline_a",
                    split,
                    critical_success=True,
                )
            )
            rows.extend(
                _paired_rows(
                    "baseline_b",
                    split,
                    critical_success=False,
                )
            )
            rows.extend(
                _paired_rows(
                    "care_opd",
                    split,
                    critical_success=True,
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = build_report(
                rows,
                output_dir=output,
                baseline="auto",
                bootstrap_samples=20,
                confidence=0.95,
                seed=7,
            )
            selection = report["baseline_selection"]
            self.assertEqual(selection["method"], "baseline_a")
            self.assertEqual(selection["selection_split"], "dev")
            self.assertFalse(selection["exploratory"])
            self.assertIn(
                "care_opd_vs_baseline_a/test",
                report["paired_tests"],
            )
            self.assertFalse(
                any(key.endswith("/dev") for key in report["paired_tests"])
            )
            persisted = json.loads(
                (output / "statistical_tests.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                persisted["baseline_selection"]["method"],
                "baseline_a",
            )

    def test_auto_falls_back_to_test_and_marks_exploratory(self) -> None:
        rows = [
            *_paired_rows(
                "baseline_a",
                "test",
                critical_success=False,
            ),
            *_paired_rows(
                "baseline_b",
                "test",
                critical_success=True,
            ),
            *_paired_rows(
                "care_opd",
                "test",
                critical_success=True,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = build_report(
                rows,
                output_dir=Path(directory),
                baseline="auto",
                bootstrap_samples=20,
                confidence=0.95,
                seed=7,
            )
        selection = report["baseline_selection"]
        self.assertEqual(selection["method"], "baseline_b")
        self.assertEqual(selection["selection_split"], "test")
        self.assertTrue(selection["exploratory"])


if __name__ == "__main__":
    unittest.main()
