"""Small deterministic tests for the offline margin-to-behavior analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from . import analyze_margin_behavior as analysis
except ImportError:  # Direct invocation from this directory.
    import analyze_margin_behavior as analysis


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _margin_row(item_id: str, label: str, full: float) -> dict:
    return {
        "id": item_id,
        "model_label": label,
        "full": {"margin": full},
        "selected": {
            "localized_complete": True,
            "localized_margin": full + 0.2,
        },
        "outside": {"margin": full - 0.2},
    }


class MarginBehaviorTests(unittest.TestCase):
    def test_rank_metrics_handle_ties(self) -> None:
        self.assertEqual(analysis._auroc([0.0, 0.0, 1.0, 1.0], [False, True, False, True]), 0.5)
        self.assertAlmostEqual(
            analysis._point_biserial([0.0, 1.0, 2.0, 3.0], [False, False, True, True]),
            2.0 / (5.0 ** 0.5),
        )

    def test_end_to_end_join_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            margins, clean, gkd, care = (
                root / "margin.jsonl",
                root / "clean.jsonl",
                root / "gkd.jsonl",
                root / "care.jsonl",
            )
            ids = [f"item-{index}" for index in range(8)]
            rows: list[dict] = []
            for index, item_id in enumerate(ids):
                rows.extend(
                    [
                        _margin_row(item_id, "clean_sft", index - 8.0),
                        _margin_row(item_id, "gkd_full", index - 5.0),
                        _margin_row(item_id, "care_opd", index - 3.0),
                    ]
                )
            _write_jsonl(margins, rows)
            _write_jsonl(clean, [{"id": item_id, "passed": False} for item_id in ids])
            _write_jsonl(
                gkd,
                [{"id": item_id, "passed": index in {3, 5, 7}} for index, item_id in enumerate(ids)],
            )
            _write_jsonl(
                care,
                [{"id": item_id, "passed": index >= 4} for index, item_id in enumerate(ids)],
            )

            output = root / "out"
            payload = analysis.analyze(
                margin_path=margins,
                clean_rollouts_path=clean,
                gkd_rollouts_path=gkd,
                care_rollouts_path=care,
                output_dir=output,
                bootstrap_replicates=200,
                bootstrap_seed=17,
            )

            self.assertEqual(payload["status"], "COMPLETE")
            self.assertEqual(payload["behavior"]["n"], 8)
            self.assertEqual(payload["behavior"]["care_rescued"], 4)
            self.assertEqual(payload["behavior"]["gkd_rescued"], 3)
            self.assertAlmostEqual(payload["behavior"]["care_minus_gkd_rescue_rate"], 0.125)
            self.assertEqual(len(payload["quartile_recovery"]["care_full"]), 4)
            full_prediction = next(
                row for row in payload["prediction"] if row["score"] == "care_opd_full"
            )
            self.assertEqual(full_prediction["auroc"], 1.0)
            for filename in analysis.OUTPUT_FILES:
                self.assertTrue((output / filename).is_file(), filename)

    def test_rejects_nonfailure_clean_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            margins = root / "margin.jsonl"
            rows = [_margin_row("item", label, 0.1) for label in analysis.MODEL_LABELS]
            _write_jsonl(margins, rows)
            clean, gkd, care = root / "clean.jsonl", root / "gkd.jsonl", root / "care.jsonl"
            _write_jsonl(clean, [{"id": "item", "passed": True}])
            _write_jsonl(gkd, [{"id": "item", "passed": True}])
            _write_jsonl(care, [{"id": "item", "passed": True}])
            with self.assertRaisesRegex(analysis.InputError, "not Clean-SFT-failure-conditioned"):
                analysis.analyze(
                    margin_path=margins,
                    clean_rollouts_path=clean,
                    gkd_rollouts_path=gkd,
                    care_rollouts_path=care,
                    output_dir=root / "out",
                    bootstrap_replicates=100,
                )


if __name__ == "__main__":
    unittest.main()
