from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("efficiency_audit.py")
SPEC = importlib.util.spec_from_file_location("efficiency_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class EfficiencyAuditTest(unittest.TestCase):
    def test_result_blind_selection_is_balanced_and_order_invariant(self) -> None:
        rows = []
        for variant in audit.EXPECTED_VARIANTS:
            for index in range(10):
                rows.append(
                    {
                        "id": f"{variant}-{index}",
                        "variant": variant,
                        "student_messages": [],
                        "irrelevant_model_result": index % 2,
                    }
                )
        ids = [row["id"] for row in rows]
        first = audit.select_result_blind_sample(rows, ids, sample_size=20, seed=7)
        reversed_rows = list(reversed(rows))
        for row in reversed_rows:
            row["irrelevant_model_result"] = 1 - row["irrelevant_model_result"]
        second = audit.select_result_blind_sample(
            reversed_rows, list(reversed(ids)), sample_size=20, seed=7
        )
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        counts = {}
        for row in first:
            counts[row["variant"]] = counts.get(row["variant"], 0) + 1
        self.assertEqual(counts, {variant: 5 for variant in audit.EXPECTED_VARIANTS})

    def test_percentile_and_summary(self) -> None:
        rows = [
            {
                "id": str(index),
                "variant": "clean",
                "generation_seconds": float(index),
                "end_to_end_seconds": float(index) + 0.5,
                "output_tokens": index * 10,
                "prompt_tokens": 100,
            }
            for index in range(1, 5)
        ]
        summary = audit.summarize_observations(rows)
        self.assertEqual(summary["requests"], 4)
        self.assertAlmostEqual(summary["generation_latency_seconds"]["p50"], 2.5)
        self.assertAlmostEqual(summary["generation_latency_seconds"]["p95"], 3.85)
        self.assertAlmostEqual(summary["generated_tokens_per_second"], 10.0)


if __name__ == "__main__":
    unittest.main()
