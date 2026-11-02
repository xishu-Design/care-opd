from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_round_dynamics.py")
SPEC = importlib.util.spec_from_file_location("round_dynamics", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class RoundDynamicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _round(self, name: str, failures: dict[str, list[str]]) -> Path:
        root = self.root / name
        rollout_root = root / "rollouts"
        train_root = root / "opd_train"
        ids = ("a", "b", "c", "d")
        rollout_rows = [
            {
                "id": item_id,
                "passed": item_id not in failures,
                "failure_types": failures.get(item_id, []),
            }
            for item_id in ids
        ]
        write_jsonl(rollout_root / "qzbc_rollouts.jsonl", rollout_rows)
        failed = [item_id for item_id in ids if item_id in failures]
        clean = [item_id for item_id in ids if item_id not in failures]
        opd_rows = [
            {"id": item_id, "clean_control": False} for item_id in failed
        ] + [{"id": item_id, "clean_control": True} for item_id in clean]
        write_jsonl(rollout_root / "qzbc_opd.jsonl", opd_rows)
        format_count = sum(
            "invalid_generated_response" in failures[item_id] for item_id in failed
        )
        summary = {
            "rollouts": len(ids),
            "passed": len(clean),
            "opd_examples": len(ids),
            "opd_failures": len(failed),
            "clean_controls": len(clean),
            "format_failures_total": format_count,
            "format_failures_selected": format_count,
            "semantic_failures": len(failed) - format_count,
            "dpo_pairs": len(failed),
        }
        (rollout_root / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        write_jsonl(
            train_root / "metrics.jsonl",
            [
                {"event": "start", "method": "care_opd", "dataset_size": len(ids)},
                {
                    "event": "train",
                    "step": 1,
                    "loss": 0.4,
                    "candidate_tokens_local": 10,
                    "trusted_tokens_local": 8,
                },
                {"event": "complete", "global_step": 1},
            ],
        )
        return root

    def test_set_turnover_and_jaccard(self) -> None:
        round1 = self._round(
            "round_01",
            {"a": ["invalid_generated_response"], "b": ["semantic_x"]},
        )
        round2 = self._round("round_02", {"b": ["semantic_x"], "c": ["semantic_y"]})
        round3 = self._round("round_03", {"c": ["semantic_y"]})
        payload = mod.build_analysis(
            [("round_01", round1), ("round_02", round2), ("round_03", round3)]
        )
        first_all = next(
            row
            for row in payload["transitions"]
            if row["from"] == "round_01" and row["failure_class"] == "all"
        )
        self.assertEqual(first_all["persistent"], 1)
        self.assertEqual(first_all["disappeared"], 1)
        self.assertEqual(first_all["new"], 1)
        self.assertAlmostEqual(first_all["jaccard"], 1 / 3)
        first_format = next(
            row
            for row in payload["transitions"]
            if row["from"] == "round_01" and row["failure_class"] == "format"
        )
        self.assertEqual(first_format["disappeared"], 1)
        self.assertEqual(first_format["current"], 0)

    def test_summary_mismatch_is_rejected(self) -> None:
        round1 = self._round("round_01", {"a": ["semantic_x"]})
        summary_path = round1 / "rollouts" / "summary.json"
        summary = json.loads(summary_path.read_text("utf-8"))
        summary["passed"] = 99
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(mod.InputError, "declared 99"):
            mod._round_record("round_01", round1)

    def test_rollout_directory_input_finds_sibling_metrics(self) -> None:
        round1 = self._round("round_01", {"a": ["semantic_x"]})
        files = mod._round_files(round1 / "rollouts")
        self.assertEqual(files["metrics"].resolve(), (round1 / "opd_train" / "metrics.jsonl").resolve())

    def test_outputs_are_atomic_artifacts(self) -> None:
        rounds = [
            self._round("round_01", {"a": ["invalid_generated_response"]}),
            self._round("round_02", {"b": ["semantic_x"]}),
            self._round("round_03", {}),
        ]
        output = self.root / "output"
        argv: list[str] = []
        for index, path in enumerate(rounds, 1):
            argv.extend(["--round", f"round_{index:02d}={path}"])
        argv.extend(["--output-dir", str(output)])
        args = mod._parser().parse_args(argv)
        payload = mod.run(args)
        self.assertEqual(payload["schema_version"], mod.SCHEMA_VERSION)
        for name in mod.OUTPUT_NAMES:
            self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
