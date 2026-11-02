from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_failure_decomposition.py")
SPEC = importlib.util.spec_from_file_location("failure_decomposition", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class FailureDecompositionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.requests = self.root / "requests.jsonl"
        self.request_rows = [
            {
                "id": "a",
                "variant": "bad_history",
                "request": {"task_type": "student_analysis"},
            },
            {
                "id": "b",
                "variant": "clean",
                "request": {"task_type": "student_analysis"},
            },
            {
                "id": "c",
                "variant": "source_outage",
                "request": {"task_type": "patient_education"},
            },
        ]
        write_jsonl(self.requests, self.request_rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _rollout(self, name: str, states: list[tuple[bool, list[str], list[str]]]) -> Path:
        path = self.root / f"{name}.jsonl"
        rows = []
        for request, (passed, failures, components) in zip(self.request_rows, states):
            rows.append(
                {
                    "id": request["id"],
                    "variant": request["variant"],
                    "request": request["request"],
                    "passed": passed,
                    "failure_types": failures,
                    "selected_components": components,
                }
            )
        write_jsonl(path, rows)
        return path

    def test_reference_conditioned_rescue_and_regression(self) -> None:
        clean = self._rollout(
            "clean_sft",
            [
                (False, ["copied_untrusted_history"], ["summary"]),
                (True, [], []),
                (False, ["failed_evidence_boundary_recovery"], ["warnings"]),
            ],
        )
        gkd = self._rollout(
            "gkd_full",
            [(True, [], []), (True, [], []), (False, ["x"], ["warnings"])],
        )
        care = self._rollout(
            "care_opd",
            [(True, [], []), (False, ["x"], ["summary"]), (True, [], [])],
        )
        requests = mod._index_requests(self.requests)
        archives = {
            "clean_sft": mod._index_rollouts(clean, requests),
            "gkd_full": mod._index_rollouts(gkd, requests),
            "care_opd": mod._index_rollouts(care, requests),
        }
        rows = mod.build_analysis(requests, archives, reference_label="clean_sft")
        care_overall = next(
            row
            for row in rows
            if row["dimension"] == "overall" and row["model"] == "care_opd"
        )
        self.assertEqual(care_overall["rescued"], 2)
        self.assertEqual(care_overall["regressed"], 1)
        self.assertEqual(care_overall["rescue_rate"], 1.0)
        self.assertEqual(care_overall["regression_rate"], 1.0)
        copied = next(
            row
            for row in rows
            if row["dimension"] == "failure_type"
            and row["stratum"] == "copied_untrusted_history"
            and row["model"] == "care_opd"
        )
        self.assertEqual(copied["n"], 1)
        self.assertEqual(copied["rescued"], 1)
        perturbed = next(
            row
            for row in rows
            if row["dimension"] == "track"
            and row["stratum"] == "perturbed"
            and row["model"] == "care_opd"
        )
        self.assertEqual(perturbed["n"], 2)
        self.assertEqual(perturbed["rescued"], 2)

    def test_rollout_may_be_a_request_subset(self) -> None:
        path = self.root / "short.jsonl"
        write_jsonl(
            path,
            [
                {
                    "id": "a",
                    "variant": "bad_history",
                    "request": {"task_type": "student_analysis"},
                    "passed": True,
                    "failure_types": [],
                    "selected_components": [],
                }
            ],
        )
        indexed = mod._index_rollouts(path, mod._index_requests(self.requests))
        self.assertEqual(set(indexed), {"a"})

    def test_model_rollout_id_sets_must_match_each_other(self) -> None:
        full_states = [
            (True, [], []),
            (True, [], []),
            (False, ["x"], ["summary"]),
        ]
        clean = self._rollout("clean_sft", full_states)
        gkd = self._rollout("gkd_full", full_states)
        care = self.root / "care_opd.jsonl"
        write_jsonl(
            care,
            [
                {
                    "id": request["id"],
                    "variant": request["variant"],
                    "request": request["request"],
                    "passed": True,
                    "failure_types": [],
                    "selected_components": [],
                }
                for request in self.request_rows[:2]
            ],
        )
        args = mod._parser().parse_args(
            [
                "--requests",
                str(self.requests),
                "--rollouts",
                f"clean_sft={clean}",
                "--rollouts",
                f"gkd_full={gkd}",
                "--rollouts",
                f"care_opd={care}",
                "--output-dir",
                str(self.root / "bad_output"),
            ]
        )
        with self.assertRaisesRegex(mod.InputError, "rollout ID set mismatch"):
            mod.run(args)

    def test_outputs_are_created_without_placeholders(self) -> None:
        states = [
            (False, ["copied_untrusted_history"], ["summary"]),
            (True, [], []),
            (False, ["failed_evidence_boundary_recovery"], ["warnings"]),
        ]
        labels = {
            label: self._rollout(label, states)
            for label in ("clean_sft", "gkd_full", "care_opd")
        }
        output = self.root / "output"
        argv = ["--requests", str(self.requests)]
        for label, path in labels.items():
            argv.extend(["--rollouts", f"{label}={path}"])
        argv.extend(["--output-dir", str(output)])
        args = mod._parser().parse_args(argv)
        payload = mod.run(args)
        self.assertEqual(payload["schema_version"], mod.SCHEMA_VERSION)
        for name in mod.OUTPUT_NAMES:
            self.assertTrue((output / name).is_file())
        self.assertNotIn("TBD", (output / "failure_decomposition.md").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
