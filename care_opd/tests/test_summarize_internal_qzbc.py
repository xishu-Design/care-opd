from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "public_eval" / "summarize_internal_qzbc.py"
SPEC = importlib.util.spec_from_file_location("summarize_internal_qzbc", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class InternalQZBCSummaryTest(unittest.TestCase):
    def test_pair_metrics_and_alignment(self) -> None:
        requests = []
        rollouts = []
        outcomes = {
            "s1": {"clean": True, "bad_history": True, "source_outage": False},
            "s2": {"clean": False, "bad_history": True, "source_outage": False},
        }
        for seed_id, variants in outcomes.items():
            for variant, passed in variants.items():
                item_id = f"{seed_id}_{variant}"
                requests.append({"id": item_id, "seed_id": seed_id, "variant": variant})
                rollouts.append(
                    {
                        "id": item_id,
                        "variant": variant,
                        "passed": passed,
                        "failure_types": [] if passed else ["synthetic_failure"],
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.jsonl"
            rollout_path = root / "model" / "qzbc_rollouts.jsonl"
            write_jsonl(request_path, requests)
            write_jsonl(rollout_path, rollouts)
            request_map = MODULE._unique_by_id(MODULE._read_jsonl(request_path), request_path)
            pairs, payload = MODULE._model_pairs(
                request_map, rollout_path, clean_variant="clean"
            )
        metric = payload["unified_metrics"]
        self.assertEqual(len(pairs), 4)
        self.assertEqual(metric["clean_task_success_rate"], 0.5)
        self.assertEqual(metric["perturbed_task_success_rate"], 0.5)
        self.assertEqual(metric["conditional_retention"], 0.5)
        self.assertEqual(metric["regression_rate"], 0.5)
        self.assertEqual(metric["clean_correct_perturbed_wrong"], 1)
        self.assertEqual(metric["clean_wrong_perturbed_correct"], 1)
        self.assertIsNone(metric["prediction_consistency"])

    def test_unbalanced_variants_fail_closed(self) -> None:
        requests = [
            {"id": "a", "seed_id": "s1", "variant": "clean"},
            {"id": "b", "seed_id": "s1", "variant": "bad_history"},
            {"id": "c", "seed_id": "s2", "variant": "clean"},
            {"id": "d", "seed_id": "s2", "variant": "bad_history"},
            {"id": "e", "seed_id": "s2", "variant": "source_outage"},
        ]
        rollouts = [
            {"id": row["id"], "variant": row["variant"], "passed": True}
            for row in requests
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.jsonl"
            rollout_path = root / "rollouts.jsonl"
            write_jsonl(request_path, requests)
            write_jsonl(rollout_path, rollouts)
            request_map = MODULE._unique_by_id(MODULE._read_jsonl(request_path), request_path)
            with self.assertRaisesRegex(ValueError, "unbalanced"):
                MODULE._model_pairs(request_map, rollout_path, clean_variant="clean")

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        rows = [
            {
                "seed_id": f"s{index // 2}",
                "clean_success": index % 3 != 0,
                "perturbed_success": index % 4 != 0,
            }
            for index in range(12)
        ]
        first = MODULE._cluster_bootstrap(rows, seed=7, replicates=100)
        second = MODULE._cluster_bootstrap(rows, seed=7, replicates=100)
        self.assertEqual(first, second)

    def test_legacy_summary_records_but_does_not_claim_suite_or_split_binding(self) -> None:
        requests = [
            {"id": "s1_clean", "seed_id": "s1", "variant": "clean"},
            {"id": "s1_fault", "seed_id": "s1", "variant": "source_outage"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.jsonl"
            eval_root = root / "eval"
            output = root / "output"
            write_jsonl(request_path, requests)
            suite_models = []
            for index, label in enumerate(MODULE.MODEL_ORDER, start=1):
                write_jsonl(
                    eval_root / label / "qzbc_rollouts.jsonl",
                    [
                        {
                            "id": row["id"],
                            "variant": row["variant"],
                            "passed": True,
                            "failure_types": [],
                        }
                        for row in requests
                    ],
                )
                suite_models.append(
                    {"label": label, "model_fingerprint": f"{index:064x}"}
                )
            suite = {"models": suite_models}
            suite["suite_fingerprint"] = MODULE._canonical_hash(
                [
                    (item["label"], item["model_fingerprint"])
                    for item in suite_models
                ]
            )
            suite_path = root / "model_suite.json"
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                MODULE.main(
                    [
                        "--requests",
                        str(request_path),
                        "--eval-root",
                        str(eval_root),
                        "--model-suite",
                        str(suite_path),
                        "--output-dir",
                        str(output),
                    ]
                )
            summary = json.loads(
                (output / "suite_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["checkpoint_identity_binding"]["status"],
                "RETROSPECTIVE_UNVERIFIED",
            )
            self.assertEqual(
                summary["split_integrity_binding"]["status"],
                "RETROSPECTIVE_UNVERIFIED",
            )
            self.assertEqual(
                summary["checkpoint_identity_binding"]["declared_expected_suite"][
                    "suite_fingerprint"
                ],
                suite["suite_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
