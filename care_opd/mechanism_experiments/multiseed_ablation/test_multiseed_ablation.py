from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for path in (HERE, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common
import launch_queue
import plan_evaluation
import plan_multiseed
import summarize_multiseed
from mechanism_experiments.token_matched_control import contract as token_contract


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_training_suite(root: Path, seeds: str = "42,43,44") -> tuple[dict[str, object], Path]:
    views = root / "views"
    rows = [
        {
            "id": rid,
            "rollout": '{"summary":"bad"}',
            "teacher_messages": [{"role": "user", "content": "repair"}],
            "repair_spans": [{"start": 12, "end": 15}],
            "clean_control": False,
            "passed": False,
        }
        for rid in ("x", "y")
    ]
    write_jsonl(views / "full.jsonl", rows)
    write_jsonl(views / "outside.jsonl", rows)
    matrix = root / "matrix.json"
    write_json(
        matrix,
        {
            "schema_version": "care_opd.causal_ablation_matrix.v1",
            "variants": [
                {
                    "name": "full_selected",
                    "data_view": "full.jsonl",
                    "overrides": {"baseline_mode": "care_opd"},
                },
                {
                    "name": "outside_only",
                    "data_view": "outside.jsonl",
                    "overrides": {"baseline_mode": "care_opd", "retain_outside_repair": False},
                },
            ],
        },
    )
    config = root / "config.json"
    write_json(config, {})
    output = root / "run"
    args = argparse.Namespace(
        matrix=matrix,
        views_dir=views,
        train_config=config,
        student_model="student",
        teacher_model="teacher",
        reference_model="student",
        output_root=output,
        max_steps=3,
        save_steps=2,
        seeds=seeds,
        python="python",
        resume=True,
    )
    suite = plan_multiseed.build_multiseed_plan(args)
    return suite, output / "multiseed_plan.json"


def materialize_completed_training(suite: dict[str, object], *, steps: int = 3) -> None:
    for seed_entry in suite["seeds"]:
        seed = seed_entry["training_seed"]
        nested = common.load_json_object(Path(seed_entry["plan"]["path"]))
        for arm in nested["arms"]:
            output = Path(arm["output_dir"])
            final = output / "final"
            final.mkdir(parents=True, exist_ok=True)
            write_json(
                final / "run_metadata.json",
                {"global_step": steps, "training": {"seed": seed, "max_steps": steps}},
            )
            (final / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (final / "adapter_model.safetensors").write_bytes(b"adapter")
            write_jsonl(output / "metrics.jsonl", [{"event": "complete", "global_step": steps}])


class MultiSeedPlanTests(unittest.TestCase):
    def test_default_design_changes_only_training_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, plan_path = make_training_suite(Path(temp))
            self.assertEqual(suite["design"]["training_seeds"], [42, 43, 44])
            self.assertEqual(len(suite["seeds"]), 3)
            self.assertTrue(plan_path.is_file())
            for entry in suite["seeds"]:
                nested = common.load_json_object(Path(entry["plan"]["path"]))
                self.assertEqual(nested["budget_contract"]["seed"], entry["training_seed"])
                for arm in nested["arms"]:
                    command = arm["command"]
                    self.assertEqual(command[command.index("--seed") + 1], str(entry["training_seed"]))
                    self.assertIn("--nproc_per_node=1", command)

    def test_duplicate_training_seed_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "unique"):
                make_training_suite(Path(temp), seeds="42,42,44")

    def test_token_matrix_uses_token_contract_and_binds_exact_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            views = root / "views"
            data = views / "full_selected.jsonl"
            write_jsonl(
                data,
                [{
                    "id": "row-1",
                    "rollout": '{"summary":"bad"}',
                    "teacher_messages": [{"role": "user", "content": "repair"}],
                    "repair_spans": [{"start": 12, "end": 15}],
                    "clean_control": False,
                    "passed": False,
                }],
            )
            audit = root / "token_audit.json"
            write_json(
                audit,
                {
                    "schema_version": "care_opd.token_matched_mask_audit.v1",
                    "status": "COMPLETE",
                    "source": {
                        "path": str(data.resolve()),
                        "sha256": common.sha256(data),
                        "rows": 1,
                    },
                    "random_mask_seed": 20260815,
                    "max_length": 4096,
                    "max_response_tokens": 1200,
                    "add_eos": True,
                    "tokenizer": "student",
                    "tokenizer_sha256": "a" * 64,
                    "implementation": {
                        label: {
                            "path": str(path),
                            "sha256": token_contract._sha256(path),
                            "bytes": path.stat().st_size,
                        }
                        for label, path in token_contract.IMPLEMENTATION_PATHS.items()
                    },
                    "overall": {
                        "rows": 1,
                        "selected_tokens": 3,
                        "exact_count_match_rate": 1.0,
                        "exact_weight_multiset_match_rate": 1.0,
                        "active_semantic_tokens": 3,
                        "active_random_tokens": 3,
                        "zero_width_selected_tokens": 0,
                        "semantic_weight_sum": 3.0,
                        "random_weight_sum": 3.0,
                        "response_truncation_rows": 0,
                    },
                },
            )
            config = root / "config.json"
            write_json(config, {})
            matrix = PROJECT_ROOT / "mechanism_experiments" / "token_matched_control" / "token_matched_matrix.json"
            args = argparse.Namespace(
                matrix=matrix,
                views_dir=views,
                train_config=config,
                student_model="student",
                teacher_model="teacher",
                reference_model="student",
                output_root=root / "run",
                max_steps=3,
                save_steps=2,
                seeds="42,43",
                python="python",
                resume=True,
                contract="auto",
                prerequisite_audit=[audit],
                contrasts_file=None,
                contrast=[],
            )
            suite = plan_multiseed.build_multiseed_plan(args)
            self.assertEqual(suite["design"]["contract_mode"], "token_matched")
            for entry in suite["seeds"]:
                nested = common.load_json_object(Path(entry["plan"]["path"]))
                self.assertEqual(nested["scientific_contract"]["status"], "PASS")
                self.assertEqual(nested["token_matching_audit"]["sha256"], common.sha256(audit))
            self.assertEqual(
                suite["design"]["matched_signature"]["token_matching_audit"]["status"],
                "PASS",
            )

    def test_queue_is_plan_only_without_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, plan_path = make_training_suite(root)
            code = launch_queue.main(
                [
                    "--plan", str(plan_path), "--gpus", "4,5", "--max-parallel", "2",
                    "--log-dir", str(root / "logs"),
                ]
            )
            self.assertEqual(code, 0)
            self.assertFalse((root / "logs").exists())

    def test_completed_job_requires_matching_seed_step_and_adapter_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            final = root / "arm" / "final"
            final.mkdir(parents=True)
            write_json(
                final / "run_metadata.json",
                {
                    "global_step": 3,
                    "training": {"seed": 42, "max_steps": 3},
                },
            )
            (final / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (final / "adapter_model.safetensors").write_bytes(b"adapter")
            write_jsonl(root / "arm" / "metrics.jsonl", [{"event": "complete", "global_step": 3}])
            job = {"output_dir": str(root / "arm"), "training_seed": 42}
            self.assertTrue(launch_queue._is_complete(job, expected_steps=3))
            self.assertFalse(launch_queue._is_complete(job, expected_steps=4))
            job["training_seed"] = 43
            self.assertFalse(launch_queue._is_complete(job, expected_steps=3))
            with self.assertRaisesRegex(ValueError, "contaminated output root"):
                launch_queue._validate_existing_identity(job, expected_steps=3)


class EvaluationPlanTests(unittest.TestCase):
    def test_evaluation_plan_binds_every_training_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite, train_plan = make_training_suite(root)
            materialize_completed_training(suite)
            requests = root / "requests.jsonl"
            family = root / "family.json"
            write_jsonl(requests, [{"id": "x"}])
            write_json(family, ["x"])
            args = argparse.Namespace(
                training_plan=train_plan,
                requests=requests,
                family_clean_ids=family,
                output_root=root / "eval",
                python="python",
                num_workers=1,
                devices="cuda:0",
                dtype="bfloat16",
                max_new_tokens=16,
                generation_seed=7,
                reference="full_selected",
            )
            result = plan_evaluation.build_evaluation_suite(args)
            self.assertEqual(
                [entry["training_seed"] for entry in result["seeds"]],
                suite["design"]["training_seeds"],
            )
            self.assertTrue((root / "eval" / "commands.sh").is_file())
            for entry in result["seeds"]:
                self.assertTrue(entry["summary"]["path"].endswith("summary.json"))
                self.assertEqual(entry["commands"][-1][-2], "--output-dir")


def fake_causal_summary(
    value: float,
    *,
    requests_binding: dict[str, str],
    family_binding: dict[str, str],
    source_bindings: dict[str, dict[str, str]],
) -> dict[str, object]:
    def model(arm: str, offset: float) -> dict[str, object]:
        return {
            "unified_metrics": {
                "clean_task_success_rate": 0.9 + offset,
                "perturbed_task_success_rate": value + offset,
                "conditional_retention": 0.8 + offset,
                "regression_rate": 0.1 - offset,
            },
            "source": source_bindings[arm],
        }

    def relative(offset: float) -> dict[str, float]:
        return {
            "rescue_rate": 0.7 + offset,
            "regression_rate": 0.2 - offset,
            "net_success_delta": 0.1 + offset,
        }

    return {
        "schema_version": "care_opd.causal_ablation_summary.v1",
        "reference": "full_selected",
        "requests": requests_binding,
        "family_ids": family_binding,
        "models": {
            "full_selected": model("full_selected", 0.0),
            "outside_only": model("outside_only", -0.05),
        },
        "reference_conditioned_request_metrics": {
            "full_selected": relative(0.0),
            "outside_only": relative(-0.05),
        },
    }


class AggregateTests(unittest.TestCase):
    def test_mean_sample_std_and_direction_aware_worst_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training = root / "training.json"
            write_json(training, {"schema_version": common.TRAIN_SUITE_SCHEMA})
            requests = root / "requests.jsonl"
            family = root / "family.json"
            write_jsonl(requests, [{"id": "x"}])
            write_json(family, ["x"])
            requests_binding = {"path": str(requests), "sha256": common.sha256(requests)}
            family_binding = {"path": str(family), "sha256": common.sha256(family)}
            entries = []
            for seed, value in ((42, 0.8), (43, 0.7), (44, 0.9)):
                train_plan = root / f"seed_{seed}" / "training_plan.json"
                evaluation = root / f"seed_{seed}" / "evaluation_plan.json"
                summary = root / f"seed_{seed}" / "summary.json"
                write_json(
                    train_plan,
                    {
                        "schema_version": "care_opd.causal_ablation_plan.v1",
                        "budget_contract": {"seed": seed},
                    },
                )
                train_binding = {"path": str(train_plan), "sha256": common.sha256(train_plan)}
                evaluation_arms = []
                source_bindings = {}
                for arm in ("full_selected", "outside_only"):
                    rollout_root = root / f"seed_{seed}" / "rollouts" / arm
                    rollout = rollout_root / "qzbc_rollouts.jsonl"
                    write_jsonl(rollout, [{"id": "x", "passed": True}])
                    evaluation_arms.append({"name": arm, "rollout_output": str(rollout_root)})
                    source_bindings[arm] = {"path": str(rollout), "sha256": common.sha256(rollout)}
                write_json(
                    evaluation,
                    {
                        "schema_version": "care_opd.causal_ablation_eval_plan.v1",
                        "training_plan": train_binding,
                        "requests": requests_binding,
                        "arms": evaluation_arms,
                    },
                )
                write_json(
                    summary,
                    fake_causal_summary(
                        value,
                        requests_binding=requests_binding,
                        family_binding=family_binding,
                        source_bindings=source_bindings,
                    ),
                )
                entries.append(
                    {
                        "training_seed": seed,
                        "training_plan": train_binding,
                        "evaluation_plan": {
                            "path": str(evaluation),
                            "sha256": common.sha256(evaluation),
                        },
                        "summary": {"path": str(summary)},
                        "arm_names": ["full_selected", "outside_only"],
                    }
                )
            plan = root / "evaluation_suite.json"
            write_json(
                plan,
                {
                    "schema_version": common.EVAL_SUITE_SCHEMA,
                    "training_suite": {"path": str(training), "sha256": common.sha256(training)},
                    "requests": requests_binding,
                    "family_clean_ids": family_binding,
                    "seeds": entries,
                },
            )
            result = summarize_multiseed.aggregate(plan, root / "out")
            perturbed = result["aggregate"]["full_selected"]["perturbed_tsr"]
            self.assertAlmostEqual(perturbed["mean"], 0.8)
            self.assertAlmostEqual(perturbed["sample_std"], 0.1)
            self.assertEqual(perturbed["worst_training_seeds"], [43])
            regression = result["aggregate"]["full_selected"]["reference_regression"]
            self.assertEqual(regression["direction"], "lower")
            self.assertEqual(regression["worst_training_seeds"], [42, 43, 44])
            self.assertIsNone(result["statistical_contract"]["confidence_interval"])
            for name in ("multiseed_summary.json", "aggregate.csv", "per_seed.csv", "multiseed_summary.md"):
                self.assertTrue((root / "out" / name).is_file())

    def test_nonfinite_metric_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_multiseed._finite_number(math.inf, where="test")


if __name__ == "__main__":
    unittest.main()
