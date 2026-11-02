from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_views
import evaluate_arms
import launch_parallel
import run_single_round


def row(item_id: str, *, span=(1, 3), clean=False, rollout="abcdef"):
    return {
        "id": item_id,
        "variant": "clean" if clean else "bad_history",
        "rollout": rollout,
        "teacher_messages": [{"role": "user", "content": "x"}],
        "repair_spans": [] if clean else [{"start": span[0], "end": span[1], "component": "summary", "risk_weight": 2.0}],
        "component_mask": [] if clean else ["summary"],
        "clean_control": clean,
        "passed": clean,
        "risk_weight": 1.0 if clean else 2.0,
    }


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


class ViewTests(unittest.TestCase):
    def test_causal_views_preserve_ids_and_outside_is_complement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "opd.jsonl"
            write_jsonl(source, [row("failure"), row("clean", clean=True)])
            manifest = build_views.build_causal_views(source, root / "views", seed=7)
            outside = build_views._read_jsonl(root / "views" / "outside_only.jsonl")
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual([item["id"] for item in outside], ["failure", "clean"])
            self.assertEqual([(s["start"], s["end"]) for s in outside[0]["repair_spans"]], [(0, 1), (3, 6)])
            self.assertEqual(outside[1]["repair_spans"], [])

    def test_freshness_is_nonclean_exact_id_intersection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fresh, stale = root / "fresh.jsonl", root / "stale.jsonl"
            write_jsonl(fresh, [row("a", rollout="fresh"), row("clean", clean=True), row("fresh-only")])
            write_jsonl(stale, [row("a", rollout="stale"), row("clean", clean=True), row("stale-only")])
            manifest = build_views.build_freshness_views(fresh, stale, root / "views", seed=1, max_rows=None, min_rows=1)
            self.assertEqual(manifest["cohort"], "persistent_failure_only")
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(build_views._read_jsonl(root / "views" / "fresh_matched.jsonl")[0]["id"], "a")

    def test_duplicate_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "x.jsonl"
            write_jsonl(source, [row("a"), row("a")])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_views._read_jsonl(source)


class PlanTests(unittest.TestCase):
    def test_plan_requires_same_order_and_is_plan_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            views = root / "views"
            views.mkdir()
            write_jsonl(views / "a.jsonl", [row("x"), row("y")])
            write_jsonl(views / "b.jsonl", [row("x"), row("y")])
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({
                "schema_version": run_single_round.MATRIX_SCHEMA,
                "variants": [
                    {"name": "a", "data_view": "a.jsonl", "overrides": {"baseline_mode": "care_opd"}},
                    {
                        "name": "b",
                        "data_view": "b.jsonl",
                        "overrides": {
                            "baseline_mode": "care_opd",
                            "mask_mode": "random_token_matched",
                            "random_mask_seed": 20260815,
                            "oracle_risk_weighting_mode": "enabled",
                            "use_teacher_trust": False,
                        },
                    },
                ],
            }), encoding="utf-8")
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                matrix=matrix, views_dir=views, train_config=config,
                student_model="student", teacher_model="teacher", reference_model=None,
                output_root=root / "out", max_steps=3, seed=42, nproc_per_node=2,
                save_steps=25, python="python", resume=False, execute=False,
            )
            plan, commands = run_single_round.build_plan(args)
            self.assertTrue(plan["budget_contract"]["same_row_ids_and_order"])
            self.assertEqual(plan["budget_contract"]["checkpoint_save_steps"], 25)
            self.assertEqual(len(commands), 2)
            self.assertIn("--max_steps", commands[0])
            self.assertEqual(commands[0][commands[0].index("--save_steps") + 1], "25")
            self.assertIn("training_code", plan)
            self.assertIn("--random_mask_seed", commands[1])
            self.assertIn("--oracle_risk_weighting_mode", commands[1])

    def test_all6_wrapper_freezes_budget_and_detaches(self):
        launcher = (HERE / "launch_all6_detached.sh").read_text(encoding="utf-8")
        stopper = (HERE / "stop_all6.sh").read_text(encoding="utf-8")
        self.assertIn("single_round_all6_v2", launcher)
        self.assertIn("--max-steps 199", launcher)
        self.assertIn("--save-steps 100", launcher)
        self.assertIn("--resume", launcher)
        self.assertIn("--seed 42", launcher)
        self.assertIn('GPUS="${GPUS:-4,5,6,7}"', launcher)
        self.assertIn("nohup setsid", launcher)
        self.assertNotIn("nvidia-smi", launcher)
        self.assertIn('kill -TERM -- "-$pid"', stopper)
        self.assertIn("is not its own session leader", stopper)

    def test_eval_plan_reuses_qzbc_rollout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            requests = root / "requests.jsonl"
            write_jsonl(requests, [row("x")])
            training = root / "experiment_plan.json"
            training.write_text(json.dumps({
                "schema_version": "care_opd.causal_ablation_plan.v1",
                "student_initialization": {"value": "base"},
                "arms": [{"name": "full_selected", "output_dir": str(root / "train" / "full_selected")}],
            }), encoding="utf-8")
            args = argparse.Namespace(
                experiment_plan=training, requests=requests, output_root=root / "eval",
                python="python", num_workers=1, devices="cuda:0", dtype="bfloat16",
                max_new_tokens=1200, seed=42, execute=False,
            )
            plan, commands = evaluate_arms.build_plan(args)
            self.assertEqual(plan["protocol"]["generator"], "care_opd.qzbc_rollout")
            self.assertIn("--adapter", commands[0])

    def test_eval_execution_preflight_requires_complete_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arm_root = root / "train" / "arm"
            final = arm_root / "final"
            final.mkdir(parents=True)
            (final / "adapter_config.json").write_text("{}", encoding="utf-8")
            (final / "adapter_model.safetensors").write_bytes(b"weights")
            (final / "run_metadata.json").write_text(
                json.dumps({"global_step": 3}), encoding="utf-8"
            )
            write_jsonl(
                arm_root / "metrics.jsonl",
                [{"event": "complete", "global_step": 3}],
            )
            plan = {
                "budget_contract": {"max_steps_per_arm": 3},
                "arms": [{"name": "arm", "output_dir": str(arm_root)}],
            }
            evaluate_arms._validate_training_outputs(plan)
            (final / "run_metadata.json").write_text(
                json.dumps({"global_step": 2}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "global_step"):
                evaluate_arms._validate_training_outputs(plan)

    def test_parallel_launcher_is_plan_only_without_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "arms": [{"name": "a", "command": ["python", "--nproc_per_node=1"]}]
            }), encoding="utf-8")
            code = launch_parallel.main([
                "--plan", str(plan), "--gpus", "4,5", "--log-dir", str(root / "logs")
            ])
            self.assertEqual(code, 0)
            self.assertFalse((root / "logs").exists())


if __name__ == "__main__":
    unittest.main()
