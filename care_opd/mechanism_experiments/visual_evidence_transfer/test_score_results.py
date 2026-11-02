"""CPU-only scorer tests; no row in this file is an empirical result."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import common
import score_results


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(common.canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _native(answer: str, context: str) -> str:
    return common.canonical(
        {
            "title": "合成评分测试",
            "summary": "仅用于验证评分语义",
            "sections": [
                {
                    "heading": "visual_answer",
                    "content": answer,
                    "source_ids": ["image:current"] if answer in {"YES", "NO"} else [],
                },
                {
                    "heading": "context_value",
                    "content": context,
                    "source_ids": ["text:context"],
                },
            ],
            "warnings": [],
            "next_actions": [],
            "disclaimer": "合成单元测试输出，不是医学结论",
        }
    )


class FrozenFixture:
    """Create one matched family and one unmatched family."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.tasks_dir = root / "tasks"
        self.run_root = root / "run"
        self.output = root / "score"
        image_root = root / "dataset-assets"  # Deliberately outside tasks_dir.
        image_root.mkdir(parents=True)
        images = {}
        for name, value in (("a", b"visual-a"), ("b", b"visual-b"), ("c", b"visual-c")):
            path = (image_root / f"{name}.png").resolve()
            path.write_bytes(value)
            images[name] = (path, common.file_sha256(path))

        task_specs = [
            ("f1-clean", "family-1", "cluster-shared", "clean", images["a"]),
            ("f1-none", "family-1", "cluster-shared", "no_image", (None, None)),
            ("f1-swap", "family-1", "cluster-shared", "swapped_image", images["b"]),
            ("f2-clean", "family-2", "cluster-unmatched", "clean", images["c"]),
            ("f2-none", "family-2", "cluster-unmatched", "no_image", (None, None)),
        ]
        contexts = {"family-1": "CTX00001", "family-2": "CTX00002"}
        self.tasks: list[dict] = []
        self.gold: list[dict] = []
        for task_id, family_id, cluster_id, condition, image in task_specs:
            path, image_sha = image
            task = {
                "schema_version": "care_opd.visual_transfer.task.v1",
                "task_id": task_id,
                "family_id": family_id,
                "cluster_id": cluster_id,
                "condition": condition,
                "question": "图中是否存在目标结构？",
                "image_path": str(path) if path is not None else None,
                "image_sha256": image_sha,
                "context_value": contexts[family_id],
            }
            original = "YES"
            answer = {"clean": "YES", "no_image": "ABSTAIN", "swapped_image": "NO"}[
                condition
            ]
            gold = {
                "schema_version": "care_opd.visual_transfer.gold.v1",
                "task_id": task_id,
                "family_id": family_id,
                "cluster_id": cluster_id,
                "condition": condition,
                "answer": answer,
                "original_answer": original,
                "context_value": contexts[family_id],
                "counterpart_qid": "donor-outside-selection" if condition == "swapped_image" else None,
            }
            self.tasks.append(task)
            self.gold.append(gold)

        tasks_path = self.tasks_dir / "tasks.jsonl"
        gold_path = self.tasks_dir / "gold.private.jsonl"
        _write_jsonl(tasks_path, self.tasks)
        _write_jsonl(gold_path, self.gold)
        selection = {
            "schema_version": "care_opd.visual_transfer.selection.v1",
            "tasks_sha256": common.file_sha256(tasks_path),
            "gold_sha256": common.file_sha256(gold_path),
        }
        selection_path = self.tasks_dir / "selection_manifest.json"
        _write_json(selection_path, selection)

        self.fingerprints = {
            name: common.digest({"synthetic-model": name})
            for name in ("vlm", *common.TEXT_ARMS)
        }
        manifest = {
            "schema_version": "care_opd.visual_transfer.run.v1",
            "config": {"synthetic_test_only": True},
            "tasks_sha256": selection["tasks_sha256"],
            "gold_sha256": selection["gold_sha256"],
            "selection_manifest_sha256": common.file_sha256(selection_path),
            "arms": list(common.ARMS),
            "model_assets": {
                name: {"sha256": digest, "files": {}, "stat": {}}
                for name, digest in self.fingerprints.items()
            },
            "model_fingerprints": dict(self.fingerprints),
        }
        manifest["protocol_sha256"] = common.digest(manifest)
        self.manifest = manifest
        _write_json(self.run_root / "run_manifest.json", manifest)

        self.observations: dict[str, dict] = {}
        observation_rows = []
        for task in self.tasks:
            visible = task["condition"] != "no_image"
            observation = {
                "task_id": task["task_id"],
                "family_id": task["family_id"],
                "cluster_id": task["cluster_id"],
                "condition": task["condition"],
                "arm": "observations",
                "protocol_sha256": manifest["protocol_sha256"],
                "model_fingerprint": self.fingerprints["vlm"],
                "input_sha256": common.input_digest(
                    common.observation_messages(task), task["image_sha256"]
                ),
                "observation_sha256": None,
                "image_sha256": task["image_sha256"],
                "raw_text": "合成可见观察" if visible else "未提供图像，无法获得视觉观察。",
                "finish_reason": "stop" if visible else "not_applicable",
                "latency_seconds": 0.25 if visible else 0.0,
                "prompt_tokens": 11 if visible else 0,
                "completion_tokens": 7 if visible else 0,
                "image_seen": visible,
                "peak_gpu_memory_bytes": 0,
                "status": "completed",
                "upstream_latency_seconds": 0.0,
                "end_to_end_latency_seconds": 0.25 if visible else 0.0,
                "upstream_prompt_tokens": 0,
                "upstream_completion_tokens": 0,
                "upstream_image_seen": False,
            }
            self.observations[task["task_id"]] = observation
            observation_rows.append(observation)
        _write_jsonl(
            self.run_root / "observations" / "observations.jsonl", observation_rows
        )

        self.response_rows: dict[str, list[dict]] = {}
        gold_by_id = {row["task_id"]: row for row in self.gold}
        for arm in common.ARMS:
            rows = []
            direct = arm == "vlm_direct"
            for task in self.tasks:
                observation = None if direct else self.observations[task["task_id"]]
                gold = gold_by_id[task["task_id"]]
                upstream_latency = 0.0 if direct else observation["latency_seconds"]
                row = {
                    "task_id": task["task_id"],
                    "family_id": task["family_id"],
                    "cluster_id": task["cluster_id"],
                    "condition": task["condition"],
                    "arm": arm,
                    "status": "completed",
                    "raw_text": _native(gold["answer"], gold["context_value"]),
                    "finish_reason": "stop",
                    "latency_seconds": 0.5,
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "protocol_sha256": manifest["protocol_sha256"],
                    "model_fingerprint": self.fingerprints["vlm" if direct else arm],
                    "input_sha256": common.input_digest(
                        common.answer_messages(task, observation, direct=direct),
                        task["image_sha256"] if direct else None,
                    ),
                    "observation_sha256": (
                        None if direct else common.digest(observation)
                    ),
                    "image_sha256": task["image_sha256"],
                    "image_seen": direct and task["condition"] != "no_image",
                    "peak_gpu_memory_bytes": 0,
                    "upstream_latency_seconds": upstream_latency,
                    "end_to_end_latency_seconds": 0.5 + upstream_latency,
                    "upstream_prompt_tokens": 0 if direct else observation["prompt_tokens"],
                    "upstream_completion_tokens": (
                        0 if direct else observation["completion_tokens"]
                    ),
                    "upstream_image_seen": (
                        False if direct else observation["image_seen"]
                    ),
                }
                rows.append(row)
            self.response_rows[arm] = rows
            self.write_arm(arm)

        self.config = {
            "schema_version": "care_opd.visual_transfer.config.v1",
            "tasks_dir": str(self.tasks_dir),
            "run_root": str(self.run_root),
        }
        _write_json(root / "CONFIG.json", self.config)
        self.config_path = root / "CONFIG.json"

    def write_arm(self, arm: str) -> None:
        _write_jsonl(
            self.run_root / "arms" / arm / "responses.jsonl",
            self.response_rows[arm],
        )

    def score(self, output: Path | None = None) -> dict:
        return score_results.score_results(
            tasks_dir=self.tasks_dir,
            run_root=self.run_root,
            output=output or self.output,
            bootstrap=25,
            seed=1234,
        )


class NativeOutputTest(unittest.TestCase):
    def test_plain_and_complete_outer_fence_are_accepted(self) -> None:
        text = _native("YES", "CTX00001")
        self.assertIsNotNone(score_results.parse_native_output(text)[0])
        self.assertIsNotNone(
            score_results.parse_native_output("```json\n" + text + "\n```")[0]
        )

    def test_prefix_incomplete_fence_and_duplicate_key_are_rejected(self) -> None:
        text = _native("YES", "CTX00001")
        self.assertIsNotNone(score_results.parse_native_output("preface\n" + text)[1])
        self.assertIsNotNone(score_results.parse_native_output("```json\n" + text)[1])
        self.assertIsNotNone(
            score_results.parse_native_output('{"title":"a","title":"b"}')[1]
        )


class ScoreResultsTest(unittest.TestCase):
    def test_complete_optional_swap_metrics_and_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            payload = fixture.score()
            self.assertEqual(payload["status"], "COMPLETE")
            self.assertEqual(payload["bundle_audit"]["matched_swapped_family_count"], 1)
            self.assertEqual(payload["bundle_audit"]["unmatched_family_count"], 1)
            family = payload["per_arm_family_metrics"]["care_opd"]
            self.assertEqual(family["clean_swapped_pair_eligible_count"], 1)
            self.assertEqual(family["clean_swapped_pair_ineligible_unmatched_count"], 1)
            self.assertEqual(family["clean_swapped_both_answer_correct"], 1.0)
            metrics = {
                (row["arm"], row["condition"]): row
                for row in payload["per_arm_condition"]
            }
            self.assertEqual(metrics[("care_opd", "clean")]["total_expected"], 2)
            self.assertEqual(metrics[("care_opd", "swapped_image")]["total_expected"], 1)
            self.assertEqual(
                metrics[("care_opd", "clean")]["end_to_end_latency_mean_seconds"],
                0.75,
            )
            for name in ("metrics.json", "metrics.csv", "paired_effects.csv", "summary.md"):
                self.assertTrue((fixture.output / name).is_file(), name)

    def test_answer_content_can_score_when_outer_schema_is_bad(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            row = fixture.response_rows["care_opd"][0]
            parsed = json.loads(row["raw_text"])
            parsed["unexpected"] = "break only the contract shape"
            row["raw_text"] = common.canonical(parsed)
            fixture.write_arm("care_opd")
            payload = fixture.score()
            clean = next(
                item
                for item in payload["per_arm_condition"]
                if item["arm"] == "care_opd" and item["condition"] == "clean"
            )
            self.assertEqual(clean["answer_correct_count"], 2)
            self.assertEqual(clean["format_valid_count"], 1)
            self.assertEqual(clean["strict_contract_success_count"], 1)
            self.assertEqual(clean["invalid_output_count"], 1)

    def test_no_image_guess_is_not_counted_as_abstention_success(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            # Guessing the original YES answer is selectively correct as a guess,
            # but wrong against the no_image ABSTAIN gold and violates source use.
            for row in fixture.response_rows["care_opd"]:
                if row["task_id"] == "f1-none":
                    row["raw_text"] = _native("YES", "CTX00001")
            fixture.write_arm("care_opd")
            payload = fixture.score()
            no_image = next(
                item
                for item in payload["per_arm_condition"]
                if item["arm"] == "care_opd" and item["condition"] == "no_image"
            )
            self.assertEqual(no_image["answer_accuracy"], 0.5)
            self.assertEqual(no_image["answered_rate"], 0.5)
            self.assertEqual(no_image["selective_accuracy"], 1.0)
            self.assertEqual(
                no_image["selective_accuracy_definition"],
                "original_answer_accuracy_among_answered_no_image",
            )

    def test_missing_row_with_infra_ledger_is_incomplete_and_in_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            missing = fixture.response_rows["care_opd"].pop()
            fixture.write_arm("care_opd")
            _write_jsonl(
                fixture.run_root
                / "arms"
                / "care_opd"
                / "infrastructure_errors.jsonl",
                [
                    {
                        "task_id": missing["task_id"],
                        "status": "infra_error",
                        "error": "SyntheticOOM: test only",
                    }
                ],
            )
            payload = fixture.score()
            self.assertEqual(payload["status"], "INCOMPLETE")
            diagnostic = next(
                item for item in payload["arm_diagnostics"] if item["arm"] == "care_opd"
            )
            self.assertEqual(diagnostic["infra_error_rows"], 1)
            overall = next(
                item
                for item in payload["per_arm_condition"]
                if item["arm"] == "care_opd" and item["condition"] == "all"
            )
            self.assertEqual(overall["total_expected"], 5)
            self.assertEqual(overall["strict_contract_success_count"], 4)

    def test_historical_infra_does_not_invalidate_later_first_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            _write_jsonl(
                fixture.run_root / "arms" / "care_opd" / "infra_attempts.jsonl",
                [
                    {
                        "task_id": "f1-clean",
                        "status": "infra_error",
                        "error": "Synthetic transient failure",
                    }
                ],
            )
            payload = fixture.score()
            self.assertEqual(payload["status"], "COMPLETE")
            diagnostic = next(
                item for item in payload["arm_diagnostics"] if item["arm"] == "care_opd"
            )
            self.assertEqual(diagnostic["historical_infra_attempts"], 1)
            self.assertTrue(diagnostic["complete_for_inference"])

    def test_observation_digest_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            rows = list(fixture.observations.values())
            rows[0] = dict(rows[0], raw_text="tampered after generation")
            _write_jsonl(
                fixture.run_root / "observations" / "observations.jsonl", rows
            )
            with self.assertRaisesRegex(
                score_results.ScoringError, "does not bind actual observation"
            ):
                fixture.score()

    def test_manifest_closure_and_model_asset_binding_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            manifest = dict(fixture.manifest)
            manifest["scope"] = "post-hoc mutation"
            _write_json(fixture.run_root / "run_manifest.json", manifest)
            with self.assertRaisesRegex(score_results.ScoringError, "does not close"):
                fixture.score()

        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            manifest = dict(fixture.manifest)
            manifest["model_assets"] = dict(manifest["model_assets"])
            manifest["model_assets"]["care_opd"] = dict(
                manifest["model_assets"]["care_opd"], sha256="0" * 64
            )
            manifest["protocol_sha256"] = common.digest(
                {key: value for key, value in manifest.items() if key != "protocol_sha256"}
            )
            _write_json(fixture.run_root / "run_manifest.json", manifest)
            with self.assertRaisesRegex(score_results.ScoringError, "model_assets"):
                fixture.score()

    def test_identical_image_bytes_cannot_cross_bootstrap_components(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            first = next(row for row in fixture.tasks if row["task_id"] == "f1-clean")
            second = next(row for row in fixture.tasks if row["task_id"] == "f2-clean")
            second["image_path"] = first["image_path"]
            second["image_sha256"] = first["image_sha256"]
            tasks_path = fixture.tasks_dir / "tasks.jsonl"
            _write_jsonl(tasks_path, fixture.tasks)
            selection_path = fixture.tasks_dir / "selection_manifest.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["tasks_sha256"] = common.file_sha256(tasks_path)
            _write_json(selection_path, selection)
            with self.assertRaisesRegex(
                score_results.ScoringError, "different bootstrap components"
            ):
                score_results._validate_bundle(fixture.tasks_dir)

    def test_cli_config_uses_run_summary_default(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            with contextlib.redirect_stdout(io.StringIO()):
                rc = score_results.main(
                    ["--config", str(fixture.config_path), "--bootstrap", "10"]
                )
            self.assertEqual(rc, 0)
            expected = fixture.run_root / "summary"
            for name in ("metrics.json", "metrics.csv", "paired_effects.csv", "summary.md"):
                self.assertTrue((expected / name).is_file(), name)

    def test_unknown_and_duplicate_response_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            fixture.response_rows["care_opd"][0]["task_id"] = "unknown"
            fixture.write_arm("care_opd")
            with self.assertRaisesRegex(score_results.ScoringError, "unknown task_id"):
                fixture.score()

        with tempfile.TemporaryDirectory() as folder:
            fixture = FrozenFixture(Path(folder))
            fixture.response_rows["care_opd"].append(
                dict(fixture.response_rows["care_opd"][0])
            )
            fixture.write_arm("care_opd")
            with self.assertRaisesRegex(score_results.ScoringError, "duplicate response"):
                fixture.score()


class BootstrapTest(unittest.TestCase):
    def test_cluster_bootstrap_is_deterministic_and_empty_safe(self) -> None:
        candidate = [
            {"task_id": "a", "cluster_id": "c1", "condition": "clean", "answer_correct": True},
            {"task_id": "b", "cluster_id": "c1", "condition": "clean", "answer_correct": False},
            {"task_id": "c", "cluster_id": "c2", "condition": "no_image", "answer_correct": True},
        ]
        baseline = [
            {"task_id": "a", "cluster_id": "c1", "condition": "clean", "answer_correct": False},
            {"task_id": "b", "cluster_id": "c1", "condition": "clean", "answer_correct": False},
            {"task_id": "c", "cluster_id": "c2", "condition": "no_image", "answer_correct": True},
        ]
        first = score_results.paired_cluster_bootstrap(
            candidate,
            baseline,
            metric="answer_correct",
            condition="all",
            replicates=50,
            seed=7,
        )
        second = score_results.paired_cluster_bootstrap(
            candidate,
            baseline,
            metric="answer_correct",
            condition="all",
            replicates=50,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 2)
        empty = score_results.paired_cluster_bootstrap(
            candidate,
            baseline,
            metric="answer_correct",
            condition="swapped_image",
            replicates=10,
            seed=7,
        )
        self.assertFalse(empty["estimable"])
        self.assertIsNone(empty["delta"])
        one_cluster = score_results.paired_cluster_bootstrap(
            candidate[:2],
            baseline[:2],
            metric="answer_correct",
            condition="clean",
            replicates=10,
            seed=7,
        )
        self.assertTrue(one_cluster["estimable"])
        self.assertFalse(one_cluster["ci_estimable"])
        self.assertIsNone(one_cluster["ci95_low"])
        self.assertIsNone(one_cluster["ci95_high"])


if __name__ == "__main__":
    unittest.main()
