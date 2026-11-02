"""CPU-only V4 contract tests, using synthetic tasks rather than test-set rows."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import common
import evaluate
import score_results


PROTOCOL_V4 = "care_opd.visual_transfer.v4"


def synthetic_task(condition: str = "clean") -> dict:
    missing = condition == "no_image"
    return {
        "task_id": "SYNTHETIC_TASK:" + condition,
        "family_id": "SYNTHETIC_FAMILY",
        "cluster_id": "SYNTHETIC_CLUSTER",
        "condition": condition,
        "question": "图像是否不包含所询问的目标结构？",
        "context_value": "CTX4A001",
        "image_path": None if missing else "/NOT_A_REAL_ASSET/SYNTHETIC_IMAGE.png",
        "image_sha256": None if missing else "SYNTHETIC_IMAGE_HASH",
    }


def synthetic_config(root: Path, evaluation_use: str = "validation") -> dict:
    return {
        "schema_version": "care_opd.visual_transfer.config.v1",
        "protocol_version": PROTOCOL_V4,
        "evaluation_use": evaluation_use,
        "tasks_dir": str(root / "tasks"),
        "run_root": str(root / "run"),
        "models": {
            name: str(root / "models" / name)
            for name in ("vlm", *common.TEXT_ARMS)
        },
        "generation": dict(evaluate.DEFAULT_GENERATION),
    }


def runtime_fixture(root: Path, condition: str = "clean") -> tuple[dict, dict, dict]:
    config = synthetic_config(root)
    item = synthetic_task(condition)
    manifest = {
        "protocol_sha256": "SYNTHETIC_PROTOCOL_HASH",
        "model_fingerprints": {
            name: "SYNTHETIC_MODEL_HASH:" + name
            for name in ("vlm", *common.TEXT_ARMS)
        },
    }
    observation = {
        "task_id": item["task_id"],
        "status": "completed",
        "raw_text": "图像概况：合成观察。问题相关观察：目标明确缺失。可判定边界：无其他信息。",
        "protocol_sha256": manifest["protocol_sha256"],
        "model_fingerprint": manifest["model_fingerprints"]["vlm"],
        "input_sha256": common.input_digest(
            common.observation_messages(item), item["image_sha256"]
        ),
        "latency_seconds": 0.01,
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "image_seen": item["image_path"] is not None,
    }
    common.append_row(
        Path(config["run_root"]) / "observations" / "observations.jsonl",
        observation,
    )
    return config, manifest, item


def capturing_generator(raw_text: str):
    class FakeGenerator:
        calls: list[dict] = []
        closed = 0

        def __init__(self, path, generation, *, visual):
            self.path = path
            self.visual = visual

        def generate(self, messages, image_path, max_tokens):
            type(self).calls.append({
                "model": self.path,
                "visual": self.visual,
                "messages": json.loads(common.canonical(messages)),
                "image_path": image_path,
                "max_tokens": max_tokens,
            })
            return {
                "raw_text": raw_text,
                "finish_reason": "length",
                "latency_seconds": 0.02,
                "prompt_tokens": 15,
                "completion_tokens": max_tokens,
                "image_seen": image_path is not None,
                "peak_gpu_memory_bytes": 0,
            }

        def close(self):
            type(self).closed += 1

    return FakeGenerator


def run_all_answer_arms(config: dict, manifest: dict, item: dict) -> None:
    evaluate.run_phase(config, manifest, [item], "direct")
    for arm in common.TEXT_ARMS:
        evaluate.run_phase(config, manifest, [item], "generate", arm)


class PromptContractV4Test(unittest.TestCase):
    def test_v4_answer_semantics_include_negative_propositions_and_evidence(self):
        self.assertEqual(common.PROTOCOL_VERSION, PROTOCOL_V4)
        for rule in (
            "当前证据明确支持问题中的命题，回答 YES",
            "当前证据明确否定问题中的命题，回答 NO",
            "否定证据也是有效证据，不等于证据不足",
            "问题本身若含否定词，应判断完整命题的真假",
            "这是来源属性，不是每道题都必须弃答的条件",
        ):
            self.assertIn(rule, common.ANSWER_SYSTEM)
        for answer in ("YES", "NO", "ABSTAIN"):
            section = {
                "heading": "visual_answer",
                "content": answer,
                "source_ids": [] if answer == "ABSTAIN" else ["image:current"],
            }
            self.assertIn(
                json.dumps(section, ensure_ascii=False, separators=(",", ":")),
                common.ANSWER_SYSTEM,
            )

    def test_observer_has_three_fields_and_preserves_uncertainty_boundary(self):
        for label in ("图像概况：", "问题相关观察：", "可判定边界："):
            self.assertEqual(common.OBSERVE_SYSTEM.count(label), 1)
        self.assertIn("看不清、未覆盖不能当作不存在", common.OBSERVE_SYSTEM)
        self.assertIn("不要输出最终 YES、NO 或 ABSTAIN 标签", common.OBSERVE_SYSTEM)
        item = synthetic_task()
        messages = common.observation_messages(item)
        self.assertEqual(messages[0], {"role": "system", "content": common.OBSERVE_SYSTEM})
        self.assertEqual(messages[1], {"role": "user", "content": "当前问题：" + item["question"]})
        self.assertNotIn(item["context_value"], common.canonical(messages))

    def test_private_annotations_and_swap_labels_cannot_change_messages(self):
        baseline = synthetic_task()
        contaminated = dict(
            baseline,
            condition="swapped_image",
            task_id="PRIVATE_TASK_LABEL",
            family_id="PRIVATE_FAMILY_LABEL",
            cluster_id="PRIVATE_CLUSTER_LABEL",
            image_path="/PRIVATE_IMAGE_LOCATION/file.png",
            image_sha256="PRIVATE_IMAGE_HASH",
            answer="PRIVATE_GOLD_ANSWER",
            gold="PRIVATE_GOLD_OBJECT",
            original_answer="PRIVATE_ORIGINAL_ANSWER",
            reference_answer="PRIVATE_REFERENCE_ANSWER",
            counterpart_qid="PRIVATE_DONOR_QID",
            source_image_id="PRIVATE_SOURCE_IMAGE_ID",
            split="PRIVATE_SPLIT",
        )
        observation = {"status": "completed", "raw_text": "固定的合成观察"}
        pairs = [
            (common.observation_messages(baseline), common.observation_messages(contaminated)),
            (common.answer_messages(baseline, observation), common.answer_messages(contaminated, observation)),
            (common.answer_messages(baseline, None, direct=True), common.answer_messages(contaminated, None, direct=True)),
        ]
        for expected, actual in pairs:
            self.assertEqual(expected, actual)
            self.assertNotIn("PRIVATE_", common.canonical(actual))
            self.assertNotIn("swapped_image", common.canonical(actual))
            self.assertNotIn("SYNTHETIC_", common.canonical(actual))

    def test_source_metadata_is_uniform_and_does_not_encode_a_gold_answer(self):
        observation = {"status": "completed", "raw_text": "固定的合成观察"}
        for direct in (False, True):
            for condition in common.CONDITIONS:
                with self.subTest(direct=direct, condition=condition):
                    item = synthetic_task(condition)
                    messages = common.answer_messages(item, observation, direct=direct)
                    payload = json.loads(messages[1]["content"])
                    packet = payload["evidence_packet"]
                    self.assertEqual(set(packet), {"text:context", "image:current"})
                    for source_id, evidence in packet.items():
                        self.assertEqual(evidence["source_id"], source_id)
                    self.assertEqual(packet["text:context"]["context_value"], item["context_value"])
                    self.assertEqual(
                        packet["image:current"]["availability"],
                        "missing" if condition == "no_image" else "available",
                    )
                    for gold in ("YES", "NO", "ABSTAIN"):
                        changed = dict(item, answer=gold, gold=gold, original_answer=gold)
                        self.assertEqual(
                            messages,
                            common.answer_messages(changed, observation, direct=direct),
                        )
                    if not direct:
                        self.assertEqual(packet["image:current"]["observation"], observation["raw_text"])
                        self.assertEqual(packet["image:current"]["origin"], "frozen_visual_model_unverified_observation")

    def test_incomplete_shared_observation_is_rejected_not_fabricated(self):
        for observation in (None, {"status": "infra_error", "raw_text": "unusable"}):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                common.answer_messages(synthetic_task(), observation)


class RuntimeContractV4Test(unittest.TestCase):
    def test_all_five_arms_receive_identical_answer_system(self):
        for condition in common.CONDITIONS:
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as folder:
                config, manifest, item = runtime_fixture(Path(folder), condition)
                fake = capturing_generator("NOT_VALID_JSON")
                with patch.object(evaluate, "HFGenerator", fake), contextlib.redirect_stdout(io.StringIO()):
                    run_all_answer_arms(config, manifest, item)
                self.assertEqual(len(fake.calls), len(common.ARMS))
                self.assertEqual(fake.closed, len(common.ARMS))
                for call in fake.calls:
                    self.assertEqual(call["messages"][0], {"role": "system", "content": common.ANSWER_SYSTEM})
                direct, *indirect = fake.calls
                self.assertTrue(direct["visual"])
                self.assertEqual(direct["image_path"], item["image_path"])
                for call in indirect:
                    self.assertFalse(call["visual"])
                    self.assertIsNone(call["image_path"])
                    self.assertEqual(call["messages"], indirect[0]["messages"])

    def test_bad_first_outputs_remain_verbatim_and_resume_never_retries(self):
        # These are deliberately malformed/truncated, not expected model answers.
        for raw_text in ("", "YES", '{"sections":[', '{"answer":"NO","source_ids":[]}'):
            for condition in common.CONDITIONS:
                with self.subTest(raw_text=raw_text, condition=condition), tempfile.TemporaryDirectory() as folder:
                    config, manifest, item = runtime_fixture(Path(folder), condition)
                    fake = capturing_generator(raw_text)
                    with patch.object(evaluate, "HFGenerator", fake), contextlib.redirect_stdout(io.StringIO()):
                        run_all_answer_arms(config, manifest, item)
                        paths = [
                            Path(config["run_root"]) / "arms" / arm / "responses.jsonl"
                            for arm in common.ARMS
                        ]
                        original_bytes = {path: path.read_bytes() for path in paths}
                        run_all_answer_arms(config, manifest, item)
                    self.assertEqual(len(fake.calls), len(common.ARMS))
                    for path in paths:
                        self.assertEqual(path.read_bytes(), original_bytes[path])
                        rows = common.read_jsonl(path)
                        self.assertEqual(len(rows), 1)
                        self.assertEqual(rows[0]["raw_text"], raw_text)
                        self.assertEqual(rows[0]["finish_reason"], "length")
                        self.assertEqual(rows[0]["status"], "completed")
                        self.assertFalse((path.parent / "infrastructure_errors.jsonl").exists())


class ConfigurationV4Test(unittest.TestCase):
    def test_missing_or_old_protocol_configs_are_rejected_without_rewriting(self):
        for version in (None, "care_opd.visual_transfer.v1", "care_opd.visual_transfer.v2", "care_opd.visual_transfer.v3"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as folder:
                config = synthetic_config(Path(folder))
                if version is None:
                    config.pop("protocol_version")
                else:
                    config["protocol_version"] = version
                path = Path(folder) / "old_config.json"
                common.write_json_once(path, config)
                original_bytes = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "protocol_version"):
                    evaluate.configuration(path)
                self.assertEqual(path.read_bytes(), original_bytes)
                self.assertFalse(Path(config["run_root"]).exists())

    def test_missing_or_unapproved_evaluation_use_is_rejected(self):
        for purpose in (None, "test", "confirmatory", ""):
            with self.subTest(purpose=purpose), tempfile.TemporaryDirectory() as folder:
                config = synthetic_config(Path(folder))
                if purpose is None:
                    config.pop("evaluation_use")
                else:
                    config["evaluation_use"] = purpose
                path = Path(folder) / "config.json"
                common.write_json_once(path, config)
                with self.assertRaisesRegex(ValueError, "evaluation_use"):
                    evaluate.configuration(path)

    def test_explicit_v4_configs_accept_only_disclosed_evaluation_uses(self):
        self.assertEqual(set(evaluate.EVALUATION_USES), {"validation", "post_test_protocol_revision"})
        for purpose in evaluate.EVALUATION_USES:
            with self.subTest(purpose=purpose), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                config = synthetic_config(Path(folder), purpose)
                common.write_json_once(path, config)
                loaded = evaluate.configuration(path)
                self.assertEqual(loaded["protocol_version"], PROTOCOL_V4)
                self.assertEqual(loaded["evaluation_use"], purpose)
                self.assertEqual(loaded["generation"], evaluate.DEFAULT_GENERATION)

    def test_evaluation_use_requires_matching_frozen_source_split(self):
        for purpose, split in (("validation", "validation"), ("post_test_protocol_revision", "test")):
            with self.subTest(purpose=purpose, split=split):
                evaluate.validate_evaluation_use(
                    {"evaluation_use": purpose}, {"dataset": {"source_split": split}}
                )
        for purpose in ("validation", "post_test_protocol_revision", "test", None):
            for selection in ({}, {"dataset": {}}, {"dataset": {"source_split": "train"}},
                              {"dataset": {"source_split": "unknown"}}):
                with self.subTest(purpose=purpose, selection=selection), self.assertRaisesRegex(ValueError, "source_split"):
                    evaluate.validate_evaluation_use({"evaluation_use": purpose}, selection)
        for purpose, split in (("validation", "test"), ("post_test_protocol_revision", "validation")):
            with self.subTest(purpose=purpose, split=split), self.assertRaisesRegex(ValueError, "source_split"):
                evaluate.validate_evaluation_use(
                    {"evaluation_use": purpose}, {"dataset": {"source_split": split}}
                )

    def test_mislabeled_split_stops_binding_before_hashes_or_run_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            config = synthetic_config(Path(folder), "validation")
            selection = {"dataset": {"source_split": "test"}}
            with patch.object(evaluate, "source_fingerprints") as sources, patch.object(evaluate, "environment") as environment, patch.object(evaluate, "fingerprint_model") as models:
                with self.assertRaisesRegex(ValueError, "source_split"):
                    evaluate.bind_run(config, [synthetic_task()], selection)
                sources.assert_not_called()
                environment.assert_not_called()
                models.assert_not_called()
            self.assertFalse(Path(config["run_root"]).exists())

    def test_init_config_requires_evaluation_use_and_writes_v4(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            argv = [
                "init-config", "--output", str(root / "config.json"),
                "--tasks-dir", str(root / "tasks"), "--run-root", str(root / "run"),
            ]
            for name in ("vlm", *common.TEXT_ARMS):
                argv.extend(["--" + name.replace("_", "-"), str(root / "models" / name)])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as missing:
                evaluate.parser().parse_args(argv)
            self.assertEqual(missing.exception.code, 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as invalid:
                evaluate.parser().parse_args(argv + ["--evaluation-use", "test"])
            self.assertEqual(invalid.exception.code, 2)
            with patch.object(sys, "argv", ["evaluate.py", *argv, "--evaluation-use", "post_test_protocol_revision"]), contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            written = common.read_json(root / "config.json")
            self.assertEqual(written["protocol_version"], PROTOCOL_V4)
            self.assertEqual(written["evaluation_use"], "post_test_protocol_revision")
            self.assertEqual(evaluate.configuration(root / "config.json")["evaluation_use"], written["evaluation_use"])


class ScoringContractV4Test(unittest.TestCase):
    def test_source_scoring_still_requires_image_reference_for_yes_and_no(self):
        for answer in ("YES", "NO", "ABSTAIN"):
            payload = {
                "title": "合成测试", "summary": "仅验证格式", "warnings": [],
                "next_actions": [], "disclaimer": "仅用于合成测试",
                "sections": [
                    {"heading": "visual_answer", "content": answer,
                     "source_ids": [] if answer == "ABSTAIN" else ["image:current"]},
                    {"heading": "context_value", "content": "CTX4A001", "source_ids": ["text:context"]},
                ],
            }
            gold = {"answer": answer, "condition": "clean", "context_value": "CTX4A001"}
            with self.subTest(answer=answer):
                scored = score_results._score_native_output(common.canonical(payload), gold)
                self.assertTrue(scored["answer_correct"])
                self.assertTrue(scored["source_valid"])
                payload["sections"][0]["source_ids"] = ["image:current"] if answer == "ABSTAIN" else []
                incorrect_source = score_results._score_native_output(common.canonical(payload), gold)
                self.assertTrue(incorrect_source["answer_correct"])
                self.assertFalse(incorrect_source["source_valid"])
                different_gold = dict(gold, answer="NO" if answer == "YES" else "YES")
                self.assertFalse(score_results._score_native_output(common.canonical(payload), different_gold)["answer_correct"])


if __name__ == "__main__":
    unittest.main()
