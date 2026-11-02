"""CPU-only synthetic integration tests. These are NOT model evaluation results."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import common
import evaluate
import prepare_slake
import score_results


def png(red):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes((0, red, 0, 0)))) + chunk(b"IEND", b""))


class SyntheticGenerator:
    calls = 0

    def __init__(self, path, generation, *, visual):
        self.label = Path(path).name

    def generate(self, messages, image_path, max_tokens):
        type(self).calls += 1
        if messages[0]["content"] == common.OBSERVE_SYSTEM:
            raw = "仅供单元测试的合成观察，不是实际医学图像结果。"
        elif self.label == "tropd_full":
            raw = "{}"  # Completed but invalid output must never be filtered/retried.
        else:
            packet = json.loads(messages[-1]["content"])["evidence_packet"]
            available = packet["image:current"]["availability"] == "available"
            raw = common.canonical({
                "title": "合成测试", "summary": "模拟生成", "disclaimer": "非真实结果",
                "warnings": [], "next_actions": [], "sections": [
                    {"heading": "visual_answer", "content": "YES" if available else "ABSTAIN",
                     "source_ids": ["image:current"] if available else []},
                    {"heading": "context_value", "content": packet["text:context"]["context_value"],
                     "source_ids": ["text:context"]},
                ],
            })
        return {"raw_text": raw, "finish_reason": "stop", "latency_seconds": 0.1,
                "prompt_tokens": 50, "completion_tokens": 20, "image_seen": image_path is not None,
                "peak_gpu_memory_bytes": 0}

    def close(self):
        pass


class EndToEndTest(unittest.TestCase):
    def test_prepare_generate_resume_score_with_optional_swaps(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            dataset = root / "dataset"
            dataset.mkdir()

            def source(qid, answer="是", question="图中有器官吗？"):
                image = dataset / "imgs" / f"xmlab{qid}" / "source.jpg"
                image.parent.mkdir(parents=True)
                image.write_bytes(png(qid))
                return {"qid": qid, "img_id": qid, "img_name": f"xmlab{qid}/source.jpg",
                        "question": question, "answer": answer,
                        "q_lang": "zh", "answer_type": "CLOSED", "base_type": "vqa"}

            (dataset / "train.json").write_text(json.dumps([source(1)]), encoding="utf-8")
            (dataset / "test.json").write_text(json.dumps([source(2)]), encoding="utf-8")
            rows = [source(3), source(4, "否"), source(5, question="另一个没有匹配对的图像问题？")]
            (dataset / "validation.json").write_text(json.dumps(rows), encoding="utf-8")
            tasks_dir, run_root = root / "tasks", root / "run"
            prepare_slake.prepare(dataset_root=dataset, output_dir=tasks_dir, split="validation")
            tasks, selection = common.load_tasks(tasks_dir)
            self.assertEqual(len(tasks), 8)  # 3 clean + 3 no-image + 2 valid swaps.
            self.assertTrue(all(Path(t["image_path"]).is_absolute() for t in tasks if t["image_path"]))
            models = {}
            for index, name in enumerate(("vlm", *common.TEXT_ARMS)):
                path = root / name
                path.mkdir()
                (path / "config.json").write_text('{}', encoding="utf-8")
                (path / "model.safetensors").write_bytes(bytes([index + 1]) * 2048)
                models[name] = str(path)
            config = {"schema_version": "care_opd.visual_transfer.config.v1", "models": models,
                      "protocol_version": common.PROTOCOL_VERSION, "evaluation_use": "validation",
                      "tasks_dir": str(tasks_dir), "run_root": str(run_root),
                      "generation": dict(evaluate.DEFAULT_GENERATION)}
            with patch.object(evaluate, "environment", return_value={"synthetic_test_only": True}):
                manifest = evaluate.bind_run(config, tasks, selection)
            SyntheticGenerator.calls = 0
            with patch.object(evaluate, "HFGenerator", SyntheticGenerator):
                for _ in range(2):
                    evaluate.run_phase(config, manifest, tasks, "observe")
                    evaluate.run_phase(config, manifest, tasks, "direct")
                    for arm in common.TEXT_ARMS:
                        evaluate.run_phase(config, manifest, tasks, "generate", arm)
            self.assertEqual(SyntheticGenerator.calls, 5 + 8 * 5)
            payload = score_results.score_results(tasks_dir=tasks_dir, run_root=run_root,
                                                  output=root / "summary", bootstrap=20)
            self.assertEqual(payload["status"], "COMPLETE")
            self.assertEqual(payload["protocol_version"], common.PROTOCOL_VERSION)
            self.assertEqual(payload["evaluation_use"], "validation")
            self.assertTrue(payload["revision_disclosure"]["prior_test_outputs_inspected"])
            self.assertEqual(payload["task_count"], 8)
            self.assertEqual(payload["family_count"], 3)
            metrics = {(r["arm"], r["condition"]): r for r in payload["per_arm_condition"]}
            self.assertEqual(metrics[("tropd_full", "clean")]["invalid_output_count"], 3)
            self.assertEqual(metrics[("tropd_full", "clean")]["total_expected"], 3)
            self.assertEqual(metrics[("care_opd", "swapped_image")]["total_expected"], 2)
            self.assertEqual(metrics[("care_opd", "clean")]["answer_correct_count"], 2)
            self.assertEqual(metrics[("care_opd", "no_image")]["strict_contract_success_count"], 3)
            self.assertAlmostEqual(metrics[("care_opd", "clean")]["end_to_end_latency_mean_seconds"], 0.2)


if __name__ == "__main__":
    unittest.main()
