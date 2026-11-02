#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
CARE_ROOT = HERE.parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare_module = load(HERE / "prepare_tcmeval_pa_agent30.py", "agent30_prepare_tested")
score_module = load(HERE / "score_tcmeval_pa_agent30.py", "agent30_score_tested")


class Agent30Test(unittest.TestCase):
    def test_freeze_and_score_round_trip(self) -> None:
        data = CARE_ROOT / "public_eval" / "tcmeval_pa" / "data" / "TCMEval-PA.xlsx"
        evaluator = score_module._load_evaluator()
        common = score_module._load_common()
        questions = {q.id: q for q in evaluator.load_questions(data)}
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task_root = root / "tasks"
            run_root = root / "runs"
            result = prepare_module.prepare(
                argparse.Namespace(data=data, output_dir=task_root, run_root=run_root)
            )
            self.assertEqual(result["task_count"], 30)
            tasks_path = task_root / "tcmeval_pa_agent30.tasks.jsonl"
            manifest_path = task_root / "tcmeval_pa_agent30.selection_manifest.json"
            tasks = common.read_jsonl(tasks_path)
            self.assertEqual(len(tasks), 30)
            self.assertEqual(sum(t["benchmark_question_type"] == "multiple" for t in tasks), 3)
            self.assertTrue(all("gold" not in json.dumps(t).lower() for t in tasks))

            for system_id in score_module.DEFAULT_SYSTEMS:
                rows = []
                for task in tasks:
                    gold = "".join(questions[str(task["benchmark_id"])].gold_options)
                    rows.append(
                        {
                            "task_id": task["task_id"],
                            "task_payload_sha256": common.canonical_json_sha256(task),
                            "status": "success",
                            "display_response_private": f"简要依据。\n最终选项：{gold}",
                            "latency_seconds": 1.0,
                        }
                    )
                target = run_root / system_id / "responses.private.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                common.write_jsonl(target, rows)

            output = root / "scored"
            scored = score_module.score(
                argparse.Namespace(
                    data=data,
                    tasks=tasks_path,
                    selection_manifest=manifest_path,
                    run_root=run_root,
                    response=[],
                    output_dir=output,
                )
            )
            self.assertEqual(scored["system_count"], 4)
            summary = json.loads((output / "summary.public.json").read_text(encoding="utf-8"))
            self.assertTrue(all(row["correct"] == 30 for row in summary["systems"]))
            self.assertTrue(all(row["exact_mcnemar_p_two_sided"] == 1.0 for row in summary["paired_vs_primary"]))


if __name__ == "__main__":
    unittest.main()
