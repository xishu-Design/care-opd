#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
CARE_ROOT = HERE.parents[1]
SCRIPT = HERE / "prepare_tcm_best4sdt_agent30.py"
SPEC = importlib.util.spec_from_file_location("prepare_tcm_best4sdt_agent30", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TcmBestAgent30Test(unittest.TestCase):
    def test_frozen_task_contract(self) -> None:
        data = CARE_ROOT / "public_eval" / "tcm_best4sdt" / "data" / "TCM_SDT.json"
        overlap = (
            CARE_ROOT.parent
            / "qzbc_source"
            / "qzbc-main"
            / "backend"
            / "app"
            / "content"
            / "data"
            / "learning_cases.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frozen"
            args = argparse.Namespace(
                data=data,
                output_dir=output,
                run_root=None,
                sample_size=30,
                overlap_source=[overlap],
            )
            first = MODULE.prepare(args)
            second = MODULE.prepare(args)
            self.assertEqual(first["tasks_sha256"], second["tasks_sha256"])
            self.assertEqual(first["tasks_sha256"], "01f802a2973fa615f007d2305d59e1ad0a893013614fef1f11fd7a946b8f92d5")

            task_path = Path(first["tasks"])
            tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(tasks), 30)
            self.assertEqual(len({row["task_id"] for row in tasks}), 30)
            self.assertEqual(len({row["benchmark_id"] for row in tasks}), 30)
            for row in tasks:
                self.assertEqual(row["role"], "doctor")
                self.assertEqual(row["variant"], "clean")
                self.assertEqual(row["arm"], "role_aware")
                self.assertLessEqual(len(row["source_text_private"]), 2000)
                self.assertLessEqual(len(row["user_message_private"]), 2000)
                self.assertEqual(
                    row["shared_payload_sha256"],
                    hashlib.sha256(row["user_message_private"].encode("utf-8")).hexdigest(),
                )
                forbidden = {"output", "reference_output", "answer", "中医疾病诊断"}
                self.assertFalse(forbidden & set(row))

            reference_rows = [
                json.loads(line)
                for line in Path(first["reference_private"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {row["task_id"] for row in tasks},
                {row["task_id"] for row in reference_rows},
            )
            manifest = json.loads(
                Path(first["selection_manifest"]).read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["performance_based_selection"])
            self.assertTrue(manifest["task_protocol"]["reference_answers_excluded_from_task_file"])
            self.assertEqual(manifest["selection"]["sample_count"], 30)

    def test_heldout_selection_is_deterministic_and_disjoint(self) -> None:
        data = CARE_ROOT / "public_eval" / "tcm_best4sdt" / "data" / "TCM_SDT.json"
        overlap = (
            CARE_ROOT.parent
            / "qzbc_source"
            / "qzbc-main"
            / "backend"
            / "app"
            / "content"
            / "data"
            / "learning_cases.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = MODULE.prepare(
                argparse.Namespace(
                    data=data,
                    output_dir=root / "development",
                    run_root=None,
                    sample_size=30,
                    overlap_source=[overlap],
                )
            )
            heldout_args = argparse.Namespace(
                data=data,
                output_dir=root / "heldout",
                run_root=None,
                sample_size=30,
                overlap_source=[overlap],
                selection_seed="20260827-final-holdout",
                sample_label="TCM-BEST4SDT 固定30例最终留出专家盲审轨",
                exclude_selection_manifest=Path(development["selection_manifest"]),
            )
            first = MODULE.prepare(heldout_args)
            second = MODULE.prepare(heldout_args)
            self.assertEqual(first["tasks_sha256"], second["tasks_sha256"])

            dev_manifest = json.loads(
                Path(development["selection_manifest"]).read_text(encoding="utf-8")
            )
            heldout_manifest = json.loads(
                Path(first["selection_manifest"]).read_text(encoding="utf-8")
            )
            dev_rows = dev_manifest["selection"]["selected_rows_in_task_order"]
            heldout_rows = heldout_manifest["selection"]["selected_rows_in_task_order"]
            self.assertTrue(
                {row["benchmark_id"] for row in dev_rows}.isdisjoint(
                    {row["benchmark_id"] for row in heldout_rows}
                )
            )
            self.assertTrue(
                {row["diagnosis_sha256"] for row in dev_rows}.isdisjoint(
                    {row["diagnosis_sha256"] for row in heldout_rows}
                )
            )
            exclusion = heldout_manifest["selection"][
                "prior_frozen_selection_exclusion"
            ]
            self.assertEqual(exclusion["excluded_benchmark_id_count"], 30)
            self.assertEqual(exclusion["excluded_normalized_diagnosis_count"], 30)

            tasks = [
                json.loads(line)
                for line in Path(first["tasks"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(tasks), 30)
            for row in tasks:
                forbidden = {"output", "reference_output", "answer", "中医疾病诊断"}
                self.assertFalse(forbidden & set(row))


if __name__ == "__main__":
    unittest.main()
