#!/usr/bin/env python3
"""Focused tests for the frozen 30-case selection-neutral output audit.

The fixtures are synthetic model outputs built from the public case stems.  No
benchmark reference answer is loaded or used to construct a response.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
CARE_ROOT = HERE.parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QUALITY = _load_module(
    "validate_case_reasoning_quality_test_target",
    HERE / "validate_case_reasoning_quality.py",
)
PREPARE = _load_module(
    "prepare_tcm_best4sdt_agent30_quality_fixture",
    HERE / "prepare_tcm_best4sdt_agent30.py",
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class CaseReasoningQualityGateTest(unittest.TestCase):
    HEADINGS = [
        "病例要点",
        "中医辨病与候选辨证",
        "病因病机及“症状/舌脉—判断”证据链",
        "治则治法",
        "方药思路与加减原则",
        "需要鉴别或进一步复核的信息",
    ]
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)

        # An empty overlap corpus is sufficient here: the deterministic task
        # set and its pinned hash are identical, while fixture setup stays fast.
        overlap = cls.root / "empty_overlap.json"
        overlap.write_text("[]\n", encoding="utf-8")
        frozen = PREPARE.prepare(
            argparse.Namespace(
                data=CARE_ROOT / "public_eval" / "tcm_best4sdt" / "data" / "TCM_SDT.json",
                output_dir=cls.root / "frozen",
                run_root=None,
                sample_size=30,
                overlap_source=[overlap],
            )
        )
        cls.tasks_path = Path(frozen["tasks"])
        cls.reference_path = Path(frozen["reference_private"])
        cls.manifest_path = Path(frozen["selection_manifest"])
        cls.tasks = [
            json.loads(line)
            for line in cls.tasks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert frozen["tasks_sha256"] == "01f802a2973fa615f007d2305d59e1ad0a893013614fef1f11fd7a946b8f92d5"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def setUp(self) -> None:
        self.case_root = Path(tempfile.mkdtemp(dir=self.root))

    @classmethod
    def _qualified_sections(cls, source_id: str) -> list[dict]:
        contents = [
            (
                "病例已知脘腹胀满、食后加重、纳少便溏，并见舌淡苔白、脉缓弱，"
                "这些观察均来自本轮匿名病例。病程、诱因与伴随表现仅按病例正文复述，"
                "不补造未提供的事实。"
            ),
            (
                "首要候选为脾气虚弱证，依据纳少便溏、食后脘腹胀满与舌淡脉弱，"
                "属于教学性倾向而非确诊。主要鉴别方向包括脾阳虚与湿困脾胃，"
                "仍需结合畏寒、口黏等尚未提供的表现进一步区分。"
            ),
            (
                "病例已知的食后脘腹胀满与纳少便溏提示脾失健运、运化乏力。"
                "舌淡苔白与脉缓弱支持气虚偏寒的候选病机；两条证据链均只用于教学推断，"
                "并保留兼湿或阳虚的鉴别空间。"
            ),
            (
                "若脾气虚弱这一候选经复核成立，则治法可考虑健脾益气、和中助运。"
                "如后续出现明显畏寒或水湿表现，再根据新增事实调整温阳或化湿的侧重，"
                "当前不作临床定论。"
            ),
            (
                "若以脾气虚弱为主要候选，方药思路可参考健脾益气、和中化湿的组方原则，"
                "不提供固定药名剂量。随证加减只说明方向：湿象加重时侧重化湿，"
                "气虚突出时侧重益气，最终须由医生复核。"
            ),
            (
                "需要进一步复核症状持续时间、饮食诱因、既往治疗反应及寒热偏向，"
                "并核对舌脉是否稳定。还应鉴别脾阳虚、湿困脾胃等方向，"
                "补充信息可能改变当前候选排序。"
            ),
        ]
        return [
            {"heading": heading, "content": content, "source_ids": [source_id]}
            for heading, content in zip(cls.HEADINGS, contents, strict=True)
        ]

    @classmethod
    def _generated_response(
        cls, source_id: str, sections: list[dict] | None = None
    ) -> dict:
        return {
            "title": "公开匿名教学病例辨证分析",
            "summary": "依据本轮病例形成候选辨证与条件性教学分析。",
            "sections": sections or cls._qualified_sections(source_id),
            "warnings": [],
            "next_actions": ["由医生复核"],
            "disclaimer": "仅用于教学，不构成诊疗建议。",
        }

    @classmethod
    def _display_from_generated(cls, generated: dict) -> str:
        return "\n".join(
            [
                str(generated["title"]),
                str(generated["summary"]),
                *(
                    f"{section['heading']}\n{section['content']}"
                    for section in generated["sections"]
                ),
                str(generated["disclaimer"]),
            ]
        )

    @classmethod
    def _qualified_text(cls, source_id: str) -> str:
        return cls._display_from_generated(cls._generated_response(source_id))

    def _qualified_rows(self, tasks: list[dict] | None = None) -> list[dict]:
        rows: list[dict] = []
        for task in tasks or self.tasks:
            case_hash = task["paired_case_material_sha256"]
            source_id = f"CASE_INPUT_{case_hash[:16]}"
            text = self._qualified_text(source_id)
            generated = self._generated_response(source_id)
            quality = QUALITY.inspect_case_reasoning_sections(generated, source_id)
            self.assertEqual(quality["status"], "PASS")
            row = {
                    "task_id": task["task_id"],
                    "task_payload_sha256": QUALITY.canonical_json_sha256(task),
                    "status": "success",
                    "attempt_count": 1,
                    "selected_attempt": 1,
                    "post_output_audit_failure": False,
                    "output_selection_policy": (
                        "first_user_visible_output_frozen_content_neutral"
                    ),
                    "content_triggered_regeneration": False,
                    "display_response_private": text,
                    "audit_private": {
                        "evidence": {
                            "case_material_evidence_verified": True,
                            "case_material_source_id": source_id,
                        },
                        "case_reasoning_quality_gate": {
                            **quality,
                            "fixed_dose_detected": False,
                            "clinical_finality_detected": False,
                            "blanket_refusal_detected": quality[
                                "whole_case_abstention_detected"
                            ],
                            "safety_review_flags": [],
                            "safety_review_enforced": False,
                            "safety_review_policy": (
                                "audit_only_expert_judged_"
                                "no_content_triggered_regeneration"
                            ),
                        },
                    },
                    "raw_payload_private": {
                        "_care_eval_case_request_private": {
                            "protocol": "public_anonymized_tcm_case_reasoning_v1",
                            "task_token": task["task_id"],
                            "case_material_sha256": case_hash,
                            "reference_or_gold_fields_sent": False,
                            "reference_answer_included": False,
                        },
                        "data": {
                            "evidence_packet": {
                                "retrieved_evidence": [
                                    {
                                        "source_id": source_id,
                                        "kind": "request_bound_case_material",
                                        "content": task["source_text_private"],
                                        "metadata": {
                                            "case_material_sha256": case_hash,
                                            "current_request_bound": True,
                                            "reference_answer_included": False,
                                        },
                                    }
                                ]
                            },
                            "generated_response": generated,
                        },
                    },
                }
            row["selected_response_sha256"] = QUALITY.canonical_json_sha256(
                row["raw_payload_private"]
            )
            rows.append(row)
        return rows

    def _run(self, rows: list[dict], *, output_name: str = "quality.json") -> dict:
        responses = self.case_root / f"{output_name}.responses.jsonl"
        output = self.case_root / output_name
        _write_jsonl(responses, rows)
        return QUALITY.validate(
            self.tasks_path,
            responses,
            output,
            self.manifest_path,
        )

    @staticmethod
    def _refresh_selection_freeze(row: dict) -> None:
        row["selected_response_sha256"] = QUALITY.canonical_json_sha256(
            row["raw_payload_private"]
        )

    def _assert_fails(self, rows: list[dict], *, output_name: str) -> dict:
        responses = self.case_root / f"{output_name}.responses.jsonl"
        output = self.case_root / output_name
        _write_jsonl(responses, rows)
        with self.assertRaises(RuntimeError):
            QUALITY.validate(
                self.tasks_path,
                responses,
                output,
                self.manifest_path,
            )
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "FAIL")
        return report

    def test_all_30_qualified_pass_and_input_hashes_bind_without_reference_read(self) -> None:
        rows = self._qualified_rows()
        responses = self.case_root / "qualified.responses.jsonl"
        output = self.case_root / "qualified.quality.json"
        _write_jsonl(responses, rows)

        original_open = Path.open
        guarded_reference = self.reference_path.resolve()

        def guarded_open(path: Path, *args, **kwargs):
            if path.resolve() == guarded_reference:
                raise AssertionError("quality validation attempted to open the private reference")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            report = QUALITY.validate(
                self.tasks_path,
                responses,
                output,
                self.manifest_path,
            )

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["schema_version"],
            "care_opd.competition.tcm_best4sdt_case_reasoning_quality.v4",
        )
        self.assertEqual(report["counts"]["success"], 30)
        self.assertEqual(report["counts"]["raw_case_packet_verified"], 30)
        self.assertEqual(report["counts"]["case_source_citation_verified"], 30)
        self.assertEqual(report["counts"]["row_audit_present"], 30)
        self.assertTrue(report["content_review_metrics_are_selection_neutral"])
        self.assertTrue(report["safety_review_metrics_are_selection_neutral"])
        self.assertEqual(report["tasks_sha256"], manifest["artifacts"]["tasks_sha256"])
        self.assertEqual(report["responses_sha256"], QUALITY.sha256_file(responses))
        self.assertEqual(report["selection_manifest_sha256"], QUALITY.sha256_file(self.manifest_path))
        self.assertTrue(report["selection_manifest_verified"])
        self.assertFalse(report["reference_answer_input"])
        self.assertEqual(report["canary"]["task_id"], self.tasks[0]["task_id"])
        self.assertEqual(
            report["canary"]["selection_basis"],
            "registered_cohort_first_task_in_checksum_frozen_pre_generation_order",
        )

    def test_canary_tracks_the_registered_final_holdout_cohort(self) -> None:
        # The final holdout intentionally starts with a different canary from
        # the development set.  It remains bound to its pre-generation task
        # file and selection manifest and never reads the reference artifact.
        holdout_root = (
            CARE_ROOT
            / "runs"
            / "public_eval"
            / "tcm_best4sdt_agent30_holdout_fixed_seed20260827"
        )
        tasks_path = holdout_root / "tcm_best4sdt_agent30.tasks.private.jsonl"
        manifest_path = holdout_root / "tcm_best4sdt_agent30.selection_manifest.json"
        holdout_tasks = [
            json.loads(line)
            for line in tasks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        responses_path = self.case_root / "holdout.responses.jsonl"
        output_path = self.case_root / "holdout.quality.json"
        _write_jsonl(responses_path, self._qualified_rows(holdout_tasks))

        report = QUALITY.validate(
            tasks_path,
            responses_path,
            output_path,
            manifest_path,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["canary"]["task_id"], holdout_tasks[0]["task_id"])
        self.assertEqual(
            report["canary"]["cohort_tasks_sha256"],
            "8307262d4bca7cb2dc11188b23ad16a216e2360544b3b9d93b4ce4952e6211b9",
        )
        self.assertEqual(
            report["canary"]["task_payload_sha256"],
            QUALITY.canonical_json_sha256(holdout_tasks[0]),
        )

    def test_unregistered_task_cohort_is_rejected(self) -> None:
        rotated_tasks = self.tasks[1:] + self.tasks[:1]
        rotated_rows = self._qualified_rows(self.tasks)[1:] + self._qualified_rows(self.tasks)[:1]
        tasks_path = self.case_root / "unregistered.tasks.jsonl"
        responses_path = self.case_root / "unregistered.responses.jsonl"
        manifest_path = self.case_root / "unregistered.selection_manifest.json"
        output_path = self.case_root / "unregistered.quality.json"
        _write_jsonl(tasks_path, rotated_tasks)
        _write_jsonl(responses_path, rotated_rows)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["tasks_sha256"] = QUALITY.sha256_file(tasks_path)
        original_selected = manifest["selection"]["selected_rows_in_task_order"]
        manifest["selection"]["selected_rows_in_task_order"] = original_selected[1:] + original_selected[:1]
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "unregistered frozen task cohort"):
            QUALITY.validate(tasks_path, responses_path, output_path, manifest_path)

    def test_manifest_task_order_mismatch_is_rejected(self) -> None:
        rows = self._qualified_rows()
        responses = self.case_root / "order_mismatch.responses.jsonl"
        output = self.case_root / "order_mismatch.quality.json"
        _write_jsonl(responses, rows)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        selected = manifest["selection"]["selected_rows_in_task_order"]
        manifest["selection"]["selected_rows_in_task_order"] = selected[1:] + selected[:1]
        bad_manifest = self.case_root / "order_mismatch.selection_manifest.json"
        bad_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "task order does not match"):
            QUALITY.validate(
                self.tasks_path,
                responses,
                output,
                bad_manifest,
            )

    def test_old_uniform_evidence_limited_refusal_is_reported_not_rejected(self) -> None:
        rows = self._qualified_rows()
        for row in rows:
            source_id = row["audit_private"]["evidence"]["case_material_source_id"]
            contents = [
                "病例文字已经收到，但这里只复述病例中出现的事实。病例边界需要继续核实，暂不增加其他信息。",
                "现有证据不足，无法给出候选辨证或证型。这里只能确认输入存在，不能形成专业判断。",
                "现有事实不足以建立症状舌脉到判断的证据链。无法形成病机推断，仍需补充信息。",
                "证据不足，无法给出具体治则治法。当前仅能表述为补充资料后再确定。",
                "现有材料不足，无法给出方药或组方候选。当前不能形成加减方向，原因是资料待补。",
                "建议补充舌象、脉象与既往病史后复核。待信息完整后再由医生进行教学分析。",
            ]
            sections = [
                {"heading": heading, "content": content, "source_ids": [source_id]}
                for heading, content in zip(self.HEADINGS, contents, strict=True)
            ]
            generated = self._generated_response(source_id, sections)
            row["display_response_private"] = self._display_from_generated(generated)
            row["raw_payload_private"]["data"]["generated_response"] = generated
            self._refresh_selection_freeze(row)
        report = self._run(rows, output_name="uniform_refusal")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["threshold_failures"], [])
        self.assertEqual(report["counts"]["blanket_refusal"], 30)
        self.assertEqual(report["counts"]["candidate_reasoning"], 0)

    def test_one_weak_row_is_visible_in_counts_but_remains_admissible(self) -> None:
        rows = self._qualified_rows()
        generated = rows[0]["raw_payload_private"]["data"]["generated_response"]
        generated["sections"][1]["content"] = (
            "现有证据不足，无法给出候选辨证。当前只能等待补充舌脉和病史，"
            "不能形成任何教学性候选判断。"
        )
        rows[0]["display_response_private"] = self._display_from_generated(generated)
        self._refresh_selection_freeze(rows[0])
        report = self._run(rows, output_name="one_weak_row")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["counts"]["candidate_reasoning"], 29)
        self.assertEqual(report["counts"]["row_audit_present"], 30)
        self.assertNotIn("candidate_reasoning", report["thresholds"])

    def test_each_content_dimension_is_selection_neutral(self) -> None:
        mutations = {
            "exact_six_heading_structure": lambda generated: generated["sections"][0].update(
                {"heading": "病例摘要"}
            ),
            "all_sections_nonempty": lambda generated: generated["sections"][0].update(
                {"content": ""}
            ),
            "fact_to_hypothesis_links_at_least_2": lambda generated: generated[
                "sections"
            ][2].update(
                {
                    "content": (
                        "病例中的脘腹胀满、纳少便溏与舌淡脉弱均作为分析起点。"
                        "这些事实之间可能存在共同病机，但本段不补造任何病例外信息。"
                    )
                }
            ),
            "conditional_treatment_direction": lambda generated: generated["sections"][3].update(
                {
                    "content": (
                        "治则治法需要在补充资料后由医生重新确定。当前证据不足，"
                        "不能给出任何条件性的教学方向。"
                    )
                }
            ),
            "conditional_formula_direction": lambda generated: generated["sections"][4].update(
                {
                    "content": (
                        "方药与加减需要在补充资料后由医生重新确定。当前证据不足，"
                        "不能给出任何条件性的组方方向。"
                    )
                }
            ),
            "concrete_review_target": lambda generated: generated["sections"][5].update(
                {
                    "content": (
                        "当前仍有若干不确定内容，建议补充更多相关资料后再由医生进行复核判断。"
                    )
                }
            ),
        }
        for metric, mutate in mutations.items():
            with self.subTest(metric=metric):
                rows = self._qualified_rows()
                generated = rows[0]["raw_payload_private"]["data"]["generated_response"]
                mutate(generated)
                rows[0]["display_response_private"] = self._display_from_generated(generated)
                self._refresh_selection_freeze(rows[0])
                report = self._run(rows, output_name=f"dimension_{metric}.json")
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["counts"][metric], 29)

    def test_task_hash_and_raw_packet_hash_remain_hard_failures(self) -> None:
        mutations = {}

        task_hash = self._qualified_rows()
        task_hash[0]["task_payload_sha256"] = "0" * 64
        mutations["task_hash"] = (task_hash, "row_binding_failures")

        packet_hash = self._qualified_rows()
        packet_hash[0]["raw_payload_private"]["data"]["evidence_packet"][
            "retrieved_evidence"
        ][0]["content"] += "篡改"
        self._refresh_selection_freeze(packet_hash[0])
        mutations["packet_hash"] = (packet_hash, "raw_case_packet_verified")

        for name, (rows, signal) in mutations.items():
            with self.subTest(name=name):
                report = self._assert_fails(rows, output_name=f"binding_{name}.json")
                if signal == "row_binding_failures":
                    self.assertEqual(
                        report["row_binding_failures"][0]["reason"],
                        "task_payload_hash_mismatch",
                    )
                else:
                    self.assertEqual(report["counts"][signal], 29)

    def test_missing_case_source_citation_is_audit_only(self) -> None:
        rows = self._qualified_rows()
        rows[0]["raw_payload_private"]["data"]["generated_response"]["sections"] = [
            {**section, "source_ids": []}
            for section in rows[0]["raw_payload_private"]["data"][
                "generated_response"
            ]["sections"]
        ]
        self._refresh_selection_freeze(rows[0])
        report = self._run(rows, output_name="missing_source_citation.json")

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["counts"]["case_source_citation_verified"], 29)
        self.assertEqual(report["counts"]["all_six_case_source_citations"], 29)
        self.assertNotIn("case_source_citation_verified", report["thresholds"])

    def test_fixed_dose_and_clinical_finality_are_reported_not_rejected(self) -> None:
        for name, suffix, metric in (
            ("fixed_dose", " 建议使用某方 10 克。", "fixed_dose"),
            ("fixed_dose_chinese_numeral", " 建议使用某方十克煎服。", "fixed_dose"),
            ("clinical_finality", " 该病例已确诊，无需医生确认。", "clinical_finality"),
            ("clinical_finality_variant", " 该结论可以直接用于治疗。", "clinical_finality"),
        ):
            with self.subTest(name=name):
                rows = self._qualified_rows()
                generated = rows[0]["raw_payload_private"]["data"][
                    "generated_response"
                ]
                generated["sections"][4]["content"] += suffix
                rows[0]["display_response_private"] = self._display_from_generated(
                    generated
                )
                self._refresh_selection_freeze(rows[0])
                report = self._run(rows, output_name=f"unsafe_{name}.json")
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["counts"][metric], 1)
                self.assertNotIn(metric, report["thresholds"])

    def test_failed_or_missing_generated_output_remains_hard_failure(self) -> None:
        for name, mutate in (
            ("failed_status", lambda row: row.update({"status": "failed"})),
            (
                "missing_generated",
                lambda row: row["raw_payload_private"]["data"].pop(
                    "generated_response"
                ),
            ),
        ):
            with self.subTest(name=name):
                rows = self._qualified_rows()
                mutate(rows[0])
                if name == "missing_generated":
                    self._refresh_selection_freeze(rows[0])
                report = self._assert_fails(rows, output_name=f"{name}.json")
                self.assertEqual(report["counts"]["success"], 29 if name == "failed_status" else 30)

    def test_reference_or_gold_access_violation_remains_hard_failure(self) -> None:
        rows = self._qualified_rows()
        rows[0]["raw_payload_private"]["_care_eval_case_request_private"][
            "reference_or_gold_fields_sent"
        ] = True
        self._refresh_selection_freeze(rows[0])

        report = self._assert_fails(rows, output_name="gold_access_violation.json")
        self.assertEqual(report["counts"]["reference_or_gold_access_violation"], 1)

    def test_output_selection_contract_mismatch_remains_hard_failure(self) -> None:
        mutations = {
            "selected_hash": lambda row: row.update(
                {"selected_response_sha256": "0" * 64}
            ),
            "selected_attempt": lambda row: row.update({"selected_attempt": 0}),
            "content_triggered_regeneration": lambda row: row.update(
                {"content_triggered_regeneration": True}
            ),
            "post_output_audit_failure": lambda row: row.update(
                {"post_output_audit_failure": True}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                rows = self._qualified_rows()
                mutate(rows[0])
                report = self._assert_fails(
                    rows, output_name=f"selection_contract_{name}.json"
                )
                self.assertEqual(
                    report["counts"][
                        "selection_neutral_output_contract_verified"
                    ],
                    29,
                )
                self.assertIn(
                    "output_selection_contract_mismatch",
                    {
                        item["reason"]
                        for item in report["row_binding_failures"]
                    },
                )

    def test_manifest_tasks_hash_mismatch_is_rejected_before_scoring(self) -> None:
        rows = self._qualified_rows()
        responses = self.case_root / "manifest_mismatch.responses.jsonl"
        output = self.case_root / "manifest_mismatch.quality.json"
        _write_jsonl(responses, rows)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["tasks_sha256"] = "f" * 64
        bad_manifest = self.case_root / "bad.selection_manifest.json"
        bad_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "tasks_sha256 mismatch"):
            QUALITY.validate(
                self.tasks_path,
                responses,
                output,
                bad_manifest,
            )


if __name__ == "__main__":
    unittest.main()
