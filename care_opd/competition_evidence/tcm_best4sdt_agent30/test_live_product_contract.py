#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "execute_three_role_eval.py"
SPEC = importlib.util.spec_from_file_location("execute_three_role_eval_live_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)


def qualified_payload(
    profile: str = "live_product",
) -> tuple[dict, dict, dict, dict]:
    if profile == "live_product":
        knowledge_backend = "lightrag"
        retrieved_by = "lightrag"
    elif profile == "local_product":
        knowledge_backend = "local_knowledge"
        retrieved_by = "search_evidence"
    else:
        raise ValueError(f"unsupported product profile: {profile}")

    task = {"task_id": "BEST30-test", "role": "doctor"}
    llm = {
        "success": True,
        "provider": "openai_compatible",
        "model": "shennong-sanyan-care-opd",
        "metadata": {
            "fallback_used": False,
            "parse_success": True,
            "schema_valid": True,
        },
    }
    packet = {
        "packet_id": "packet-1",
        "trace_id": "trace-1",
        "role": "doctor",
        "visibility": "doctor_internal",
        "graph_node_path": [
            "AuthNode",
            "RouterNode",
            "ConversationContextNode",
            "MemoryRouterNode",
            "MCPToolNode",
        ],
        "backend_summary": {
            "knowledge_backend": knowledge_backend,
            "graph_backend": "graphiti",
            "memory_backend": "mem0",
            "fallback_used": False,
            "context_strategy": "recent_window+mem0+graphiti",
            "context_mode": "topic_continuation",
            "recent_turn_count": 1,
            "available_recent_turn_count": 1,
        },
        "conversation_context": {
            "strategy": "recent_window+mem0+graphiti",
            "scope": {"role": "doctor", "user_id": 7},
            "context_mode": "topic_continuation",
            "turn_count": 1,
            "available_turn_count": 1,
        },
        "retrieved_evidence": [{"source_id": "SRC-1", "retrieved_by": retrieved_by}],
        "memory_used": [
            {
                "source": "mem0",
                "owner_role": "doctor",
                "owner_user_id": 7,
                "content": "先给结论",
            }
        ],
        "graph_reasoning_path": [{"nodes": ["a", "b"], "edges": ["prefers"]}],
        "tool_calls": [
            {
                "tool_name": "get_bound_patient_roster",
                "success": True,
                "metadata": {"backend": "local_business"},
            },
            {
                "tool_name": "search_evidence",
                "success": True,
                "metadata": {
                    "backend": knowledge_backend,
                    "fallback_used": False,
                },
            },
            {
                "tool_name": "get_reasoning_path",
                "success": True,
                "metadata": {"backend": "graphiti", "fallback_used": False},
            },
            {
                "tool_name": "search_role_memory",
                "success": True,
                "metadata": {"backend": "mem0", "fallback_used": False},
            },
        ],
    }
    data = {
        "trace_id": "trace-1",
        "evidence_packet_id": "packet-1",
        "evidence_packet": packet,
        "llm_result": llm,
        "graph_node_path": [
            "AuthNode",
            "RouterNode",
            "ConversationContextNode",
            "MemoryRouterNode",
            "MCPToolNode",
            "EvidencePacketNode",
            "LLMGenerateNode",
            "SafetyPostNode",
            "ResponseNode",
            "TraceNode",
        ],
        "generated_response": {
            "title": "辨证分析",
            "summary": "摘要",
            "sections": [{"heading": "依据", "content": "内容", "source_ids": ["SRC-1"]}],
            "warnings": [],
            "next_actions": [],
            "disclaimer": "教学用途",
        },
    }
    raw = {
        "data": data,
        "_care_eval_auth_private": {
            "user_id": 7,
            "role": "doctor",
            "identity_sha256": "a" * 64,
        },
        "_care_eval_product_setup_private": {
            "profile": profile,
            "knowledge_backend": knowledge_backend,
            "protocol": "isolated_neutral_longitudinal_state_v1",
            "seed_contains_case_material": False,
            "warmup_chat_id_present": True,
            "probes": {
                "knowledge": {
                    "backend": knowledge_backend,
                    "fallback_used": False,
                },
                "graph": {"backend": "graphiti", "fallback_used": False},
                "memory": {"backend": "mem0", "fallback_used": False},
            },
        },
    }
    return task, raw, data, llm


def qualified_case_payload() -> tuple[dict, dict, dict, dict]:
    task, raw, data, llm = qualified_payload("local_product")
    case_material = "患者反复脘腹胀满，食后加重，纳少便溏，舌淡苔白，脉缓弱。"
    case_sha256 = hashlib.sha256(case_material.encode("utf-8")).hexdigest()
    source_id = f"CASE_INPUT_{case_sha256[:16]}"
    task.update(
        {
            "modality": "tcm_open_case_reasoning",
            "source_text_private": case_material,
            "paired_case_material_sha256": case_sha256,
            "family_token": "BEST-F-test",
            "variant": "clean",
            "arm": "role_aware",
            "user_message_private": f"请分析下列匿名教学病例。病例：{case_material}",
        }
    )
    contract = {
        "protocol": "public_anonymized_tcm_case_reasoning_v1",
        "task_token": MODULE._private_control_task_token(task),
        "source_kind": "public_anonymized_teaching_case",
        "case_material_sha256": case_sha256,
        "case_source_id": source_id,
        "inference_mode": "uncertain_teaching_hypothesis",
        "reference_answer_included": False,
        "clinical_use_authorized": False,
        "current_request_bound": True,
        "complete_material_bound": True,
        "model_reference_answer_access": False,
        "product_adapters_bypassed": False,
    }
    case_item = {
        "source_id": source_id,
        "kind": "request_bound_case_material",
        "content": case_material,
        "retrieved_by": "request_evidence_binding",
        "metadata": {**contract, "claim_status": "user_supplied_case_fact"},
    }
    packet = data["evidence_packet"]
    packet["retrieved_evidence"].insert(0, case_item)
    packet["structured_context"] = {"case_reasoning_contract": contract}
    packet["backend_summary"]["case_reasoning_contract"] = contract
    raw["_care_eval_case_request_private"] = {
        "protocol": contract["protocol"],
        "task_token": contract["task_token"],
        "source_kind": contract["source_kind"],
        "case_material_sha256": case_sha256,
        "inference_mode": contract["inference_mode"],
        "reference_answer_included": False,
        "clinical_use_authorized": False,
        "endpoint": "/api/doctor/case-reasoning",
        "complete_case_material_sent": True,
        "reference_or_gold_fields_sent": False,
    }
    headings = [
        "病例要点",
        "中医辨病与候选辨证",
        "病因病机及“症状/舌脉—判断”证据链",
        "治则治法",
        "方药思路与加减原则",
        "需要鉴别或进一步复核的信息",
    ]
    contents = [
        (
            "病例已知脘腹胀满、食后加重、纳少便溏，并见舌淡苔白、脉缓弱，"
            "这些均为本轮病例事实。病程和伴随表现只按当前文字复述，"
            "不补造病例之外的信息。"
        ),
        (
            "首要候选为脾气虚弱证，依据纳少便溏、食后脘腹胀满与舌淡脉弱，"
            "属于教学性倾向而不是确诊。主要鉴别方向包括脾阳虚与湿困脾胃，"
            "仍需结合寒热和口黏等表现进一步区分。"
        ),
        (
            "病例已知的食后脘腹胀满与纳少便溏提示脾失健运、运化乏力。"
            "舌淡苔白与脉缓弱支持气虚偏寒的候选病机；这些证据链只用于教学推断，"
            "并保留兼湿或阳虚的鉴别空间。"
        ),
        (
            "若脾气虚弱候选经复核成立，则治法可考虑健脾益气、和中助运。"
            "如后续出现明显畏寒或水湿表现，再根据新增事实调整温阳或化湿的侧重，"
            "当前不作临床定论。"
        ),
        (
            "若以脾气虚弱为主要候选，方药思路可参考健脾益气、和中化湿的组方原则，"
            "不提供固定剂量。随证加减只说明教学方向，最终仍须由医生复核。"
        ),
        (
            "需要进一步复核症状持续时间、饮食诱因、既往治疗反应及寒热偏向，"
            "并核对舌脉是否稳定。还应鉴别脾阳虚、湿困脾胃等方向，"
            "补充信息可能改变当前候选排序。"
        ),
    ]
    data["generated_response"] = {
        "title": "匿名教学病例辨证分析",
        "summary": "依据本轮病例形成候选辨证与条件性教学分析。",
        "sections": [
            {
                "heading": heading,
                "content": content,
                "source_ids": [source_id],
            }
            for heading, content in zip(headings, contents, strict=True)
        ],
        "warnings": ["仅用于教学推理，仍需医生复核。"],
        "next_actions": ["补充病程和兼症以完成鉴别。"],
        "disclaimer": "不构成真实临床诊疗或处方。",
    }
    return task, raw, data, llm


class LiveProductContractTest(unittest.TestCase):
    def test_product_request_uses_audited_case_endpoint_without_gold(self) -> None:
        task, _, _, _ = qualified_case_payload()
        task["reference_answer_private"] = "GOLD_SENTINEL_MUST_NOT_LEAK"
        args = SimpleNamespace(
            base_url="http://127.0.0.1:8000",
            timeout_seconds=60,
            system_api_profile="local_product",
            private_eval_key_env="CARE_TEST_CASE_KEY",
        )
        captured: dict = {}

        def fake_post(url, payload, headers, timeout):
            captured.update(
                {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
            )
            return {"data": {}}

        with mock.patch.dict("os.environ", {"CARE_TEST_CASE_KEY": "a" * 64}), mock.patch.object(
            MODULE,
            "_provision_task_identity",
            return_value={
                "token": "token",
                "user_id": 7,
                "role": "doctor",
                "identity_sha256": "b" * 64,
            },
        ), mock.patch.object(
            MODULE,
            "_prepare_product_state",
            return_value={"profile": "local_product"},
        ), mock.patch.object(MODULE, "_post_json", side_effect=fake_post):
            raw = MODULE._system_request(task, args, 0)

        self.assertTrue(captured["url"].endswith("/api/doctor/case-reasoning"))
        self.assertEqual(captured["payload"]["case_material"], task["source_text_private"])
        self.assertEqual(
            captured["payload"]["case_material_sha256"],
            task["paired_case_material_sha256"],
        )
        self.assertIs(captured["payload"]["reference_answer_included"], False)
        self.assertNotIn("GOLD_SENTINEL", repr(captured["payload"]))
        self.assertIn("X-Care-Case-Reasoning-Key", captured["headers"])
        self.assertFalse(
            raw["_care_eval_case_request_private"]["reference_or_gold_fields_sent"]
        )

    def test_case_reasoning_contract_and_quality_pass(self) -> None:
        task, raw, data, llm = qualified_case_payload()
        audit = MODULE._validate_local_product_response(
            task, raw, data, data["evidence_packet"], llm
        )
        self.assertTrue(audit["evidence"]["case_material_evidence_verified"])
        quality = audit["case_reasoning_quality_gate"]
        self.assertEqual(quality["status"], "PASS")
        self.assertFalse(quality["content_review_enforced"])
        self.assertFalse(quality["safety_review_enforced"])
        self.assertEqual(
            quality["content_review_policy"],
            "audit_only_expert_judged_no_content_triggered_regeneration",
        )
        self.assertEqual(quality["safety_review_policy"], quality["content_review_policy"])

    def test_outside_packet_citation_is_audited_not_regenerated(self) -> None:
        task, raw, data, llm = qualified_case_payload()
        data["generated_response"]["sections"][0]["source_ids"] = [
            "SOURCE_NOT_IN_PACKET"
        ]

        audit = MODULE._validate_local_product_response(
            task, raw, data, data["evidence_packet"], llm
        )
        quality = audit["case_reasoning_quality_gate"]
        citation = quality["citation_integrity_audit"]

        self.assertEqual(quality["status"], "PASS")
        self.assertIn("source_id_outside_evidence_packet", quality["content_review_flags"])
        self.assertEqual(citation["outside_packet_source_count"], 1)
        self.assertFalse(citation["enforced"])
        self.assertNotIn("SOURCE_NOT_IN_PACKET", json.dumps(citation, sort_keys=True))

    def test_compact_conditional_treatment_and_formula_sections_pass(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        sections = data["generated_response"]["sections"]
        sections[3]["content"] = (
            "若脾气虚弱候选经复核成立，则治法可考虑健脾益气、和中助运，"
            "并随新增寒湿表现调整侧重。"
        )
        sections[4]["content"] = (
            "若该候选成立，方药思路可参考健脾益气、和中化湿的无剂量组方原则，"
            "并根据新增事实随证加减。"
        )

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertEqual(quality["section_clause_counts"][3:5], [1, 1])
        self.assertTrue(quality["all_sections_nonempty"])

    def test_compact_specific_review_section_passes(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        data["generated_response"]["sections"][5]["content"] = (
            "需鉴别脾阳虚与湿困脾胃，并复核症状持续时间、饮食诱因及舌脉是否稳定。"
        )

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertEqual(quality["section_clause_counts"][5], 1)
        self.assertTrue(quality["concrete_review_target_present"])
        self.assertTrue(quality["all_sections_nonempty"])

    def test_open_vocabulary_reverse_review_target_passes(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        data["generated_response"]["sections"][5]["content"] = (
            "系统性红斑狼疮等相似疾病仍需排查，并结合后续面诊检查完善候选判断。"
        )

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertTrue(quality["concrete_review_target_present"])

    def test_generic_review_padding_is_recorded_without_regeneration(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        data["generated_response"]["sections"][5]["content"] = (
            "当前仍有若干不确定内容，建议补充更多相关资料后再由医生进行复核判断。"
        )

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertFalse(quality["content_review_enforced"])
        self.assertIn(
            "concrete_review_target_in_section_6",
            quality["content_review_flags"],
        )

    def test_character_and_clause_guidance_is_advisory_only(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        contents = [
            "患儿见呕吐腹泻、潮热与抽搐。",
            "首要候选为疟后脾虚兼肝风内动证；仅作教学假设。",
            "呕吐腹泻提示脾胃受损；抽搐与角弓反张支持肝风内动。",
            "若该候选成立，则可考虑健脾和胃、息风止痉。",
            "若该候选成立，方药可参考健脾息风的无剂量组方方向并随证加减。",
            "需鉴别慢惊风并复核舌脉、脱水程度和神志变化。",
        ]
        for section, content in zip(
            data["generated_response"]["sections"], contents, strict=True
        ):
            section["content"] = content

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertFalse(quality["length_guidance_pass"])
        self.assertTrue(quality["total_below_character_guidance"])
        self.assertTrue(quality["length_guidance_warnings"])

    def test_safety_signals_are_audited_without_content_triggered_retry(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        data["generated_response"]["sections"][4]["content"] += (
            " 建议使用某方十克煎服，该病例已确诊，无需医生复核。"
        )

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertFalse(quality["safety_review_enforced"])
        self.assertTrue(quality["fixed_dose_detected"])
        self.assertTrue(quality["clinical_finality_detected"])
        self.assertIn("fixed_dose", quality["safety_review_flags"])
        self.assertIn("clinical_finality", quality["safety_review_flags"])

    def test_procedural_placeholder_is_a_selection_neutral_audit_flag(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        data["generated_response"]["sections"][1]["content"] = (
            "候选方向：需结合舌脉、病程与检查进一步鉴别。"
            "目前仅说明不确定性，不直接给出具体候选名称。"
        )

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertFalse(quality["substantive_candidate_present"])
        self.assertIn(
            "positive_candidate_in_section_2", quality["content_review_flags"]
        )

    def test_negated_or_undetermined_candidate_phrases_are_audited(self) -> None:
        invalid_candidates = (
            "候选方向：尚需补资料。",
            "候选方向：后续再定。",
            "候选方向：后再定。",
            "候选方向：不作判断。",
            "候选方向：不予判断。",
            "候选方向：尚不能确定。",
            "候选方向：没有结论。",
            "候选方向：未有结论。",
            "候选方向：补齐资料后再判断。",
            "候选方向：收集资料后再判断。",
            "候选方向：人工复核。",
            "候选方向：医生判断。",
            "候选方向：由专家判断。",
            "初步辨证为不确定。",
            "证属未定。",
            "证属未明。",
            "证属尚不明确。",
            "无法判断是否证属脾虚湿盛证。",
            "暂不考虑为脾虚湿盛证。",
            "是否考虑为脾虚湿盛证？",
            "能否辨证为脾虚湿盛证？",
            "辨证考虑是否为脾虚湿盛证。",
            "候选是否为脾虚湿盛证。",
            "证型是否为脾虚湿盛证。",
            "辨证判断是否为脾虚湿盛证。",
            "候选方向：先补舌脉后再判断。",
            "证属哪一型需补资料。",
            "候选方向：尚待补充。",
            "候选方向：有待明确。",
            "未考虑为脾虚湿盛证。",
            "没有考虑为脾虚湿盛证。",
            "并非证属脾虚湿盛证。",
            "不是候选为脾虚湿盛证。",
            "排除候选为脾虚湿盛证。",
            "不支持证属脾虚湿盛证。",
            "本节讨论辨证属性，无法判断。",
            "难以判断是否证属脾虚湿盛证。",
            "初步辨证为脾虚湿盛证？",
            "候选方向是脾虚湿盛证吗？",
            "候选为无。",
            "候选为待。",
            "候选为X。",
        )
        for candidate_text in invalid_candidates:
            with self.subTest(candidate_text=candidate_text):
                _, _, data, _ = qualified_case_payload()
                source_id = data["evidence_packet"]["structured_context"][
                    "case_reasoning_contract"
                ]["case_source_id"]
                data["generated_response"]["sections"][1]["content"] = (
                    candidate_text + "本节仅记录不确定性。"
                )
                quality = MODULE._validate_case_reasoning_output(data, source_id)
                self.assertEqual(quality["status"], "PASS")
                self.assertIn(
                    "positive_candidate_in_section_2",
                    quality["content_review_flags"],
                )

    def test_natural_positive_candidate_phrases_pass(self) -> None:
        candidate_phrases = (
            "初步辨证为脾虚湿盛证。",
            "证属脾虚湿盛。",
            "辨证考虑脾虚湿盛证。",
            "主要证型：脾虚湿盛证。",
            "拟辨为脾虚湿盛证。",
            "拟辨为痹。",
            "证属虚。",
        )
        for candidate_text in candidate_phrases:
            with self.subTest(candidate_text=candidate_text):
                _, _, data, _ = qualified_case_payload()
                source_id = data["evidence_packet"]["structured_context"][
                    "case_reasoning_contract"
                ]["case_source_id"]
                data["generated_response"]["sections"][1]["content"] = (
                    candidate_text + "该判断仅作为教学候选，并保留其他鉴别方向。"
                )
                quality = MODULE._validate_case_reasoning_output(data, source_id)
                self.assertEqual(quality["status"], "PASS")

    def test_nonlexical_sections_and_string_source_ids_are_audited(self) -> None:
        for invalid_content in ("", "\u200b", "。。。", 1):
            with self.subTest(invalid_content=repr(invalid_content)):
                _, _, data, _ = qualified_case_payload()
                source_id = data["evidence_packet"]["structured_context"][
                    "case_reasoning_contract"
                ]["case_source_id"]
                data["generated_response"]["sections"][0]["content"] = invalid_content
                quality = MODULE._validate_case_reasoning_output(data, source_id)
                self.assertEqual(quality["status"], "PASS")
                self.assertIn(
                    "all_six_sections_nonempty", quality["content_review_flags"]
                )

        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        data["generated_response"]["sections"][0]["source_ids"] = source_id
        quality = MODULE._validate_case_reasoning_output(data, source_id)
        self.assertEqual(quality["status"], "PASS")
        self.assertIn(
            "all_six_case_source_citations", quality["content_review_flags"]
        )

    def test_short_but_semantic_fact_links_and_review_target_pass(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        data["generated_response"]["sections"][2]["content"] = (
            "舌红提示热证。脉数支持里热。"
        )
        data["generated_response"]["sections"][5]["content"] = "复核舌脉。"

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertEqual(quality["fact_to_hypothesis_link_count"], 2)
        self.assertTrue(quality["concrete_review_target_present"])
        self.assertFalse(quality["length_guidance_pass"])

    def test_hollow_conditional_one_liners_are_audited_not_retried(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        sections = data["generated_response"]["sections"]
        sections[3]["content"] = "若候选成立，则可考虑调理。"
        sections[4]["content"] = "若候选成立，方药可参考某方并随证加减。"

        quality = MODULE._validate_case_reasoning_output(data, source_id)

        self.assertEqual(quality["status"], "PASS")
        self.assertIn(
            "conditional_treatment_direction_in_section_4",
            quality["content_review_flags"],
        )
        self.assertIn(
            "conditional_formula_direction_in_section_5",
            quality["content_review_flags"],
        )

    def test_content_audit_keeps_only_content_free_diagnostics(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        sentinel = "SENSITIVE_GENERATED_SENTINEL"
        data["generated_response"]["sections"][0]["content"] = ""
        data["generated_response"]["sections"][1]["content"] += sentinel

        quality = MODULE._validate_case_reasoning_output(data, source_id)
        detail_text = json.dumps(quality, ensure_ascii=False, sort_keys=True)

        self.assertEqual(quality["status"], "PASS")
        self.assertIn("all_six_sections_nonempty", quality["content_review_flags"])
        self.assertEqual(quality["empty_section_numbers"], [1])
        self.assertEqual(quality["sections_below_character_guidance"], [1])
        self.assertEqual(quality["sections_below_clause_guidance"], [1])
        self.assertNotIn(sentinel, detail_text)
        self.assertNotIn(source_id, detail_text)

    def test_missing_structured_generated_output_remains_hard_failure(self) -> None:
        _, _, data, _ = qualified_case_payload()
        source_id = data["evidence_packet"]["structured_context"][
            "case_reasoning_contract"
        ]["case_source_id"]
        data.pop("generated_response")

        with self.assertRaisesRegex(RuntimeError, "no structured response"):
            MODULE._validate_case_reasoning_output(data, source_id)

    def test_case_reasoning_hash_mismatch_fails_closed(self) -> None:
        task, raw, data, llm = qualified_case_payload()
        data["evidence_packet"]["retrieved_evidence"][0]["content"] += "篡改"
        with self.assertRaisesRegex(RuntimeError, "complete frozen case|hash mismatch"):
            MODULE._validate_local_product_response(
                task, raw, data, data["evidence_packet"], llm
            )

    def test_case_reasoning_reference_access_fails_closed(self) -> None:
        task, raw, data, llm = qualified_case_payload()
        data["evidence_packet"]["structured_context"]["case_reasoning_contract"][
            "reference_answer_included"
        ] = True
        with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
            MODULE._validate_local_product_response(
                task, raw, data, data["evidence_packet"], llm
            )

    def test_case_reasoning_records_blanket_refusal_without_regeneration(self) -> None:
        task, raw, data, llm = qualified_case_payload()
        refusal_contents = [
            "只能复述输入，无法形成病例分析。",
            "证据不足，无法给出候选辨证。",
            "现有事实不足以建立病机证据链。",
            "证据不足，无法给出治则治法。",
            "资料不足，不能形成方药方向。",
        ]
        for section, content in zip(
            data["generated_response"]["sections"][:5],
            refusal_contents,
            strict=True,
        ):
            section["content"] = content
        audit = MODULE._validate_local_product_response(
            task, raw, data, data["evidence_packet"], llm
        )
        quality = audit["case_reasoning_quality_gate"]
        self.assertEqual(quality["status"], "PASS")
        self.assertTrue(quality["blanket_refusal_detected"])
        self.assertIn("whole_case_abstention", quality["content_review_flags"])

    def test_runner_separates_care_context_and_enforces_one_writer(self) -> None:
        runner = (
            HERE.parent
            / "local_four_agent_server"
            / "run_four_agent_local_server.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('CARE_MAX_MODEL_LEN="${CARE_MAX_MODEL_LEN:-32768}"', runner)
        self.assertIn(
            'BASELINE_MAX_MODEL_LEN="${BASELINE_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-8192}}"',
            runner,
        )
        self.assertIn("ensure_single_writer", runner)
        self.assertIn("--close --conflict-exit-code 73", runner)
        self.assertIn("preflight_product_task_routing", runner)
        self.assertIn("MEMORY_ROUTE_TASKS_OK", runner)
        self.assertIn(
            'VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"',
            runner,
        )
        self.assertIn(
            'VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER"',
            runner,
        )
        self.assertIn(
            "frozen_reasoning_parser=qwen3 effective_reasoning_parser=omitted "
            "reason=tokenizer_has_no_qwen3_think_delimiters",
            runner,
        )
        self.assertIn(
            'huatuogpt3-local-medical-baseline "" false',
            runner,
        )
        self.assertNotIn(
            "huatuogpt3-local-medical-baseline qwen3 false",
            runner,
        )
        self.assertIn(
            "frozen_trust_remote_code=false effective_trust_remote_code=true "
            "reason=frozen_snapshot_requires_custom_model_code",
            runner,
        )
        self.assertIn(
            'internlm3-local-role-baseline "" true',
            runner,
        )
        self.assertNotIn(
            'internlm3-local-role-baseline "" false',
            runner,
        )
        self.assertIn(
            'glm4-local-role-baseline "" false',
            runner,
        )

    def test_study_runner_uses_reference_free_frozen_tasks_only(self) -> None:
        study_runner = (HERE / "run_tcm_best4sdt_agent30_gpu3.sh").read_text(
            encoding="utf-8"
        )
        builder = (HERE / "build_server_delta_bundle.py").read_text(
            encoding="utf-8"
        )
        execution_validator = (HERE.parent / "validate_local_four_agent_run.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "tcm_best4sdt_agent30_reference_free_fixed_seed20260825",
            study_runner,
        )
        self.assertIn("FROZEN_REFERENCE_FREE_TASKS_OK=30", study_runner)
        self.assertIn("MANIFEST_SHA256", study_runner)
        self.assertIn("OVERLAP_SHA256", study_runner)
        self.assertIn("reference artifact must not be colocated", study_runner)
        self.assertNotIn('require_file "$DATA"', study_runner)
        self.assertNotIn('"$EVAL_PY" "$PREPARE"', study_runner)
        self.assertIn("FROZEN_TASK_ARCHIVE_ROOT", builder)
        self.assertIn("gpu3_fair_blind_review_bundle.v18", builder)
        self.assertIn('PHYSICAL_GPUS="${PHYSICAL_GPUS:-3}"', study_runner)
        self.assertIn("ACTIVE_GPU_PROCESS_CHECK_ACK", study_runner)
        self.assertIn(
            "care_opd.competition.local_four_agent_execution.v2",
            execution_validator,
        )
        self.assertIn(
            'quality.get("content_review_enforced") is not False',
            execution_validator,
        )
        self.assertNotIn(
            'quality.get("concrete_review_target_present") is not True',
            execution_validator,
        )
        self.assertIn(
            "care_opd.competition.tcm_best4sdt_case_reasoning_quality.v4",
            study_runner,
        )
        self.assertIn("row_audit_present", study_runner)
        self.assertNotIn(
            'get("concrete_review_target") or 0) != 30', study_runner
        )
        self.assertIn("freezes the first user-visible", builder)
        self.assertNotIn(
            '"public_eval/tcm_best4sdt/data/TCM_SDT.json"',
            builder,
        )

    def test_safe_llm_failure_detail_does_not_copy_private_reason_text(self) -> None:
        sentinel = "PRIVATE_CASE_SENTINEL_7f3a"
        detail = MODULE._safe_llm_failure_detail(
            {
                "provider": "openai_compatible",
                "model": "shennong-sanyan-care-opd",
                "error_code": "LLM_PROVIDER_FAILED",
                "metadata": {
                    "fallback_reason": (
                        "openai_compatible:http_400:context_length_exceeded:"
                        + sentinel
                    ),
                    "fallback_used": False,
                    "parse_success": True,
                    "schema_valid": True,
                },
            }
        )

        self.assertIn("context_length_exceeded", detail)
        self.assertNotIn(sentinel, detail)

    def test_local_product_uses_explicit_graphiti_contract_without_background_extraction(
        self,
    ) -> None:
        runner = (
            HERE.parent
            / "local_four_agent_server"
            / "run_four_agent_local_server.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ "$SYSTEM_API_PROFILE" == "live_product" ]]', runner)
        self.assertIn("export GRAPHITI_INGEST_CHAT_HISTORY=false", runner)

    def test_product_setup_writes_neutral_mem0_seed_before_retrieval(self) -> None:
        calls: list[tuple[str, dict]] = []

        def fake_mcp_call(**kwargs):
            tool_name = kwargs["tool_name"]
            calls.append((tool_name, kwargs["arguments"]))
            if tool_name == "write_role_memory":
                return {
                    "metadata": {"backend": "mem0", "fallback_used": False},
                    "data": {
                        "memory_id": "memory-1",
                        "memory_used": [{"source": "mem0"}],
                    },
                }
            if tool_name == "search_role_memory":
                return {
                    "metadata": {"backend": "mem0", "fallback_used": False},
                    "data": {"memory_used": [{"source": "mem0"}]},
                }
            if tool_name == "create_case_graph":
                return {
                    "metadata": {
                        "backend": "graphiti",
                        "fallback_used": False,
                        "write_mode": "structured_triplet_v1",
                    },
                    "data": {},
                }
            if tool_name == "get_reasoning_path":
                return {
                    "metadata": {
                        "backend": "graphiti",
                        "fallback_used": False,
                        "retrieval_mode": "typed_group_edge_query_v1",
                    },
                    "data": {"graph_reasoning_path": [{"nodes": ["a", "b"]}]},
                }
            if tool_name == "search_evidence":
                return {
                    "metadata": {"backend": "local_knowledge", "fallback_used": False},
                    "data": {"evidence": [{"source_id": "SRC-1"}]},
                }
            raise AssertionError(tool_name)

        task = {"task_id": "BEST30-test", "role": "doctor"}
        args = SimpleNamespace(base_url="http://127.0.0.1:8000", timeout_seconds=60)
        identity = {"token": "token", "user_id": 7}
        with mock.patch.object(
            MODULE, "_post_json", return_value={"data": {"chat_id": 1}}
        ), mock.patch.object(MODULE, "_mcp_call", side_effect=fake_mcp_call):
            setup = MODULE._prepare_product_state(
                task=task,
                args=args,
                identity=identity,
                profile="local_product",
            )

        tool_names = [name for name, _ in calls]
        self.assertLess(
            tool_names.index("write_role_memory"),
            tool_names.index("search_role_memory"),
        )
        search_arguments = next(
            arguments for name, arguments in calls if name == "search_role_memory"
        )
        self.assertEqual(search_arguments["query"], MODULE.LIVE_PRODUCT_MEMORY_SEEDS["doctor"])
        graph_arguments = next(
            arguments for name, arguments in calls if name == "create_case_graph"
        )
        self.assertEqual(
            graph_arguments["graphiti_write_mode"], "structured_triplet_v1"
        )
        self.assertTrue(setup["probes"]["memory"]["seed_write_verified"])
        self.assertTrue(setup["probes"]["memory"]["seed_memory_id_present"])

    def test_qualified_payload_passes_for_both_product_profiles(self) -> None:
        contracts = {
            "live_product": (
                MODULE._validate_live_product_response,
                "lightrag",
            ),
            "local_product": (
                MODULE._validate_local_product_response,
                "local_knowledge",
            ),
        }
        for profile, (validator, knowledge_backend) in contracts.items():
            with self.subTest(profile=profile):
                task, raw, data, llm = qualified_payload(profile)
                audit = validator(task, raw, data, data["evidence_packet"], llm)
                self.assertEqual(audit["profile"], profile)
                self.assertEqual(
                    audit["orchestration"]["knowledge_backend"],
                    knowledge_backend,
                )
                self.assertEqual(audit["orchestration"]["graph_backend"], "graphiti")

    def test_local_product_accepts_search_evidence_provenance(self) -> None:
        task, raw, data, llm = qualified_payload("local_product")
        audit = MODULE._validate_product_response(
            task,
            raw,
            data,
            data["evidence_packet"],
            llm,
            profile="local_product",
            knowledge_backend="local_knowledge",
        )
        self.assertEqual(audit["profile"], "local_product")
        self.assertEqual(audit["evidence"]["evidence_count"], 1)

    def test_local_product_wrong_knowledge_backend_fails_closed(self) -> None:
        task, raw, data, llm = qualified_payload("local_product")
        broken = copy.deepcopy(data["evidence_packet"])
        broken["backend_summary"]["knowledge_backend"] = "lightrag"
        with self.assertRaisesRegex(RuntimeError, "backend mismatch"):
            MODULE._validate_local_product_response(task, raw, data, broken, llm)

    def test_local_product_wrong_retrieval_provenance_fails_closed(self) -> None:
        task, raw, data, llm = qualified_payload("local_product")
        broken = copy.deepcopy(data["evidence_packet"])
        broken["retrieved_evidence"][0]["retrieved_by"] = "lightrag"
        with self.assertRaisesRegex(RuntimeError, "retrieval provenance"):
            MODULE._validate_local_product_response(task, raw, data, broken, llm)

    def test_local_product_adapter_fallback_fails_closed(self) -> None:
        task, raw, data, llm = qualified_payload("local_product")
        broken = copy.deepcopy(data["evidence_packet"])
        broken["backend_summary"]["fallback_used"] = True
        with self.assertRaisesRegex(RuntimeError, "adapter fallback"):
            MODULE._validate_local_product_response(task, raw, data, broken, llm)

    def test_local_product_empty_retrieval_requires_explicit_degraded_audit(self) -> None:
        task, raw, data, llm = qualified_payload("local_product")
        degraded = copy.deepcopy(data["evidence_packet"])
        degraded["retrieved_evidence"] = []
        degraded["tool_calls"][1]["metadata"]["degraded"] = True
        degraded_data = copy.deepcopy(data)
        degraded_data["evidence_packet"] = degraded
        degraded_data["generated_response"]["sections"][0]["source_ids"] = []
        audit = MODULE._validate_local_product_response(
            task, raw, degraded_data, degraded, llm
        )
        self.assertEqual(audit["evidence"]["evidence_count"], 0)
        self.assertIn("search_evidence", audit["orchestration"]["degraded_tools"])

        unaudited = copy.deepcopy(degraded)
        unaudited["tool_calls"][1]["metadata"].pop("degraded")
        with self.assertRaisesRegex(RuntimeError, "explicit degraded retrieval audit"):
            MODULE._validate_local_product_response(
                task, raw, degraded_data, unaudited, llm
            )

    def test_adapter_fallback_fails_closed(self) -> None:
        task, raw, data, llm = qualified_payload("live_product")
        broken = copy.deepcopy(data["evidence_packet"])
        broken["backend_summary"]["fallback_used"] = True
        with self.assertRaisesRegex(RuntimeError, "adapter fallback"):
            MODULE._validate_live_product_response(task, raw, data, broken, llm)

    def test_locked_control_is_rejected(self) -> None:
        task, raw, data, llm = qualified_payload("live_product")
        broken = copy.deepcopy(data["evidence_packet"])
        broken["backend_summary"]["private_eval_evidence_control"] = {"enabled": True}
        with self.assertRaisesRegex(RuntimeError, "locked evidence control"):
            MODULE._validate_live_product_response(task, raw, data, broken, llm)


if __name__ == "__main__":
    unittest.main()
