from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from app.agents.nodes.evidence_packet_node import EvidencePacketNode
from app.core.config import get_settings
from app.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from app.llm.providers.ollama_provider import (
    _case_reasoning_contract,
    _hard_constraints,
    _normalize_case_reasoning_output,
    _prompt_evidence_packet,
    _user_payload,
)
from app.llm.schemas import LLMRequest
from app.schemas.agent import DoctorCaseReasoningRequest


CASE = "患者脘腹胀满，食后加重，纳少便溏，舌淡苔白，脉缓弱。"
CASE_SHA = hashlib.sha256(CASE.encode("utf-8")).hexdigest()
SOURCE_ID = f"CASE_INPUT_{CASE_SHA[:16]}"


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):  # noqa: ARG002
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def _payload() -> dict:
    return {
        "message": f"请分析匿名教学病例。病例：{CASE}",
        "protocol": "public_anonymized_tcm_case_reasoning_v1",
        "task_token": "T-BEST30case0001",
        "source_kind": "public_anonymized_teaching_case",
        "case_material_sha256": CASE_SHA,
        "case_material": CASE,
        "inference_mode": "uncertain_teaching_hypothesis",
        "reference_answer_included": False,
        "clinical_use_authorized": False,
    }


def _bound_packet() -> dict:
    state = {
        "role": "doctor",
        "task_type": "doctor_ai_chat",
        "input_text": _payload()["message"],
        "input_payload": {"case_reasoning_context": _payload() | {"authorization": "test"}},
    }
    state["input_payload"]["case_reasoning_context"].pop("message", None)
    evidence, contract = EvidencePacketNode(None)._request_bound_case(state)
    return {
        "structured_context": {"case_reasoning_contract": contract},
        "retrieved_evidence": evidence,
    }


def test_schema_forbids_reference_or_gold_fields() -> None:
    DoctorCaseReasoningRequest.model_validate(_payload())
    with pytest.raises(ValidationError):
        DoctorCaseReasoningRequest.model_validate(
            _payload() | {"reference_answer": "不得进入模型请求"}
        )
    with pytest.raises(ValidationError):
        DoctorCaseReasoningRequest.model_validate(
            _payload() | {"reference_answer_included": True}
        )


def test_complete_case_is_bound_as_one_hash_verified_evidence_item() -> None:
    packet = _bound_packet()
    evidence = packet["retrieved_evidence"]
    contract = packet["structured_context"]["case_reasoning_contract"]
    assert len(evidence) == 1
    assert evidence[0]["content"] == CASE
    assert evidence[0]["source_id"] == SOURCE_ID
    assert hashlib.sha256(evidence[0]["content"].encode("utf-8")).hexdigest() == CASE_SHA
    assert contract["product_adapters_bypassed"] is False
    assert contract["model_reference_answer_access"] is False


def test_case_hash_or_partial_message_fails_closed() -> None:
    context = _payload()
    context.pop("message")
    context["case_material_sha256"] = "0" * 64
    state = {
        "role": "doctor",
        "task_type": "doctor_ai_chat",
        "input_text": f"病例摘要：{CASE[:8]}",
        "input_payload": {"case_reasoning_context": context},
    }
    with pytest.raises(RuntimeError, match="hash failed closed"):
        EvidencePacketNode(None)._request_bound_case(state)

    context["case_material_sha256"] = CASE_SHA
    with pytest.raises(RuntimeError, match="not bound to input_text"):
        EvidencePacketNode(None)._request_bound_case(state)


def test_reasoning_prompt_requires_matching_contract_and_evidence() -> None:
    packet = _bound_packet()
    request = LLMRequest(
        role="doctor",
        task_type="doctor_ai_chat",
        input_text=_payload()["message"],
        evidence_packet=packet,
    )
    assert _case_reasoning_contract(request) is not None
    constraints = _hard_constraints(False, case_reasoning=True)
    assert any("教学判断" in item for item in constraints)
    assert any("禁止固定剂量" in item for item in constraints)
    assert any("缺少舌象、脉象" in item and "不得据此取消" in item for item in constraints)
    assert any("至少写出两条" in item and "事实—推断链" in item for item in constraints)
    assert any("条件表达" in item and "无剂量" in item for item in constraints)

    packet_without_case = {
        **packet,
        "retrieved_evidence": [],
    }
    unsafe_request = request.model_copy(update={"evidence_packet": packet_without_case})
    assert _case_reasoning_contract(unsafe_request) is None


def test_case_reasoning_output_contract_expands_all_six_sections() -> None:
    request = LLMRequest(
        role="doctor",
        task_type="doctor_ai_chat",
        input_text=_payload()["message"],
        evidence_packet=_bound_packet(),
    )
    payload = _user_payload(request)
    sections = payload["output_schema"]["sections"]

    assert [section["heading"] for section in sections] == [
        "病例要点",
        "中医辨病与候选辨证",
        "病因病机及“症状/舌脉—判断”证据链",
        "治则治法",
        "方药思路与加减原则",
        "需要鉴别或进一步复核的信息",
    ]
    assert all(section["source_ids"] == [SOURCE_ID] for section in sections)
    assert any("不得把六部分合并进一个 section" in item for item in payload["hard_constraints"])
    assert any("六个 sections 必须全部引用" in item for item in payload["hard_constraints"])


def test_case_reasoning_prompt_projection_keeps_case_and_drops_repository_noise() -> None:
    packet = _bound_packet()
    packet.update(
        {
            "packet_id": "packet-test",
            "role": "doctor",
            "task_type": "doctor_ai_chat",
            "visibility": "doctor_internal",
            "graph_reasoning_path": [{"nodes": ["偏好", "六节结构"]}],
            "memory_used": [{"format_preference": "case_reasoning_six_sections"}],
            "safety_decision": {"allow_answer": True, "need_human_confirm": True},
            "backend_summary": {
                "knowledge_backend": "local_knowledge",
                "graph_backend": "graphiti",
                "memory_backend": "mem0",
            },
            "tool_calls": [{"private_operational_trace": "x" * 5000}],
        }
    )
    packet["retrieved_evidence"].append(
        {
            "source_id": "SRC_TEST",
            "source_ids": ["SRC_TEST"],
            "title": "知识候选",
            "snippet": "与病例相关的简短知识片段。",
            "retrieved_by": "search_evidence",
            "metadata": {
                "evidence_level": "open_dataset_extracted",
                "functions": ["清热", "解表"],
                "cautions": ["须由医生复核"],
                "keywords": ["无关长关键词"] * 1000,
                "source_records": [{"private_repository_path": "x" * 1000}],
            },
        }
    )

    projected = _prompt_evidence_packet(packet, case_reasoning=True)

    assert projected["request_bound_case_material"]["content"] == CASE
    assert projected["request_bound_case_material"]["source_id"] == SOURCE_ID
    assert projected["retrieved_knowledge"][0]["snippet"] == "与病例相关的简短知识片段。"
    assert projected["retrieved_knowledge"][0]["functions"] == ["清热", "解表"]
    assert "metadata" not in projected["retrieved_knowledge"][0]
    assert "tool_calls" not in projected
    assert "private_repository_path" not in json.dumps(projected, ensure_ascii=False)
    assert projected["graph_reasoning_path"] == packet["graph_reasoning_path"]
    assert projected["memory_used"] == packet["memory_used"]
    assert projected["prompt_projection"] == {
        "policy": "content_neutral_bounded_evidence_projection_v1",
        "full_packet_retained_for_audit": True,
        "retrieved_item_count": 2,
    }
    assert packet["tool_calls"][0]["private_operational_trace"] == "x" * 5000
    assert len(json.dumps(projected, ensure_ascii=False)) * 4 < len(
        json.dumps(packet, ensure_ascii=False)
    )


def test_single_section_with_six_headings_is_recovered_without_new_content() -> None:
    request = LLMRequest(
        role="doctor",
        task_type="doctor_ai_chat",
        input_text=_payload()["message"],
        evidence_packet=_bound_packet(),
    )
    contents = [
        "见脘腹胀满、纳少便溏等病例观察。",
        "候选辨证倾向脾气虚弱，仍属教学推断。",
        "食后加重与舌淡脉缓弱共同支持脾气不足的病机链。",
        "治则治法可考虑健脾益气。",
        "方药仅讨论健脾益气的无剂量思路与随证加减原则。",
        "仍需鉴别兼夹湿邪并补充病程后复核。",
    ]
    headings = [
        "病例要点",
        "中医辨病与候选辨证",
        "病因病机及“症状/舌脉—判断”证据链",
        "治则治法",
        "方药思路与加减原则",
        "需要鉴别或进一步复核的信息",
    ]
    structured = {
        "title": "病例分析",
        "summary": "教学性候选分析。",
        "sections": [
            {
                "heading": "病例分析",
                "content": "\n".join(
                    f"{heading}：{content}"
                    for heading, content in zip(headings, contents, strict=True)
                ),
                "source_ids": [SOURCE_ID],
            }
        ],
        "warnings": [],
        "next_actions": [],
        "disclaimer": "教学用途。",
    }

    recovered = _normalize_case_reasoning_output(request, structured)

    assert [section["heading"] for section in recovered["sections"]] == headings
    assert [section["content"] for section in recovered["sections"]] == contents
    assert all(section["source_ids"] == [SOURCE_ID] for section in recovered["sections"])


def test_incomplete_single_section_is_not_synthetically_expanded() -> None:
    request = LLMRequest(
        role="doctor",
        task_type="doctor_ai_chat",
        input_text=_payload()["message"],
        evidence_packet=_bound_packet(),
    )
    structured = {
        "title": "病例分析",
        "summary": "教学性分析。",
        "sections": [
            {
                "heading": "病例要点",
                "content": "只包含病例要点，没有其余五部分。",
                "source_ids": [SOURCE_ID],
            }
        ],
        "warnings": [],
        "next_actions": [],
        "disclaimer": "教学用途。",
    }

    assert _normalize_case_reasoning_output(request, structured) == structured


def test_openai_compatible_case_generation_applies_section_recovery(monkeypatch) -> None:
    request = LLMRequest(
        role="doctor",
        task_type="doctor_ai_chat",
        input_text=_payload()["message"],
        evidence_packet=_bound_packet(),
    )
    headings = [
        "病例要点",
        "中医辨病与候选辨证",
        "病因病机及“症状/舌脉—判断”证据链",
        "治则治法",
        "方药思路与加减原则",
        "需要鉴别或进一步复核的信息",
    ]
    content = {
        "title": "病例分析",
        "summary": "教学性候选分析。",
        "sections": [
            {
                "heading": "病例分析",
                "content": "\n".join(f"{heading}：测试内容{index}" for index, heading in enumerate(headings)),
                "source_ids": [SOURCE_ID],
            }
        ],
        "warnings": [],
        "next_actions": [],
        "disclaimer": "教学用途。",
    }
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(content, ensure_ascii=False)},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    captured: dict = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured.update(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(response)

    settings = get_settings()
    original = {
        "openai_compatible_base_url": settings.openai_compatible_base_url,
        "openai_compatible_api_key": settings.openai_compatible_api_key,
        "openai_compatible_model": settings.openai_compatible_model,
    }
    settings.openai_compatible_base_url = "http://local.test/v1"
    settings.openai_compatible_api_key = "test-key"
    settings.openai_compatible_model = "shennong-sanyan-care-opd"
    monkeypatch.setattr(
        "app.llm.providers.openai_compatible_provider.urllib.request.urlopen",
        fake_urlopen,
    )
    try:
        result = OpenAICompatibleProvider().generate(request)
    finally:
        for key, value in original.items():
            setattr(settings, key, value)

    assert [section["heading"] for section in result.structured_output["sections"]] == headings
    assert all(section["source_ids"] == [SOURCE_ID] for section in result.structured_output["sections"])
    assert len(captured["messages"][-1]["content"]) > 0
