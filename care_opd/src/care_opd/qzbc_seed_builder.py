"""Build formal qzbc-native request/response seeds for base-model post-training.

The generated rows live exactly at qzbc's model boundary: ``request`` follows
``LLMRequest`` and ``target`` follows ``GeneratedResponse``.  No CARE-OPD
action schema or deploy-time agent is introduced.  Business/case context is
controlled and synthetic; domain facts come from qzbc's reviewed,
provenance-bearing assets.  Every target is limited to its Evidence Packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .qzbc_assets import (
    QZBCGroundingAssets,
    load_qzbc_grounding_assets,
    partition_qzbc_grounding_assets,
)
from .qzbc_contract import (
    OUTPUT_FIELDS,
    ROLE_TASKS,
    canonical_response,
    extract_source_ids,
    forbidden_tokens,
    load_qzbc_prompts,
    parse_generated_response,
    qzbc_contract_manifest,
)


TASK_ROLE: Mapping[str, str] = {
    task: role for role, tasks in ROLE_TASKS.items() for task in tasks
}
TASKS: Tuple[str, ...] = (
    "doctor_review_text",
    "doctor_ai_chat",
    "patient_education",
    "patient_diet_guidance",
    "student_analysis",
    "student_quiz_explanation",
    "student_ai_chat",
    "quiz_generation",
)
TASK_CODES: Mapping[str, str] = {
    "doctor_review_text": "DREV",
    "doctor_ai_chat": "DCHAT",
    "patient_education": "PEDU",
    "patient_diet_guidance": "PDIET",
    "student_analysis": "SAN",
    "student_quiz_explanation": "SQE",
    "student_ai_chat": "SCHAT",
    "quiz_generation": "QGEN",
}

BACKGROUNDS: Tuple[str, ...] = (
    "随访记录中有一项状态字段需要核对",
    "审方训练记录中需要确认来源可追溯性",
    "复核记录中出现一项尚未闭环的信息",
    "教学样本要求区分业务记录与模型记忆",
    "当前资料只允许完成证据边界说明",
)
FOCI: Tuple[str, ...] = (
    "证据是否来自当前记录",
    "状态字段是否已由人工确认",
    "时间顺序是否与本轮任务一致",
    "缺失字段是否被如实标注",
    "结论是否超出可追溯资料",
)
RESEARCH_TERMS: Tuple[str, ...] = (
    "证据充分性",
    "来源可追溯性",
    "时间一致性",
    "人工复核边界",
    "缺失信息标注",
)
SYNTHETIC_FOODS: Tuple[str, ...] = (
    "样本食物甲",
    "样本饮品乙",
    "样本食材丙",
)
INPUT_FRAMES: Mapping[str, Tuple[str, ...]] = {
    "doctor_review_text": (
        "请根据本轮 Evidence Packet 生成医生审方说明，并标明复核边界。",
        "汇总当前风险记录和可追溯证据，不要补写资料外结论。",
        "请说明这份受控研究记录为什么仍需人工复核。",
    ),
    "doctor_ai_chat": (
        "请根据现有资料说明当前证据和专业判断边界。",
        "这条记录的复核重点是什么？只回答本轮资料已有内容。",
        "请从当前 Evidence Packet 汇总可确认信息。",
    ),
    "patient_education": (
        "请把这条资料解释成患者能理解的中医药科普。",
        "这份记录能确认什么、不能确认什么？",
        "请用通俗语言说明当前资料，不要针对个人作判断。",
    ),
    "patient_diet_guidance": (
        "请通俗解释医生已经确认的饮食备注，并说明资料边界。",
        "这条饮食备注应怎样理解？不要添加新的食忌结论。",
        "请仅转述本轮处方页已有的饮食说明。",
    ),
    "student_analysis": (
        "请根据脱敏 Evidence Packet 生成标准教学解析。",
        "总结本案例的证据使用风险点和复盘步骤。",
        "请评价学生拟方理由是否覆盖当前资料。",
    ),
    "student_quiz_explanation": (
        "请讲解本题标准答案，并完整写出正确选项。",
        "依据题干、选项和标准答案生成教学讲解。",
        "请解释为什么只能选择资料支持的答案。",
    ),
    "student_ai_chat": (
        "怎样区分 Evidence Packet 中的证据与模型记忆？",
        "请用教学结构解释本轮资料的可追溯性。",
        "怎样复习证据边界，避免补造病例事实？",
    ),
    "quiz_generation": (
        "请根据脱敏教学资料生成审方练习题。",
        "生成一组考查证据边界的风险判断题。",
        "请用本轮 Evidence Packet 生成复习题。",
    ),
}

RESEARCH_DISCLAIMER = (
    "本输出仅复述本轮 Evidence Packet，供受控研究与教学验证使用，不构成临床建议。"
)


def _choice(rng: random.Random, values: Sequence[str]) -> str:
    return values[rng.randrange(len(values))]


def _source(
    source_id: str,
    statement: str,
    *,
    kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    source_metadata = {
        "origin": "controlled_synthetic_research",
        "review_status": "verified_for_training_fixture",
        "research_only": True,
        "contains_real_patient_data": False,
    }
    if metadata:
        source_metadata.update(dict(metadata))
    resolved_title = title
    if not resolved_title:
        resolved_title = (
            f"qzbc 已审校{kind}"
            if source_metadata.get("origin") == "qzbc_curated_asset"
            else f"受控合成{kind}"
        )
    return {
        "source_id": source_id,
        "title": resolved_title,
        "summary": statement,
        "snippet": statement,
        "metadata": source_metadata,
    }


def _asset_metadata(
    asset_type: str,
    asset: Mapping[str, Any],
    *,
    license_value: str = "",
) -> Dict[str, Any]:
    return {
        "origin": "qzbc_curated_asset",
        "asset_type": asset_type,
        "asset_id": str(
            asset.get("entry_id")
            or asset.get("constraint_id")
            or asset.get("case_id")
            or ""
        ),
        "evidence_level": str(asset.get("evidence_level") or ""),
        "review_status": str(asset.get("review_status") or ""),
        "license": license_value
        or str(asset.get("license") or asset.get("license_note") or ""),
        "source_record_id": str(asset.get("source_record_id") or ""),
        "research_only": True,
        "contains_real_patient_data": False,
    }


def _conversation_context(index: int, case_id: str) -> Dict[str, Any]:
    mode = index % 4
    if mode == 0:
        turns: List[Dict[str, Any]] = []
        follow_up = False
        include_answers = False
    elif mode == 1:
        turns = [
            {
                "question": "上一轮使用了什么依据？",
                "answer": "上一轮只整理了当时 Evidence Packet 中的受控记录。",
                "answer_trusted": True,
            }
        ]
        follow_up = True
        include_answers = True
    elif mode == 2:
        turns = [
            {
                "question": "能否直接使用模型记忆补全缺失字段？",
                "answer": "这是一条未核验历史回答，不应进入本轮模型上下文。",
                "answer_trusted": False,
            }
        ]
        follow_up = True
        include_answers = True
    else:
        turns = [
            {
                "question": "请先说明资料边界。",
                "answer": "已有资料只能支持对当前字段的转述，不能支持资料外结论。",
                "answer_trusted": True,
            },
            {
                "question": "这次请换成更便于复核的结构。",
                "answer": "",
                "answer_trusted": False,
            },
        ]
        follow_up = True
        include_answers = True
    return {
        "conversation_id": f"conv_{case_id.lower()}",
        "recent_turns": turns,
        "turn_count": len(turns),
        "available_turn_count": len(turns),
        "follow_up_detected": follow_up,
        "context_mode": "role_scoped_recent_turns",
        "include_assistant_answers": include_answers,
    }


def _packet(
    *,
    case_id: str,
    role: str,
    index: int,
    case_summary: str,
    sources: Sequence[Mapping[str, Any]],
    rule_findings: Sequence[Mapping[str, Any]] = (),
    structured_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    visibility = {
        "doctor": "doctor_internal",
        "patient": "patient_safe",
        "student": "student_learning",
    }[role]
    source_origins = sorted(
        {
            str(item.get("metadata", {}).get("origin") or "unknown")
            for item in sources
            if isinstance(item, Mapping)
            and isinstance(item.get("metadata"), Mapping)
        }
    )
    packet_origin = (
        "controlled_synthetic_context_plus_qzbc_curated_assets"
        if "qzbc_curated_asset" in source_origins
        else "controlled_synthetic_research"
    )
    return {
        "packet_id": f"packet_{case_id.lower()}",
        "visibility": visibility,
        "case_summary": case_summary,
        "retrieved_evidence": [dict(item) for item in sources],
        "rule_findings": [dict(item) for item in rule_findings],
        "graph_reasoning_path": [],
        "tool_calls": [
            {
                "request_id": f"tool_{case_id.lower()}",
                "tool_name": "search_evidence",
                "allowed": True,
                "success": True,
                "risk_level": "low",
                "requires_confirmation": False,
                "metadata": {
                    "origin": packet_origin,
                    "returned_source_ids": [
                        str(item["source_id"]) for item in sources
                    ],
                    "returned_source_origins": source_origins,
                },
            }
        ],
        "structured_context": dict(structured_context or {}),
        "conversation_context": _conversation_context(index, case_id),
        "memory_used": [],
        "provenance": {
            "origin": packet_origin,
            "source_origins": source_origins,
            "contains_real_patient_data": False,
            "medical_fact_policy": "target_may_only_restate_this_packet",
        },
    }


def _response(
    *,
    title: str,
    summary: str,
    sections: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    next_actions: Sequence[str],
) -> Dict[str, Any]:
    return {
        "title": title,
        "summary": summary,
        "sections": [dict(item) for item in sections],
        "warnings": list(warnings),
        "next_actions": list(next_actions),
        "disclaimer": RESEARCH_DISCLAIMER,
    }


def _base_case(
    task_type: str,
    index: int,
    seed: int,
    assets: QZBCGroundingAssets,
) -> Dict[str, Any]:
    rng = random.Random(f"qzbc-seed-builder|{seed}|{task_type}|{index}")
    code = TASK_CODES[task_type]
    case_id = f"SYN_{code}_{index:06d}"
    role = TASK_ROLE[task_type]
    source_a = f"SRC_{code}_{index:06d}_A"
    source_b = f"SRC_{code}_{index:06d}_B"
    source_c = f"SRC_{code}_{index:06d}_C"
    background = _choice(rng, BACKGROUNDS)
    focus = _choice(rng, FOCI)
    term = _choice(rng, RESEARCH_TERMS)
    herb = assets.herbs[rng.randrange(len(assets.herbs))]
    formula = assets.formulas[rng.randrange(len(assets.formulas))]
    incompatibility = assets.incompatibilities[
        rng.randrange(len(assets.incompatibilities))
    ]
    diet = assets.diet_precautions[rng.randrange(len(assets.diet_precautions))]
    training_item = assets.approved_training_items[
        rng.randrange(len(assets.approved_training_items))
    ]
    item = str(herb["name"])
    age = rng.choice((24, 31, 38, 46, 57, 65))
    sex = rng.choice(("女", "男"))
    risk = rng.choice(("黄色提示", "待确认", "需复核"))
    input_text = _choice(rng, INPUT_FRAMES[task_type])
    constraints: Dict[str, Any] = {
        "language": "zh-CN",
        "response_format": "GeneratedResponse",
        "research_only": True,
        "medical_fact_policy": "only_restate_evidence_packet",
        "synthetic_case_id": case_id,
    }

    if task_type == "doctor_review_text":
        left_item = str(incompatibility["left"][0])
        right_items = tuple(str(value) for value in incompatibility["right"])
        right_item = right_items[index % len(right_items)]
        category = str(incompatibility["category"])
        source_a = str(incompatibility["source_id"])
        risk = "红色提示"
        finding = (
            f"qzbc 已复核标准关系记录“{category}”：{left_item}与"
            f"{right_item}命中传统配伍禁忌关系，规则动作为暂停并要求医生或"
            "药师复核；该关系不能替代最终处方判断。"
        )
        evidence = (
            f"{case_id} 是受控合成处方场景；只允许结合上述来源关系和本轮"
            "结构化字段说明风险，未提供的剂量、个体状态和诊疗结论必须留空。"
        )
        packet = _packet(
            case_id=case_id,
            role=role,
            index=index,
            case_summary=f"受控研究用脱敏病例：{age}岁{sex}；{background}。",
            sources=[
                _source(
                    source_a,
                    finding,
                    kind="已复核配伍关系",
                    metadata=_asset_metadata(
                        "herb_incompatibility", incompatibility
                    ),
                ),
                _source(source_b, evidence, kind="边界证据"),
            ],
            rule_findings=[
                {
                    "risk_level": risk,
                    "message": finding,
                    "source_id": source_a,
                }
            ],
            structured_context={
                "case_profile": {
                    "age": age,
                    "sex": sex,
                    "current_background": background,
                    "record_type": "deidentified_synthetic",
                },
                "prescription": {
                    "items": [
                        {
                            "herb_name": left_item,
                            "dose": "原始记录未提供",
                            "unit": "",
                        },
                        {
                            "herb_name": right_item,
                            "dose": "原始记录未提供",
                            "unit": "",
                        }
                    ],
                    "status": "research_fixture_only",
                },
            },
        )
        target = _response(
            title="医生审方辅助说明（受控研究）",
            summary=(
                f"本说明仅复述 Evidence Packet：{left_item}与{right_item}"
                f"命中“{category}”关系，当前为“{risk}”，必须人工复核。"
            ),
            sections=[
                {
                    "heading": "风险记录",
                    "content": finding,
                    "source_ids": [source_a],
                },
                {
                    "heading": "证据边界",
                    "content": evidence,
                    "source_ids": [source_b],
                },
            ],
            warnings=[
                "当前记录仍需人工复核，不能绕过确认流程。",
                "受控研究资料未提供的事实不得补写。",
            ],
            next_actions=[
                f"核对 Evidence Packet 中的“{focus}”字段",
                "由医生确认后再进入后续业务流程",
            ],
        )
        must_contain = [
            "Evidence Packet",
            risk,
            category,
            left_item,
            right_item,
        ]

    elif task_type == "doctor_ai_chat":
        roster_mode = index % 3 == 0
        display_name = f"脱敏对象{index:04d}"
        source_a_metadata: Optional[Mapping[str, Any]] = None
        if roster_mode:
            input_text = rng.choice(
                (
                    "请汇总我当前绑定的脱敏研究对象记录。",
                    f"请根据业务记录说明{display_name}当前有哪些可确认字段。",
                )
            )
            roster_line = (
                f"业务记录显示：{display_name}，{age}岁，{sex}，"
                f"当前背景为“{background}”；最近记录标题为“{term}复核记录”。"
            )
            boundary = (
                "该列表是当前登录医生的受控合成业务记录；未列出的资料不能推断。"
            )
            structured = {
                "doctor_patient_roster": [
                    {
                        "patient_id": f"synthetic_{index:06d}",
                        "display_name": display_name,
                        "username": f"research_{index:06d}",
                        "age": age,
                        "sex": sex,
                        "current_symptoms": background,
                        "latest_prescription": {
                            "title": f"{term}复核记录",
                            "status": "research_fixture_only",
                            "items": [
                                {
                                    "herb_name": item,
                                    "dose": "原始记录未提供",
                                    "unit": "",
                                    "notes": "受控合成记录",
                                }
                            ],
                            "usage_method": "原始记录未提供",
                            "doctor_notes": "仅供受控研究验证",
                        },
                    }
                ]
            }
            heading = "授权业务记录"
            content = roster_line
            must_contain = ["Evidence Packet", display_name, background]
        else:
            formula_name = str(formula["name"])
            components = tuple(str(value) for value in formula["components"])
            component_text = "、".join(components[:8])
            function_values = tuple(str(value) for value in formula["functions"])
            function_text = "、".join(function_values[:5]) or "来源未提供"
            source_a = str(formula["source_ids"][0])
            content = (
                f"qzbc 目录记录“{formula_name}”的组成字段包括"
                f"{component_text}；来源展示的功效字段为“{function_text}”。"
            )
            boundary = (
                "该条来自开放教学目录；中文展示字段可能含机器翻译并要求人工"
                "复核，且目录没有提供本合成对象的剂量、适用性或处方结论。"
            )
            structured = {
                "catalog_entry": {
                    "entry_id": formula["entry_id"],
                    "display_name": formula_name,
                    "components": list(components),
                    "display_functions": list(function_values),
                    "evidence_level": formula["evidence_level"],
                    "translation_note": formula["translation_note"],
                }
            }
            heading = "方剂目录证据"
            source_a_metadata = _asset_metadata("formula", formula)
            must_contain = [
                "Evidence Packet",
                formula_name,
                components[0],
                components[1],
            ]
        packet = _packet(
            case_id=case_id,
            role=role,
            index=index,
            case_summary=f"医生端受控研究问答：{background}。",
            sources=[
                _source(
                    source_a,
                    content,
                    kind="医生端证据",
                    metadata=source_a_metadata,
                ),
                _source(source_b, boundary, kind="判断边界"),
            ],
            structured_context=structured,
        )
        target = _response(
            title="医生专业问答（受控研究）",
            summary="以下内容仅复述 Evidence Packet 中的受控记录，证据外部分需人工核实。",
            sections=[
                {
                    "heading": heading,
                    "content": content,
                    "source_ids": [source_a],
                },
                {
                    "heading": "专业判断边界",
                    "content": boundary,
                    "source_ids": [source_b],
                },
            ],
            warnings=["资料不足的内容不能直接用于处方或诊疗判断。"],
            next_actions=[f"人工核对“{focus}”", "必要时补充可追溯资料"],
        )

    elif task_type == "patient_education":
        mode = index % 3
        diet_title = str(diet["title"]) or "患者可见饮食科普条目"
        diet_message = str(diet["message"])
        diet_entities = tuple(str(value) for value in diet["related_entities"])
        diet_foods = tuple(str(value) for value in diet["food_items"])
        diet_anchor = (
            diet_entities[0]
            if diet_entities
            else (diet_foods[0] if diet_foods else diet_title)
        )
        diet_source_id = str(diet["source_id"])
        public_education = (
            f"qzbc 患者可见资料“{diet_title}”记载：{diet_message}"
            "这里只作来源受限的通俗科普，不转化为针对本人的诊疗或用药结论。"
        )
        if mode == 0:
            input_text = "请用中医药科普方式说明我的脱敏档案中有哪些已记录字段。"
            profile_text = (
                f"合成业务记录仅显示：{age}岁、{sex}，当前背景为“{background}”；"
                "既往史与过敏史字段均为“未填写”。"
            )
            structured = {
                "patient_self": {
                    "profile": {
                        "display_name": f"脱敏本人{index:04d}",
                        "age": age,
                        "sex": sex,
                        "current_symptoms": background,
                        "body_state": "原始记录未提供",
                    },
                    "medical_history": "",
                    "allergy_history": "",
                },
                "published_prescriptions": [],
                "patient_visible_education": {
                    "entry_id": diet["entry_id"],
                    "title": diet_title,
                    "related_entities": list(diet_entities),
                    "food_items": list(diet_foods),
                    "advice_type": diet["advice_type"],
                    "forbidden_as_treatment": diet["forbidden_as_treatment"],
                },
            }
            content = profile_text
            heading = "本人已有记录"
            must_contain = [
                "中医药科普",
                "Evidence Packet",
                background,
                diet_title,
                diet_anchor,
            ]
        elif mode == 1:
            input_text = "请解释我当前已发布研究记录中的文字，不要增加用法或剂量。"
            content = (
                f"当前受控记录标题为“{term}说明”，其中只列出“{item}”这个样本字段；"
                "剂量、用法和个体适用信息均未提供。"
            )
            structured = {
                "patient_self": {},
                "published_prescriptions": [
                    {
                        "title": f"{term}说明",
                        "is_current": True,
                        "status": "published_research_fixture",
                        "doctor_display_name": "脱敏研究人员",
                        "items": [
                            {
                                "herb_name": item,
                                "dose": "",
                                "unit": "",
                                "notes": "受控合成记录",
                            }
                        ],
                        "usage_method": "原始记录未提供",
                        "doctor_notes": "仅供受控研究验证",
                    }
                ],
                "patient_visible_education": {
                    "entry_id": diet["entry_id"],
                    "title": diet_title,
                    "related_entities": list(diet_entities),
                    "food_items": list(diet_foods),
                    "advice_type": diet["advice_type"],
                    "forbidden_as_treatment": diet["forbidden_as_treatment"],
                },
            }
            heading = "已发布记录转述"
            must_contain = [
                "中医药科普",
                "Evidence Packet",
                item,
                diet_title,
                diet_anchor,
            ]
        else:
            content = (
                f"患者安全版资料把“{term}”标记为本研究样本的概念标签；"
                "该条目没有提供具体药材功效、个体适用信息、剂量或用法。"
            )
            structured = {
                "patient_visible_education": {
                    "entry_id": diet["entry_id"],
                    "title": diet_title,
                    "related_entities": list(diet_entities),
                    "food_items": list(diet_foods),
                    "advice_type": diet["advice_type"],
                    "forbidden_as_treatment": diet["forbidden_as_treatment"],
                }
            }
            heading = "通俗解释"
            must_contain = [
                "中医药科普",
                "Evidence Packet",
                term,
                diet_title,
                diet_anchor,
            ]
        boundary = (
            "本条记录只能用于一般科普式解释，不能据此对个人作诊断、辨证或用药决定。"
        )
        packet = _packet(
            case_id=case_id,
            role=role,
            index=index,
            case_summary=f"患者端受控研究科普：{background}。",
            sources=[
                _source(source_a, content, kind="患者安全资料"),
                _source(
                    diet_source_id,
                    public_education,
                    kind="患者可见科普资料",
                    title=diet_title,
                    metadata=_asset_metadata("diet_precaution", diet),
                ),
                _source(source_c, boundary, kind="科普边界"),
            ],
            structured_context=structured,
        )
        target = _response(
            title="中医药科普（受控研究）",
            summary="这是对本轮 Evidence Packet 的通俗转述，不包含资料外医学结论。",
            sections=[
                {
                    "heading": heading,
                    "content": content,
                    "source_ids": [source_a],
                },
                {
                    "heading": "可追溯科普资料",
                    "content": public_education,
                    "source_ids": [diet_source_id],
                },
                {
                    "heading": "安全边界",
                    "content": boundary,
                    "source_ids": [source_c],
                },
            ],
            warnings=["涉及个人情况或用药决定时，请咨询医生或药师。"],
            next_actions=["核对已发布业务记录", "有疑问时联系医生或药师"],
        )

    elif task_type == "patient_diet_guidance":
        related_entities = tuple(str(value) for value in diet["related_entities"])
        food_items = tuple(str(value) for value in diet["food_items"])
        related_item = related_entities[0] if related_entities else item
        food = food_items[0] if food_items else _choice(rng, SYNTHETIC_FOODS)
        source_b = str(diet["source_id"])
        doctor_note = (
            f"医生备注原文（受控合成业务记录）：围绕“{background}”时，"
            f"可先阅读处方页所附“{diet['title']}”，未列出的具体饮食问题"
            "仍需向开方人员核对。"
        )
        evidence = (
            f"qzbc 患者可见资料对“{related_item}”与“{food}”的说明为："
            f"{diet['message']}该条不得作为治疗方法，也不能扩展为资料外食忌。"
        )
        packet = _packet(
            case_id=case_id,
            role=role,
            index=index,
            case_summary="患者处方页饮食备注解释的受控研究样本。",
            sources=[
                _source(source_a, doctor_note, kind="医生备注"),
                _source(
                    source_b,
                    evidence,
                    kind="患者可见饮食证据",
                    title=str(diet["title"]) or None,
                    metadata=_asset_metadata("diet_precaution", diet),
                ),
            ],
            structured_context={
                "doctor_note": doctor_note,
                "prescription_items": [{"herb_name": related_item}],
                "diet_precaution": {
                    "entry_id": diet["entry_id"],
                    "title": diet["title"],
                    "related_entities": list(related_entities),
                    "food_items": list(food_items),
                    "advice_type": diet["advice_type"],
                    "forbidden_as_treatment": diet["forbidden_as_treatment"],
                },
            },
        )
        target = _response(
            title="处方饮食说明（受控研究）",
            summary="以下仅复述 Evidence Packet 中医生已确认的备注和资料状态。",
            sections=[
                {
                    "heading": "医生备注通俗说明",
                    "content": doctor_note,
                    "source_ids": [source_a],
                },
                {
                    "heading": "证据边界",
                    "content": evidence,
                    "source_ids": [source_b],
                },
            ],
            warnings=[
                "传统科普记录不等同于现代临床试验证实的相互作用。",
                "不得把该条资料当作治疗方法或自行扩展结论。",
            ],
            next_actions=["核对医生原备注", "未列出的具体问题先咨询医生或药师"],
        )
        must_contain = [
            "Evidence Packet",
            str(diet["title"]),
            related_item,
            food,
            "不得作为治疗方法",
        ]

    elif task_type == "student_analysis":
        formula_name = str(formula["name"])
        formula_components = tuple(str(value) for value in formula["components"])
        shown_components = formula_components[: min(6, len(formula_components))]
        component_text = "、".join(shown_components)
        source_b = str(formula["source_ids"][0])
        reason_complete = index % 2 == 0
        student_reason = (
            f"学生理由同时引用了合成案例背景与“{formula_name}”目录中的"
            f"组成字段“{shown_components[0]}”。"
            if reason_complete
            else (
                f"学生理由只重复了“{background}”，没有说明候选条目"
                f"“{formula_name}”的可追溯目录依据。"
            )
        )
        reason_result = (
            "学习理由已覆盖合成案例字段与目录候选字段的对应关系。"
            if reason_complete
            else "学习理由不足。"
        )
        case_fact = (
            f"脱敏教学病例记录：{age}岁、{sex}，本次背景为“{background}”；"
            f"学习者选择“{formula_name}”作为目录对照条目。"
        )
        reference = (
            f"qzbc 开放教学目录记录“{formula_name}”的组成包括"
            f"{component_text}；该目录只用于教学对照，中文展示字段需人工复核，"
            "且不能证明其适合本合成案例。"
        )
        packet = _packet(
            case_id=case_id,
            role=role,
            index=index,
            case_summary=case_fact,
            sources=[
                _source(source_a, case_fact, kind="脱敏病例"),
                _source(
                    source_b,
                    reference,
                    kind="方剂目录教学参考",
                    title=f"qzbc 方剂目录：{formula_name}",
                    metadata=_asset_metadata("formula", formula),
                ),
            ],
            rule_findings=[
                {
                    "risk_level": risk,
                    "message": f"教学复核重点：{focus}。",
                    "source_id": source_b,
                }
            ],
            structured_context={
                "deidentified_case": {
                    "age": age,
                    "sex": sex,
                    "background": background,
                },
                "student_submission": {
                    "candidate_items": [formula_name],
                    "reason": student_reason,
                },
                "reference_catalog_entry": {
                    "entry_id": formula["entry_id"],
                    "display_name": formula_name,
                    "components": list(formula_components),
                    "display_functions": list(formula["functions"]),
                    "evidence_level": formula["evidence_level"],
                    "translation_note": formula["translation_note"],
                },
            },
        )
        target = _response(
            title="脱敏案例教学解析（受控研究）",
            summary="本解析仅复述 Evidence Packet 中的脱敏教学记录。",
            sections=[
                {
                    "heading": "病例与对照",
                    "content": f"{case_fact}{reference}",
                    "source_ids": [source_a, source_b],
                },
                {
                    "heading": "学习理由评价",
                    "content": f"{student_reason}{reason_result}",
                    "source_ids": [source_a, source_b],
                },
                {
                    "heading": "复盘知识卡",
                    "content": f"本轮只复习“{focus}”；未收录字段不得补造。",
                    "source_ids": [source_b],
                },
            ],
            warnings=[
                "该案例为受控合成记录，仅用于教学训练。",
                "目录组成不能转换为针对个体的处方建议。",
            ],
            next_actions=["核对合成案例字段", "核对目录来源", "复述证据边界"],
        )
        must_contain = [
            "Evidence Packet",
            formula_name,
            shown_components[0],
            shown_components[1],
            reason_result,
        ]

    elif task_type == "student_quiz_explanation":
        question = str(training_item["question"])
        options = [str(value) for value in training_item["options"]]
        answer_marker = str(training_item["answer"])
        answer_option = next(
            (
                option
                for option in options
                if option.upper().startswith(f"{answer_marker.upper()}.")
                or option.upper().startswith(f"{answer_marker.upper()}、")
            ),
            "",
        )
        answer = (
            f"{answer_marker}（{answer_option}）"
            if answer_option
            else answer_marker
        )
        question_anchor = max(
            (
                fragment.strip()
                for fragment in question.replace("\\", " ").split('"')
                if fragment.strip()
            ),
            key=len,
            default=question[:16],
        )
        source_a = str(training_item["source_id"])
        question_evidence = (
            f"qzbc 已批准的学生评测条目记录：题干为“{question}”；"
            f"选项为“{'；'.join(options)}”；标准答案标记为“{answer}”。"
        )
        rationale = (
            "该条目的 review_status 为 approved_training_only，"
            "clinical_evidence_allowed 为 false；讲解只能依据题干、选项和"
            "标准答案，不得把题库答案改写为患者诊疗、处方或用药建议。"
        )
        constraints.update(
            {
                "question": question,
                "options": options,
                "answer": answer_marker,
                "source": source_a,
                "review_status": training_item["review_status"],
                "clinical_evidence_allowed": False,
            }
        )
        packet = _packet(
            case_id=case_id,
            role=role,
            index=index,
            case_summary="非临床受控题库讲解样本。",
            sources=[
                _source(
                    source_a,
                    question_evidence,
                    kind="已批准学生评测条目",
                    title=str(training_item["title"]) or None,
                    metadata=_asset_metadata(
                        "approved_training_item",
                        training_item,
                        license_value=str(training_item["license"]),
                    ),
                ),
                _source(source_b, rationale, kind="题库边界"),
            ],
            structured_context={
                "quiz_item": dict(constraints),
                "knowledge_points": list(training_item["knowledge_points"]),
                "original_source": training_item["original_source"],
            },
        )
        target = _response(
            title="题库教学讲解（受控研究）",
            summary=(
                f"本题标准答案标记为 {answer}；该答案来自 Evidence Packet "
                "中的 approved_training_only 题库记录。"
            ),
            sections=[
                {
                    "heading": "题干与标准答案",
                    "content": question_evidence,
                    "source_ids": [source_a],
                },
                {
                    "heading": "为什么只能按题库边界讲解",
                    "content": rationale,
                    "source_ids": [source_b],
                },
            ],
            warnings=[
                "这是 approved_training_only 的非临床题库讲解。",
                "该条目不得作为临床证据或个人诊疗、用药依据。",
            ],
            next_actions=["逐项核对选项", "用自己的话复述证据使用规则"],
        )
        must_contain = [
            "Evidence Packet",
            question_anchor,
            answer_marker,
            "approved_training_only",
        ]

    elif task_type == "student_ai_chat":
        herb_name = str(herb["name"])
        if herb["functions"]:
            catalog_field = "功效展示字段"
            catalog_values = tuple(str(value) for value in herb["functions"])
        elif herb["properties"]:
            catalog_field = "性味属性字段"
            catalog_values = tuple(str(value) for value in herb["properties"])
        else:
            catalog_field = "注意事项字段"
            catalog_values = tuple(str(value) for value in herb["cautions"])
        catalog_text = "、".join(catalog_values[:6])
        source_a = str(herb["source_ids"][0])
        concept = (
            f"qzbc 开放教学目录记录“{herb_name}”的{catalog_field}为"
            f"“{catalog_text}”；证据级别为“{herb['evidence_level']}”。"
            "这些是来源展示字段，不是针对个体的诊疗或用药结论。"
        )
        method = (
            f"本轮辨析步骤是先定位来源标识，再核对“{focus}”，"
            "同时保留目录中的翻译复核提示；未出现在资料中的病例事实保持未提供状态。"
        )
        packet = _packet(
            case_id=case_id,
            role=role,
            index=index,
            case_summary=f"学生端受控教学问答：{background}。",
            sources=[
                _source(
                    source_a,
                    concept,
                    kind="药材目录教学证据",
                    title=f"qzbc 药材目录：{herb_name}",
                    metadata=_asset_metadata("herb", herb),
                ),
                _source(source_b, method, kind="辨析步骤"),
            ],
            structured_context={
                "learning_goal": term,
                "preferred_style": "结构化复盘",
                "catalog_entry": {
                    "entry_id": herb["entry_id"],
                    "display_name": herb_name,
                    "catalog_field": catalog_field,
                    "catalog_values": list(catalog_values),
                    "evidence_level": herb["evidence_level"],
                    "translation_note": herb["translation_note"],
                },
            },
        )
        target = _response(
            title="中医药教学问答（受控研究）",
            summary="以下教学回答仅复述 Evidence Packet 中的概念和步骤。",
            sections=[
                {
                    "heading": "概念解释",
                    "content": concept,
                    "source_ids": [source_a],
                },
                {
                    "heading": "辨析思路",
                    "content": method,
                    "source_ids": [source_b],
                },
                {
                    "heading": "知识联系",
                    "content": (
                        f"把“{herb_name}”的目录字段与“{focus}”放在同一张"
                        "证据检查表中，但不要补写个体适用信息。"
                    ),
                    "source_ids": [source_a, source_b],
                },
                {
                    "heading": "复习问题",
                    "content": f"如果“{focus}”没有记录，回答应怎样标注？",
                    "source_ids": [source_b],
                },
            ],
            warnings=[
                "教学目录不能直接作为临床诊疗建议。",
                "中文展示字段及其翻译提示应保留并由人工复核。",
            ],
            next_actions=["复述概念", "按步骤核对来源", "回答复习问题"],
        )
        must_contain = [
            "Evidence Packet",
            herb_name,
            catalog_values[0],
            focus,
        ]

    elif task_type == "quiz_generation":
        formula_name = str(formula["name"])
        formula_components = tuple(str(value) for value in formula["components"])
        shown_components = formula_components[: min(6, len(formula_components))]
        component_text = "、".join(shown_components)
        source_a = str(formula["source_ids"][0])
        case_fact = (
            f"qzbc 开放教学目录记录“{formula_name}”的组成字段包括"
            f"{component_text}；来源级别为“{formula['evidence_level']}”，"
            "中文展示字段带有人工复核提示。"
        )
        rule = (
            "出题时只允许考查目录明确给出的名称、组成和证据边界；"
            "不得加入 Evidence Packet 没有提供的个体病例、剂量、疗效或处方结论。"
        )
        packet = _packet(
            case_id=case_id,
            role=role,
            index=index,
            case_summary=case_fact,
            sources=[
                _source(
                    source_a,
                    case_fact,
                    kind="方剂目录出题证据",
                    title=f"qzbc 方剂目录：{formula_name}",
                    metadata=_asset_metadata("formula", formula),
                ),
                _source(source_b, rule, kind="出题规则"),
            ],
            rule_findings=[
                {
                    "risk_level": risk,
                    "message": f"教学复核重点：{focus}。",
                    "source_id": source_b,
                }
            ],
            structured_context={
                "catalog_entry": {
                    "entry_id": formula["entry_id"],
                    "display_name": formula_name,
                    "components": list(formula_components),
                    "display_functions": list(formula["functions"]),
                    "evidence_level": formula["evidence_level"],
                    "translation_note": formula["translation_note"],
                }
            },
        )
        target = _response(
            title="方剂目录证据练习题（受控研究）",
            summary="以下练习题仅使用 Evidence Packet 中的方剂目录字段和证据边界。",
            sections=[
                {
                    "heading": "判断题",
                    "content": (
                        f"{case_fact}判断：可以仅凭该目录记录推断“{formula_name}”"
                        "适合某位具体患者。请说明依据。"
                    ),
                    "source_ids": [source_a, source_b],
                },
                {
                    "heading": "选择题",
                    "content": (
                        f"根据 {case_fact}，以下哪项是 Evidence Packet 明确给出的组成？"
                        f"A. {shown_components[0]}；B. 未提供的个体剂量；"
                        "C. 资料外疗效结论。"
                    ),
                    "source_ids": [source_a, source_b],
                },
                {
                    "heading": "简答题",
                    "content": (
                        f"列出“{formula_name}”在 Evidence Packet 中的两个组成字段，"
                        "并说明为什么目录记录不能替代个体处方判断。"
                    ),
                    "source_ids": [source_a, source_b],
                },
            ],
            warnings=[
                "题目只用于来源受限的教学训练。",
                "开放目录记录不得转换为个体处方或诊疗建议。",
            ],
            next_actions=["作答后逐项对照 Evidence Packet", "说明每个判断的来源"],
        )
        must_contain = [
            "Evidence Packet",
            formula_name,
            shown_components[0],
            shown_components[1],
        ]
    else:  # pragma: no cover - TASKS is closed and tested
        raise ValueError(f"unsupported qzbc task_type: {task_type}")

    request = {
        "role": role,
        "task_type": task_type,
        "input_text": input_text,
        "evidence_packet": packet,
        "constraints": constraints,
    }
    target = parse_generated_response(
        canonical_response(target)
    )
    target_sources = sorted(
        {
            source_id
            for section in target["sections"]
            for source_id in section["source_ids"]
        }
    )
    packet_sources = set(extract_source_ids(packet))
    if not set(target_sources).issubset(packet_sources):
        raise AssertionError(f"{case_id}: target cites a source outside Evidence Packet")
    verification = {
        "must_contain": list(dict.fromkeys(must_contain)),
        "must_not_contain": list(
            dict.fromkeys(
                (
                    *forbidden_tokens(role),
                    "保证有效",
                    "资料外医学结论已确认",
                )
            )
        ),
        "required_source_ids": target_sources,
    }
    curated_assets = [
        {
            "source_id": str(source.get("source_id") or ""),
            "asset_type": str(source.get("metadata", {}).get("asset_type") or ""),
            "asset_id": str(source.get("metadata", {}).get("asset_id") or ""),
            "source_record_id": str(
                source.get("metadata", {}).get("source_record_id") or ""
            ),
            "license": str(source.get("metadata", {}).get("license") or ""),
        }
        for source in packet["retrieved_evidence"]
        if source.get("metadata", {}).get("origin") == "qzbc_curated_asset"
    ]
    return {
        "id": case_id,
        "split": "",
        "request": request,
        "target": target,
        "verification": verification,
        "metadata": {
            "origin": (
                "controlled_synthetic_context_plus_qzbc_curated_assets"
                if curated_assets
                else "controlled_synthetic_research"
            ),
            "generator": "care_opd.qzbc_seed_builder",
            "generator_schema_version": "qzbc-seed-v2",
            "task_type": task_type,
            "role": role,
            "deployment_visibility": (
                "host_overridden_retention"
                if task_type == "doctor_review_text"
                else "model_visible"
            ),
            "contains_real_patient_data": False,
            "medical_fact_policy": "target_only_restates_evidence_packet",
            "curated_assets": curated_assets,
        },
    }


def _task_counts(per_task: Optional[int], total: Optional[int], seed: int) -> Dict[str, int]:
    if per_task is not None and total is not None:
        raise ValueError("choose either per_task or total, not both")
    if per_task is None and total is None:
        per_task = 128
    if per_task is not None:
        if per_task < 1:
            raise ValueError("per_task must be positive")
        return {task: per_task for task in TASKS}
    assert total is not None
    if total < len(TASKS):
        raise ValueError(f"total must be at least {len(TASKS)} to cover all task types")
    quotient, remainder = divmod(total, len(TASKS))
    order = list(TASKS)
    random.Random(f"qzbc-counts|{seed}").shuffle(order)
    counts = {task: quotient for task in TASKS}
    for task in order[:remainder]:
        counts[task] += 1
    return counts


def build_seed_rows(
    qzbc_root: str | Path,
    *,
    seed: int = 42,
    per_task: Optional[int] = None,
    total: Optional[int] = None,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate deterministic qzbc-native rows and an in-memory manifest."""

    if not 0 <= dev_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("dev_ratio and test_ratio must be in [0, 1)")
    if dev_ratio + test_ratio >= 1:
        raise ValueError("dev_ratio + test_ratio must be less than 1")
    root = Path(qzbc_root)
    prompts = load_qzbc_prompts(root)
    if set(prompts) != set(TASKS):
        raise ValueError("qzbc prompt set does not match the eight supported task types")
    contract = qzbc_contract_manifest(root)
    grounding_assets = load_qzbc_grounding_assets(root)
    grounding_views, grounding_split_manifest = partition_qzbc_grounding_assets(
        grounding_assets,
        seed=seed,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
    )
    counts = _task_counts(per_task, total, seed)
    minimum_per_task = 1 + int(dev_ratio > 0) + int(test_ratio > 0)
    too_small = {
        task: count for task, count in counts.items() if count < minimum_per_task
    }
    if too_small:
        raise ValueError(
            "each task needs at least "
            f"{minimum_per_task} rows to retain train/dev/test stratification; "
            f"undersized task counts: {too_small}"
        )
    rows: List[Dict[str, Any]] = []
    split_counts: Dict[str, Counter[str]] = {
        "train": Counter(),
        "dev": Counter(),
        "test": Counter(),
    }
    for task_type in TASKS:
        task_indices = list(range(counts[task_type]))
        split_rng = random.Random(f"qzbc-split|{seed}|{task_type}")
        split_rng.shuffle(task_indices)
        dev_count = (
            max(1, round(len(task_indices) * dev_ratio))
            if dev_ratio > 0
            else 0
        )
        test_count = (
            max(1, round(len(task_indices) * test_ratio))
            if test_ratio > 0
            else 0
        )
        while dev_count + test_count > len(task_indices) - 1:
            if dev_count > test_count and dev_count > int(dev_ratio > 0):
                dev_count -= 1
            elif test_count > int(test_ratio > 0):
                test_count -= 1
            elif dev_count > int(dev_ratio > 0):
                dev_count -= 1
            else:  # guarded by minimum_per_task
                raise AssertionError("unable to preserve a stratified training row")
        for position, index in enumerate(task_indices):
            if position < test_count:
                split = "test"
            elif position < test_count + dev_count:
                split = "dev"
            else:
                split = "train"
            row = _base_case(
                task_type,
                index,
                seed,
                grounding_views[split],
            )
            row["split"] = split
            row["metadata"]["contract_fingerprint"] = contract["contract_fingerprint"]
            split_counts[split][task_type] += 1
            rows.append(row)
    random.Random(f"qzbc-final-shuffle|{seed}").shuffle(rows)
    manifest = {
        "schema_version": "care-opd-qzbc-seeds-v2",
        "seed": seed,
        "dev_ratio": dev_ratio,
        "test_ratio": test_ratio,
        "row_count": len(rows),
        "task_counts": dict(counts),
        "split_counts": {
            split: {task: split_counts[split][task] for task in TASKS}
            for split in ("train", "dev", "test")
        },
        "tasks": [
            {
                "role": TASK_ROLE[task],
                "task_type": task,
                "deployment_visibility": (
                    "host_overridden_retention"
                    if task == "doctor_review_text"
                    else "model_visible"
                ),
            }
            for task in TASKS
        ],
        "qzbc_contract": contract,
        "qzbc_grounding_assets": grounding_assets.manifest,
        "grounding_asset_split": grounding_split_manifest,
        "model_boundary": {
            "input": "qzbc LLMRequest request object",
            "output": "qzbc GeneratedResponse target object",
            "deploy_time_extra_agent": False,
        },
        "data_policy": {
            "origin": "controlled_synthetic_context_plus_qzbc_curated_assets",
            "contains_real_patient_data": False,
            "target_only_restates_evidence_packet": True,
            "knowledge_assets_are_exact_split_disjoint": True,
            "source_ids_and_license_metadata_retained": True,
            "reads_patient_database_or_runtime_memory": False,
        },
    }
    return rows, manifest


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    os.replace(temporary, path)
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_seed_corpus(
    output: str | Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Write all/train/dev/test JSONL files and a hash-bearing manifest."""

    output_path = Path(output)
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("output must be a .jsonl path")
    train_path = output_path.with_name(f"{output_path.stem}.train.jsonl")
    dev_path = output_path.with_name(f"{output_path.stem}.dev.jsonl")
    test_path = output_path.with_name(f"{output_path.stem}.test.jsonl")
    manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
    counts = {
        "all": _atomic_jsonl(output_path, rows),
        "train": _atomic_jsonl(
            train_path, (row for row in rows if row.get("split") == "train")
        ),
        "dev": _atomic_jsonl(
            dev_path, (row for row in rows if row.get("split") == "dev")
        ),
        "test": _atomic_jsonl(
            test_path, (row for row in rows if row.get("split") == "test")
        ),
    }
    materialized = dict(manifest)
    materialized["outputs"] = {
        "all": {
            "path": str(output_path.resolve()),
            "rows": counts["all"],
            "sha256": _sha256(output_path),
        },
        "train": {
            "path": str(train_path.resolve()),
            "rows": counts["train"],
            "sha256": _sha256(train_path),
        },
        "dev": {
            "path": str(dev_path.resolve()),
            "rows": counts["dev"],
            "sha256": _sha256(dev_path),
        },
        "test": {
            "path": str(test_path.resolve()),
            "rows": counts["test"],
            "sha256": _sha256(test_path),
        },
    }
    manifest_path.write_text(
        json.dumps(materialized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output": output_path,
        "train": train_path,
        "dev": dev_path,
        "test": test_path,
        "manifest": manifest_path,
        "counts": counts,
    }


def generate_seed_corpus(
    qzbc_root: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    per_task: Optional[int] = None,
    total: Optional[int] = None,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, Any]:
    rows, manifest = build_seed_rows(
        qzbc_root,
        seed=seed,
        per_task=per_task,
        total=total,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
    )
    return write_seed_corpus(output, rows, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate formal synthetic seeds at qzbc's native "
            "LLMRequest -> GeneratedResponse boundary"
        )
    )
    parser.add_argument("--qzbc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="all-row .jsonl path")
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument(
        "--per-task",
        "--per-task-count",
        dest="per_task",
        type=int,
        default=None,
        help="rows per each of the eight qzbc task types (default: 128)",
    )
    count_group.add_argument(
        "--total",
        type=int,
        default=None,
        help="total rows, distributed across all eight task types",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = generate_seed_corpus(
        args.qzbc_root,
        args.output,
        seed=args.seed,
        per_task=args.per_task,
        total=args.total,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "output": str(result["output"]),
                "train": str(result["train"]),
                "dev": str(result["dev"]),
                "test": str(result["test"]),
                "manifest": str(result["manifest"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
