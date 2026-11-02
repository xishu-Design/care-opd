#!/usr/bin/env python3
"""Freeze a same-case, doctor/patient/student real-record evaluation package.

The script performs no inference.  It samples de-identified Structured-track
families without looking at model performance, then emits private task rows and
an aggregate-only public plan manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .three_role_eval_common import (
        DEFAULT_VARIANTS,
        ROLES,
        assert_public_aggregate,
        case_key,
        family_key,
        opaque_token,
        read_jsonl,
        sha256_file,
        write_json,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - direct script execution
    from three_role_eval_common import (  # type: ignore
        DEFAULT_VARIANTS,
        ROLES,
        assert_public_aggregate,
        case_key,
        family_key,
        opaque_token,
        read_jsonl,
        sha256_file,
        write_json,
        write_jsonl,
    )


SCHEMA_VERSION = "care_opd.competition.three_role_plan.v1"
OUTPUT_CONDITION = {
    "clean": "clean",
    "source_outage": "evidence_unavailable_counterfactual",
}
ROLE_SYSTEM_PROMPTS = {
    "doctor": (
        "你是中医药医疗智能体的医生专业辅助角色。严格依据给定记录与证据状态，"
        "区分已知事实、尚待核实内容与复核建议；不得补造事实，最终判断由执业医生完成。"
    ),
    "patient": (
        "你是中医药医疗智能体的患者科普角色。保持医疗事实不变，用通俗语言解释并提示风险；"
        "不得作确定性诊断、开方、改药、加减量或替代线下就医。"
    ),
    "student": (
        "你是中医药医疗智能体的学生教学角色。保持医疗事实不变，给出知识解释、"
        "可核验的依据链与学习反馈；不要求也不输出隐藏思维过程，不把教学分析当作临床结论。"
    ),
}
GENERIC_SYSTEM_PROMPT = (
    "严格依据给定的去标识化病例记录与证据状态回答。保持事实一致，不补造信息；"
    "证据不足时明确降级，并说明需要补充或复核的内容。"
)
GENERIC_TASK = (
    "请按当前系统角色解释；只依据共同证据，不补造事实。证据不足时明确降级，"
    "并给出角色权限内的复核建议。"
)


def _parse_values(values: list[str] | None, default: tuple[str, ...]) -> list[str]:
    if not values:
        return list(default)
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return list(dict.fromkeys(result))


def _stratified_sample(
    clean_rows: list[dict[str, Any]], sample_size: int, seed: int
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    strata: dict[str, list[str]] = defaultdict(list)
    for row in clean_rows:
        modality = str(row.get("modality") or "unknown")
        strata[modality].append(family_key(row))
    strata = {name: sorted(set(values)) for name, values in strata.items()}
    available = {name: len(values) for name, values in strata.items()}
    total = sum(available.values())
    if sample_size > total:
        raise ValueError(f"sample size {sample_size} exceeds {total} available families")

    exact = {name: sample_size * count / total for name, count in available.items()}
    allocation = {name: min(available[name], int(exact[name])) for name in strata}
    remaining = sample_size - sum(allocation.values())
    order = sorted(strata, key=lambda name: (-(exact[name] - int(exact[name])), name))
    while remaining:
        progressed = False
        for name in order:
            if allocation[name] < available[name]:
                allocation[name] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("failed to allocate stratified sample")

    selected: list[str] = []
    for name in sorted(strata):
        ranked = sorted(
            strata[name],
            key=lambda family: hashlib.sha256(
                f"{seed}\x00{name}\x00{family}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[: allocation[name]])
    return sorted(selected), available, allocation


def _evidence_status(row: dict[str, Any]) -> tuple[str, int]:
    request = row.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"locked row {case_key(row)} has no request object")
    packet = request.get("evidence_packet")
    if not isinstance(packet, dict):
        raise ValueError(f"locked row {case_key(row)} has no evidence_packet")
    retrieved = packet.get("retrieved_evidence")
    count = len(retrieved) if isinstance(retrieved, list) else 0
    degraded = bool(packet.get("degraded"))
    if row.get("variant") == "source_outage":
        if count != 0 or not degraded:
            raise ValueError("source_outage row must be degraded with zero retrieved evidence")
        return "证据来源当前不可用，未返回可核验的检索证据。", count
    return f"证据来源可用；锁定记录中包含{count}条检索证据。", count


def _allowed_evidence_text(row: dict[str, Any]) -> str:
    """Serialize only kind/content; source IDs and metadata stay excluded."""

    packet = row["request"]["evidence_packet"]
    retrieved = packet.get("retrieved_evidence")
    if not isinstance(retrieved, list) or not retrieved:
        return "（当前无可用检索证据）"
    items: list[str] = []
    for index, item in enumerate(retrieved, start=1):
        if not isinstance(item, dict):
            raise ValueError("retrieved_evidence item must be an object")
        kind = str(item.get("kind") or "未分类").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            raise ValueError("retrieved_evidence item has empty content")
        items.append(f"证据{index}（{kind}）：{content}")
    return "\n".join(items)


def _build_message(row: dict[str, Any], paired_case_material: str) -> str:
    request = row["request"]
    source_text = request.get("input_text")
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError(f"locked row {case_key(row)} has empty request.input_text")
    status, _ = _evidence_status(row)
    evidence_qualification = (
        "上述材料来自锁定检索证据，可作为当前回答依据。"
        if row.get("variant") == "clean"
        else "来源不可用；上述材料只用于固定同一病例事实，不得视为已核验证据。"
    )
    return (
        "去标识化、结构化真实医疗记录；仅评测，不训练。\n"
        f"【同病例材料】\n{paired_case_material}\n"
        f"【证据资格】{status}{evidence_qualification}\n"
        f"【统一任务】{GENERIC_TASK}"
    )


def prepare(args: argparse.Namespace) -> dict[str, Path]:
    locked = args.locked.expanduser().resolve()
    if not locked.is_file():
        raise FileNotFoundError(locked)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tasks": output_dir / "three_role_tasks.private.jsonl",
        "key": output_dir / "three_role_selection_key.private.jsonl",
        "private_manifest": output_dir / "three_role_plan.private.json",
        "public_manifest": output_dir / "three_role_plan.public.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing plan files; use a new directory or --overwrite"
        )

    variants = _parse_values(args.variant, DEFAULT_VARIANTS)
    unknown_variants = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown_variants:
        raise ValueError(f"unsupported variants for the controlled study: {unknown_variants}")
    if set(variants) != set(DEFAULT_VARIANTS):
        raise ValueError("the primary protocol requires both clean and source_outage")

    rows = read_jsonl(locked)
    tracks = sorted({str(row.get("evaluation_track")) for row in rows if row.get("evaluation_track")})
    track = args.track
    if track is None:
        if len(tracks) != 1:
            raise ValueError("--track is required when the locked file has multiple tracks")
        track = tracks[0]
    normalized_track = str(track).lower()
    if "structured" not in normalized_track or "raw" in normalized_track:
        raise ValueError(
            "the preregistered three-role study uses the Structured interface view; "
            "Raw strict is not an additional patient cohort"
        )
    rows = [row for row in rows if row.get("evaluation_track") == track and row.get("variant") in variants]
    if not rows:
        raise ValueError("no rows remain after track/variant filtering")

    by_family_variant: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (family_key(row), str(row.get("variant")))
        if key in by_family_variant:
            raise ValueError(f"duplicate family/variant row: {key}")
        by_family_variant[key] = row
    clean_rows = [row for row in rows if row.get("variant") == "clean"]
    selected, available_strata, selected_strata = _stratified_sample(
        clean_rows, args.sample_size, args.sample_seed
    )
    for family in selected:
        missing = [variant for variant in variants if (family, variant) not in by_family_variant]
        if missing:
            raise ValueError(f"selected family lacks variants: {missing}")
        input_values = {
            str(by_family_variant[(family, variant)]["request"].get("input_text", ""))
            for variant in variants
        }
        if len(input_values) != 1:
            raise ValueError("clean/source_outage must share exactly the same case facts")

    locked_hash = sha256_file(locked)
    secret = hashlib.sha256(
        json.dumps(
            {
                "locked_sha256": locked_hash,
                "sample_seed": args.sample_seed,
                "task_seed": args.task_seed,
                "selected": selected,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).digest()
    arms = ["role_aware"]
    if args.include_prompt_ablation:
        arms.append("role_agnostic_prompt_ablation")

    tasks: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for family in selected:
        family_token = "F-" + opaque_token(secret, "family", family)
        clean = by_family_variant[(family, "clean")]
        paired_case_material = _allowed_evidence_text(clean)
        paired_case_material_hash = hashlib.sha256(
            paired_case_material.encode("utf-8")
        ).hexdigest()
        key_rows.append(
            {
                "family_token": family_token,
                "source_family_private": family,
                "clean_case_private": case_key(clean),
                "modality": clean.get("modality", "unknown"),
            }
        )
        for variant in variants:
            row = by_family_variant[(family, variant)]
            condition = OUTPUT_CONDITION[variant]
            status_text, evidence_count = _evidence_status(row)
            evidence_text = _allowed_evidence_text(row)
            evidence_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
            original_request = str(row["request"]["input_text"]).strip()
            for arm in arms:
                for role in ROLES:
                    task_id = "T-" + opaque_token(
                        secret, "task", f"{family}|{condition}|{arm}|{role}"
                    )
                    user_message = _build_message(row, paired_case_material)
                    if len(user_message) > args.max_message_chars:
                        raise ValueError(
                            f"prepared message exceeds {args.max_message_chars} characters; "
                            "do not silently truncate real-case facts"
                        )
                    tasks.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "task_id": task_id,
                            "family_token": family_token,
                            "role": role,
                            "variant": condition,
                            "source_variant_private": variant,
                            "arm": arm,
                            "modality": row.get("modality", "unknown"),
                            "source_case_private": case_key(row),
                            "source_text_private": paired_case_material,
                            "paired_case_material_sha256": paired_case_material_hash,
                            "original_request_sha256": hashlib.sha256(
                                original_request.encode("utf-8")
                            ).hexdigest(),
                            "evidence_status_private": status_text,
                            "retrieved_evidence_count_private": evidence_count,
                            "evidence_content_sha256": evidence_hash,
                            "shared_payload_sha256": hashlib.sha256(
                                user_message.encode("utf-8")
                            ).hexdigest(),
                            "system_prompt_private": (
                                ROLE_SYSTEM_PROMPTS[role]
                                if arm == "role_aware"
                                else GENERIC_SYSTEM_PROMPT
                            ),
                            "user_message_private": user_message,
                        }
                    )

    tasks.sort(
        key=lambda row: hashlib.sha256(
            f"{args.task_seed}\x00{row['task_id']}".encode("utf-8")
        ).hexdigest()
    )
    payload_audit: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    for task in tasks:
        payload_audit[(task["family_token"], task["variant"], task["arm"])].add(
            (task["shared_payload_sha256"], task["user_message_private"])
        )
    nonidentical = [key for key, values in payload_audit.items() if len(values) != 1]
    if nonidentical:
        raise RuntimeError(
            "same-family/state/arm user payloads are not byte-identical across roles"
        )
    write_jsonl(paths["tasks"], tasks)
    write_jsonl(paths["key"], key_rows)
    selection_digest = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
    private_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PLAN_PREPARED_NO_MODEL_OUTPUTS",
        "data_source": (
            "数据提供方提供的真实医疗业务甲状腺记录；已去标识化并结构化；"
            "仅用于独立外部评测，不进入训练"
        ),
        "track": track,
        "cohort_interpretation": (
            "Structured与Raw strict是同批真实记录按不同资格形成的接口视图，不相加；"
            "本实验仅从Structured视图按family抽样"
        ),
        "sampling": {
            "strategy": "proportional_stratified_without_replacement_by_modality_then_hash_rank",
            "performance_based_selection": False,
            "sample_seed": args.sample_seed,
            "available_strata": available_strata,
            "selected_strata": selected_strata,
            "selected_count": len(selected),
            "selection_digest": selection_digest,
        },
        "task_order_seed": args.task_seed,
        "roles": list(ROLES),
        "locked_source_variants": variants,
        "evaluation_conditions": list(OUTPUT_CONDITION.values()),
        "arms": arms,
        "planned_task_count": len(tasks),
        "expected_primary_response_count": args.sample_size * 2 * 3,
        "shared_payload_audit": {
            "group_count": len(payload_audit),
            "three_role_byte_identity_verified": True,
            "group_digests": sorted(next(iter(values))[0] for values in payload_audit.values()),
        },
        "evidence_serialization": {
            "allowlist": ["retrieved_evidence.kind", "retrieved_evidence.content"],
            "excluded": ["source_id", "metadata", "all other evidence fields"],
            "content_hashes": sorted({task["evidence_content_sha256"] for task in tasks}),
            "paired_case_material_control": (
                "每个family用clean的allowlisted kind+content锁定病例事实；source_outage的"
                "retrieved_evidence仍为0，同一材料仅作为未核验病例陈述，不能当作证据"
            ),
            "derived_from_clean": True,
            "original_source_outage_verified_empty": True,
            "causal_treatment": (
                "在病例材料文本保持不变时，将检索证据资格从可核验置为不可用；"
                "输出条件命名为evidence_unavailable_counterfactual，不声称直接复现原始空输入"
            ),
            "unique_case_material_count": len(
                {task["paired_case_material_sha256"] for task in tasks}
            ),
            "unique_condition_payload_count": len(
                {task["shared_payload_sha256"] for task in tasks}
            ),
            "original_request_handling": (
                "原锁定input_text为医生轨任务/格式说明，未并入三角色共同用户消息；"
                "仅记录哈希，避免把医生式任务措辞混入角色对照"
            ),
        },
        "locked_sha256": locked_hash,
        "files": {
            "tasks": paths["tasks"].name,
            "selection_key": paths["key"].name,
        },
    }
    write_json(paths["private_manifest"], private_manifest)
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PLAN_PREPARED_NO_MODEL_OUTPUTS",
        "data_provenance": (
            "数据提供方提供的真实医疗业务甲状腺记录；经去标识化与结构化处理；"
            "仅用于独立外部评测，不进入训练"
        ),
        "track": "Structured接口视图",
        "cohort_note": "Structured与Raw strict属于同批记录的不同接口视图，本实验未将二者相加",
        "selection": {
            "method": "按模态比例分层、固定种子、family级无放回抽样",
            "performance_based_selection": False,
            "available_record_count": sum(available_strata.values()),
            "selected_record_count": len(selected),
            "available_modality_counts": available_strata,
            "selected_modality_counts": selected_strata,
            "selection_digest": selection_digest,
        },
        "design": {
            "roles": ["医生", "患者", "学生"],
            "conditions": ["clean", "evidence_unavailable_counterfactual"],
            "primary_arm": "真实三角色路由与角色指令",
            "generic_instruction_ablation_included": args.include_prompt_ablation,
            "planned_task_count": len(tasks),
            "expected_primary_answer_count": args.sample_size * 2 * 3,
            "shared_user_payload_byte_identity_verified": True,
            "shared_payload_group_count": len(payload_audit),
            "evidence_serialization": "仅纳入kind与content；排除source_id、metadata及其他字段",
            "evidence_unavailable_counterfactual": (
                "从同family clean的allowlisted事实材料派生；保留文本但撤销检索证据资格。"
                "原始source_outage已核验为0条检索证据；本条件不声称直接复现其空输入"
            ),
            "unique_case_material_count": len(
                {task["paired_case_material_sha256"] for task in tasks}
            ),
            "unique_condition_payload_count": len(
                {task["shared_payload_sha256"] for task in tasks}
            ),
        },
        "inference_result_available": False,
        "privacy": "公开文件仅含设计与计数，不含病例、回答、可逆标识或源标识",
    }
    assert_public_aggregate(public_manifest)
    write_json(paths["public_manifest"], public_manifest)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked", type=Path, required=True, help="locked Structured private JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="PRIVATE working directory")
    parser.add_argument("--track", default=None)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--sample-seed", type=int, default=20260824)
    parser.add_argument("--task-seed", type=int, default=20260825)
    parser.add_argument("--variant", action="append")
    parser.add_argument("--max-message-chars", type=int, default=2000)
    parser.add_argument(
        "--include-prompt-ablation",
        action="store_true",
        help="add a role-instruction ablation; this is not a product or prior-work baseline",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    paths = prepare(_parser().parse_args())
    print(
        json.dumps(
            {"status": "PLAN_PREPARED_NO_MODEL_OUTPUTS", "files": {k: str(v) for k, v in paths.items()}},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
