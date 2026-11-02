#!/usr/bin/env python3
"""Build a private review packet and summarize the controlled three-role study.

Automatic checks are lexical audit signals, not clinical judgments.  The final
public result requires two complete human rating rows per generated answer and
uses family-cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from .three_role_eval_common import (
        ROLE_LABELS,
        ROLES,
        assert_public_aggregate,
        automatic_signals,
        canonical_json_sha256,
        cohen_kappa,
        extract_fact_anchors,
        family_cluster_bootstrap,
        group_mean,
        quadratic_weighted_kappa,
        read_csv,
        read_jsonl,
        rows_by,
        sha256_file,
        write_csv,
        write_json,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - direct script execution
    from three_role_eval_common import (  # type: ignore
        ROLE_LABELS,
        ROLES,
        assert_public_aggregate,
        automatic_signals,
        canonical_json_sha256,
        cohen_kappa,
        extract_fact_anchors,
        family_cluster_bootstrap,
        group_mean,
        quadratic_weighted_kappa,
        read_csv,
        read_jsonl,
        rows_by,
        sha256_file,
        write_csv,
        write_json,
        write_jsonl,
    )


ORDINAL_FIELDS = {
    "fact_fidelity_0_4": (0, 4),
    "role_fit_0_4": (0, 4),
    "evidence_handling_0_4": (0, 4),
}
BINARY_PASS_FIELDS = (
    "permission_boundary_pass_0_1",
    "overall_usable_0_1",
)
BINARY_FAILURE_FIELDS = (
    "unsupported_major_claim_0_1",
    "doctor_language_leak_0_1_or_na",
)
CONDITIONAL_PASS_FIELD = "degradation_correct_0_1_or_na"
RATING_COLUMNS = [
    "reviewer_id",
    "task_id",
    "intended_role",
    "condition",
    "arm",
    *ORDINAL_FIELDS,
    *BINARY_PASS_FIELDS,
    *BINARY_FAILURE_FIELDS,
    CONDITIONAL_PASS_FIELD,
    "notes_private",
]
DEGRADED_CONDITION = "evidence_unavailable_counterfactual"
BUNDLE_RATING_COLUMNS = [
    "reviewer_id",
    "bundle_id",
    "condition",
    "arm",
    "guessed_doctor_candidate",
    "guessed_patient_candidate",
    "guessed_student_candidate",
    "shared_core_fact_consistency_0_1",
    "cross_role_contradiction_0_1",
    "patient_doctor_style_leakage_0_1",
    "notes_private",
]


def _bundle_id(family_token: str, variant: str, arm: str) -> str:
    digest = hashlib.sha256(f"{family_token}|{variant}|{arm}".encode("utf-8")).hexdigest()
    return "B-" + digest[:24]


def _candidate_role_map(bundle_id: str, blinding_secret: bytes) -> dict[str, str]:
    """Create a mapping that cannot be reconstructed from the visible bundle ID."""

    ordered = sorted(
        ROLES,
        key=lambda role: hmac.new(
            blinding_secret,
            f"{bundle_id}|{role}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest(),
    )
    return {candidate: role for candidate, role in zip(("A", "B", "C"), ordered)}


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("task_id") or "")
        if not key:
            raise ValueError(f"{label} row lacks task_id")
        if key in result:
            raise ValueError(f"duplicate task_id in {label}: {key}")
        result[key] = row
    return result


def _validate_task_response_contract(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate the frozen factorial grid and bind every answer to its exact task."""

    task_index = _index(tasks, "tasks")
    response_index = _index(responses, "responses")
    if set(task_index) != set(response_index):
        raise ValueError(
            f"task/response coverage mismatch: missing={len(set(task_index)-set(response_index))}, "
            f"extra={len(set(response_index)-set(task_index))}"
        )

    expected_conditions = {"clean", DEGRADED_CONDITION}
    for task_id, task in task_index.items():
        if task.get("role") not in ROLES:
            raise ValueError(f"task has unsupported role: {task_id}")
        if task.get("variant") not in expected_conditions:
            raise ValueError(f"task has unsupported controlled condition: {task_id}")
        message = str(task.get("user_message_private") or "")
        source = str(task.get("source_text_private") or "")
        if not message.strip() or not source.strip():
            raise ValueError(f"task has empty shared input/case facts: {task_id}")
        observed_payload_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if task.get("shared_payload_sha256") != observed_payload_hash:
            raise ValueError(f"task shared payload hash mismatch: {task_id}")

        response = response_index[task_id]
        for field in ("family_token", "role", "variant", "arm"):
            if response.get(field) != task.get(field):
                raise ValueError(f"response/task {field} mismatch: {task_id}")
        if response.get("task_payload_sha256") != canonical_json_sha256(task):
            raise ValueError(f"response is not bound to the exact frozen task: {task_id}")
        if response.get("status") != "success":
            raise ValueError(f"generation task is not successful: {task_id}")
        if not str(response.get("display_response_private") or "").strip():
            raise ValueError(f"successful generation has an empty visible answer: {task_id}")

    for key, rows in rows_by(tasks, "family_token", "variant", "arm").items():
        roles = [str(row.get("role")) for row in rows]
        if len(rows) != len(ROLES) or set(roles) != set(ROLES):
            raise ValueError(f"factorial group must contain exactly one answer per role: {key}")
        if len({str(row["user_message_private"]) for row in rows}) != 1:
            raise ValueError(f"three roles do not share byte-identical user input: {key}")
        if len({str(row["source_text_private"]) for row in rows}) != 1:
            raise ValueError(f"three roles do not share identical case facts: {key}")

    for key, rows in rows_by(tasks, "family_token", "arm").items():
        if len(rows) != len(ROLES) * len(expected_conditions):
            raise ValueError(f"family/arm does not contain the complete paired grid: {key}")
        if {str(row.get("variant")) for row in rows} != expected_conditions:
            raise ValueError(f"family/arm condition pairing is incomplete: {key}")
        if len({str(row["source_text_private"]) for row in rows}) != 1:
            raise ValueError(f"paired conditions changed the locked case facts: {key}")
        paired_hashes = {
            str(row.get("paired_case_material_sha256") or "") for row in rows
        }
        if "" in paired_hashes or len(paired_hashes) != 1:
            raise ValueError(f"paired case-material binding is missing or inconsistent: {key}")

    return task_index, response_index


def _reviewers(values: list[str] | None) -> list[str]:
    raw = values or ["reviewer_1", "reviewer_2"]
    result: list[str] = []
    for value in raw:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    if len(result) != 2 or len(set(result)) != 2:
        raise ValueError("exactly two distinct reviewer IDs are required")
    return result


def _runtime_audit_fields(response: dict[str, Any]) -> dict[str, int | None]:
    """Extract non-content runtime diagnostics from the PRIVATE API payload."""

    value = response.get("raw_payload_private")
    if not isinstance(value, dict):
        return {
            "runtime_packet_present": None,
            "runtime_degraded": None,
            "runtime_fallback_used": None,
            "runtime_retrieved_evidence_count": None,
            "runtime_trace_present": None,
        }
    if isinstance(value.get("data"), dict):
        value = value["data"]
    packet = value.get("evidence_packet") if isinstance(value, dict) else None
    retrieved = packet.get("retrieved_evidence") if isinstance(packet, dict) else None
    return {
        "runtime_packet_present": int(isinstance(packet, dict)),
        "runtime_degraded": (
            int(bool(value.get("degraded"))) if "degraded" in value else None
        ),
        "runtime_fallback_used": (
            int(bool(value.get("fallback_used"))) if "fallback_used" in value else None
        ),
        "runtime_retrieved_evidence_count": (
            len(retrieved) if isinstance(retrieved, list) else None
        ),
        "runtime_trace_present": int(bool(value.get("trace_id"))),
    }


def _safe_paths(output_dir: Path, names: list[str], overwrite: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in names}
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite review/result files; use a new directory")
    return paths


def _input_paths(value: Path | list[Path], label: str) -> list[Path]:
    raw = value if isinstance(value, list) else [value]
    if not raw:
        raise ValueError(f"at least one {label} file is required")
    paths = [Path(item).expanduser().resolve() for item in raw]
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate {label} file")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def prepare_review(args: argparse.Namespace) -> dict[str, Path]:
    tasks_path = args.tasks.expanduser().resolve()
    responses_path = args.responses.expanduser().resolve()
    tasks = read_jsonl(tasks_path)
    responses = read_jsonl(responses_path)
    task_index, response_index = _validate_task_response_contract(tasks, responses)
    reviewers = _reviewers(args.reviewers)
    paths = _safe_paths(
        args.output_dir.expanduser().resolve(),
        [
            "three_role_review_packet.private.jsonl",
            "three_role_expert_ratings.private.csv",
            "three_role_expert_ratings.reviewer_1.private.csv",
            "three_role_expert_ratings.reviewer_2.private.csv",
            "three_role_blind_bundle_packet.private.jsonl",
            "three_role_blind_bundle_key.private.jsonl",
            "three_role_bundle_ratings.private.csv",
            "three_role_bundle_ratings.reviewer_1.private.csv",
            "three_role_bundle_ratings.reviewer_2.private.csv",
            "three_role_automatic_signals.private.csv",
            "three_role_automatic_summary.public.json",
            "three_role_automatic_summary.public.md",
            "three_role_review_manifest.private.json",
            "three_role_review_manifest.public.json",
        ],
        args.overwrite,
    )

    packet_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for task_id in sorted(task_index):
        task = task_index[task_id]
        response = response_index[task_id]
        response_text = str(response.get("display_response_private") or "")
        source_text = str(task.get("source_text_private") or "")
        packet_rows.append(
            {
                "task_id": task_id,
                "intended_role": ROLE_LABELS[str(task["role"])],
                "condition": task["variant"],
                "arm": task["arm"],
                "modality": task.get("modality", "unknown"),
                "evidence_status_private": task.get("evidence_status_private", ""),
                "shared_case_facts_private": source_text,
                "shared_user_payload_private": task.get("user_message_private", ""),
                "generated_answer_private": response_text,
                "review_instruction": (
                    "只评价可见答案与给定材料；不推测隐藏思维过程。"
                    "重点核对事实、角色权限、患者端诊疗话术泄漏与证据不足降级。"
                ),
            }
        )
        signals = automatic_signals(
            str(task.get("user_message_private") or source_text),
            response_text,
            str(task["role"]),
            str(task["variant"]),
        )
        signal_rows.append(
            {
                "task_id": task_id,
                "family_token": task["family_token"],
                "role": task["role"],
                "variant": task["variant"],
                "arm": task["arm"],
                "modality": task.get("modality", "unknown"),
                **signals,
                **_runtime_audit_fields(response),
            }
        )

    grouped = rows_by(signal_rows, "family_token", "variant", "arm")
    for rows in grouped.values():
        if {str(row["role"]) for row in rows} != set(ROLES):
            raise ValueError("each family/state/arm must contain all three roles")
        anchor_sets = {
            str(row["role"]): extract_fact_anchors(
                str(response_index[str(row["task_id"])].get("display_response_private") or "")
            )
            for row in rows
        }
        intersection = set.intersection(*(anchor_sets[role] for role in ROLES))
        union = set.union(*(anchor_sets[role] for role in ROLES))
        for row in rows:
            row["three_role_common_anchor_count"] = len(intersection)
            row["three_role_anchor_union_count"] = len(union)
            row["three_role_anchor_jaccard"] = len(intersection) / len(union) if union else None

    write_jsonl(paths["three_role_review_packet.private.jsonl"], packet_rows)
    rating_rows = [
        {
            "reviewer_id": reviewer,
            "task_id": task["task_id"],
            "intended_role": ROLE_LABELS[str(task["role"])],
            "condition": task["variant"],
            "arm": task["arm"],
            **{field: "" for field in ORDINAL_FIELDS},
            **{field: "" for field in BINARY_PASS_FIELDS},
            **{field: "" for field in BINARY_FAILURE_FIELDS},
            CONDITIONAL_PASS_FIELD: "",
            "notes_private": "",
        }
        for reviewer in reviewers
        for task in sorted(tasks, key=lambda row: str(row["task_id"]))
    ]
    write_csv(paths["three_role_expert_ratings.private.csv"], RATING_COLUMNS, rating_rows)
    for reviewer_index, reviewer in enumerate(reviewers, start=1):
        write_csv(
            paths[f"three_role_expert_ratings.reviewer_{reviewer_index}.private.csv"],
            RATING_COLUMNS,
            [row for row in rating_rows if row["reviewer_id"] == reviewer],
        )

    bundle_packet_rows: list[dict[str, Any]] = []
    bundle_key_rows: list[dict[str, Any]] = []
    bundle_rating_rows: list[dict[str, Any]] = []
    blinding_secret = secrets.token_bytes(32)
    grouped_tasks = rows_by(tasks, "family_token", "variant", "arm")
    for (family_token, variant, arm), rows in sorted(grouped_tasks.items()):
        if {str(row["role"]) for row in rows} != set(ROLES):
            raise ValueError("each bundle must contain doctor, patient and student")
        bundle_id = _bundle_id(str(family_token), str(variant), str(arm))
        role_to_task = {str(row["role"]): row for row in rows}
        candidate_to_role = _candidate_role_map(bundle_id, blinding_secret)
        candidates = {
            candidate: str(
                response_index[str(role_to_task[role]["task_id"])].get(
                    "display_response_private"
                )
                or ""
            )
            for candidate, role in candidate_to_role.items()
        }
        source_values = {str(row["source_text_private"]) for row in rows}
        evidence_values = {str(row.get("evidence_status_private") or "") for row in rows}
        payload_values = {str(row.get("user_message_private") or "") for row in rows}
        if len(source_values) != 1 or len(evidence_values) != 1 or len(payload_values) != 1:
            raise ValueError("bundle roles do not share identical case facts/evidence status")
        bundle_packet_rows.append(
            {
                "bundle_id": bundle_id,
                "condition": variant,
                "arm": arm,
                "shared_case_facts_private": next(iter(source_values)),
                "evidence_status_private": next(iter(evidence_values)),
                "shared_user_payload_private": next(iter(payload_values)),
                "anonymous_candidates_private": candidates,
                "round_1_instruction": (
                    "候选A/B/C的真实角色已隐藏。先独立完成医生/患者/学生的一一映射，"
                    "再评价核心事实是否一致、是否出现跨角色矛盾及患者式候选是否泄漏诊疗话术。"
                    "不得查看角色密钥或第二轮表格。"
                ),
            }
        )
        bundle_key_rows.append(
            {
                "bundle_id": bundle_id,
                "family_token": family_token,
                "condition": variant,
                "arm": arm,
                "candidate_role_mapping_private": candidate_to_role,
            }
        )
        for reviewer in reviewers:
            bundle_rating_rows.append(
                {
                    "reviewer_id": reviewer,
                    "bundle_id": bundle_id,
                    "condition": variant,
                    "arm": arm,
                    "guessed_doctor_candidate": "",
                    "guessed_patient_candidate": "",
                    "guessed_student_candidate": "",
                    "shared_core_fact_consistency_0_1": "",
                    "cross_role_contradiction_0_1": "",
                    "patient_doctor_style_leakage_0_1": "",
                    "notes_private": "",
                }
            )
    write_jsonl(paths["three_role_blind_bundle_packet.private.jsonl"], bundle_packet_rows)
    write_jsonl(paths["three_role_blind_bundle_key.private.jsonl"], bundle_key_rows)
    write_csv(
        paths["three_role_bundle_ratings.private.csv"],
        BUNDLE_RATING_COLUMNS,
        bundle_rating_rows,
    )
    for reviewer_index, reviewer in enumerate(reviewers, start=1):
        write_csv(
            paths[f"three_role_bundle_ratings.reviewer_{reviewer_index}.private.csv"],
            BUNDLE_RATING_COLUMNS,
            [row for row in bundle_rating_rows if row["reviewer_id"] == reviewer],
        )
    signal_fields = list(signal_rows[0]) if signal_rows else []
    write_csv(paths["three_role_automatic_signals.private.csv"], signal_fields, signal_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for (arm, variant, role), rows in sorted(rows_by(signal_rows, "arm", "variant", "role").items()):
        aggregate_rows.append(
            {
                "arm": arm,
                "condition": variant,
                "role": ROLE_LABELS[str(role)],
                "count": len(rows),
                "completion_rate": group_mean(rows, "response_present"),
                "case_anchor_recall_mean": group_mean(rows, "anchor_recall"),
                "unsupported_numeric_count_mean": group_mean(rows, "unsupported_numeric_count"),
                "degradation_phrase_rate": group_mean(rows, "degradation_signal"),
                "role_structure_signal_rate": group_mean(
                    rows,
                    {
                        "doctor": "doctor_evidence_review_signal",
                        "patient": "patient_plain_language_signal",
                        "student": "student_teaching_structure_signal",
                    }[str(role)],
                ),
                "boundary_phrase_rate": (
                    group_mean(rows, "patient_boundary_signal") if role == "patient" else None
                ),
                "diagnostic_language_flag_rate": (
                    group_mean(rows, "patient_diagnostic_language_signal")
                    if role == "patient"
                    else None
                ),
                "three_role_anchor_jaccard_mean": group_mean(rows, "three_role_anchor_jaccard"),
                "runtime_packet_rate": group_mean(rows, "runtime_packet_present"),
                "runtime_degraded_rate": group_mean(rows, "runtime_degraded"),
                "runtime_fallback_rate": group_mean(rows, "runtime_fallback_used"),
                "runtime_retrieved_evidence_count_mean": group_mean(
                    rows, "runtime_retrieved_evidence_count"
                ),
                "runtime_trace_rate": group_mean(rows, "runtime_trace_present"),
            }
        )
    public_summary = {
        "schema_version": "care_opd.competition.three_role_automatic.v1",
        "status": "AUTOMATIC_AUDIT_ONLY_NOT_CLINICAL_RESULT",
        "record_count": len({str(row["family_token"]) for row in signal_rows}),
        "generated_output_count": len(signal_rows),
        "data_provenance": (
            "数据提供方提供的真实医疗业务记录，经去标识化与结构化处理；"
            "仅用于独立外部评测，不进入训练"
        ),
        "rows": aggregate_rows,
        "interpretation_limit": (
            "词汇锚点、数字与边界短语仅用于审计和定位，不等同于医学事实正确性、"
            "角色适配性或临床安全结论；最终指标须由两位评审先完成A/B/C盲态三角色映射，"
            "锁定后再完成显示目标角色的逐答独立评分。运行时检索证据计数只描述后端实际调用，"
            "不能替代对病例事实是否获得有效证据资格的人工判断"
        ),
    }
    assert_public_aggregate(public_summary)
    write_json(paths["three_role_automatic_summary.public.json"], public_summary)
    md_lines = [
        "# 三角色真实病例实验：自动审计摘要",
        "",
        "> 状态：仅完成词汇与格式层面的自动审计，不构成临床结果。",
        "",
        f"- 去标识化真实记录：{public_summary['record_count']}组；",
        f"- 已生成答案：{public_summary['generated_output_count']}条；",
        "- 最终结论须等待两位评审完成事实、角色权限与证据降级评分。",
        "",
        "| 条件 | 角色 | 完成率 | 病例锚点召回 | 无依据数字均值 | 降级短语率 | 角色结构信号率 | 边界短语率 | 诊疗话术标记率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def _rate(value: Any) -> str:
        return "—" if value is None else f"{float(value):.1%}"

    for row in aggregate_rows:
        md_lines.append(
            f"| {row['condition']} | {row['role']} | {_rate(row['completion_rate'])} | "
            f"{_rate(row['case_anchor_recall_mean'])} | "
            f"{float(row['unsupported_numeric_count_mean'] or 0):.2f} | "
            f"{_rate(row['degradation_phrase_rate'])} | "
            f"{_rate(row['role_structure_signal_rate'])} | "
            f"{_rate(row['boundary_phrase_rate'])} | "
            f"{_rate(row['diagnostic_language_flag_rate'])} |"
        )
    md_lines.extend(
        [
            "",
            "| 条件 | 角色 | 运行证据包率 | 后端降级率 | 后端回退率 | 检索证据条数均值 | Trace率 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate_rows:
        evidence_count = row["runtime_retrieved_evidence_count_mean"]
        md_lines.append(
            f"| {row['condition']} | {row['role']} | {_rate(row['runtime_packet_rate'])} | "
            f"{_rate(row['runtime_degraded_rate'])} | {_rate(row['runtime_fallback_rate'])} | "
            f"{'—' if evidence_count is None else f'{float(evidence_count):.2f}'} | "
            f"{_rate(row['runtime_trace_rate'])} |"
        )
    md_lines.extend(
        [
            "",
            "自动信号用于发现异常数字、患者端诊疗式话术和证据不足时的降级短语，不能替代医学判断。",
            "运行时检索计数只说明后端实际调用，不能代替对病例证据资格的人工判断。",
        ]
    )
    paths["three_role_automatic_summary.public.md"].write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    frozen_hashes = {
        "role_shown_packet": sha256_file(
            paths["three_role_review_packet.private.jsonl"]
        ),
        "individual_rating_template": sha256_file(
            paths["three_role_expert_ratings.private.csv"]
        ),
        "individual_rating_template_reviewer_1": sha256_file(
            paths["three_role_expert_ratings.reviewer_1.private.csv"]
        ),
        "individual_rating_template_reviewer_2": sha256_file(
            paths["three_role_expert_ratings.reviewer_2.private.csv"]
        ),
        "blind_triplet_packet": sha256_file(
            paths["three_role_blind_bundle_packet.private.jsonl"]
        ),
        "blind_role_key": sha256_file(
            paths["three_role_blind_bundle_key.private.jsonl"]
        ),
        "bundle_rating_template": sha256_file(
            paths["three_role_bundle_ratings.private.csv"]
        ),
        "bundle_rating_template_reviewer_1": sha256_file(
            paths["three_role_bundle_ratings.reviewer_1.private.csv"]
        ),
        "bundle_rating_template_reviewer_2": sha256_file(
            paths["three_role_bundle_ratings.reviewer_2.private.csv"]
        ),
        "automatic_signal_rows": sha256_file(
            paths["three_role_automatic_signals.private.csv"]
        ),
        "automatic_aggregate_json": sha256_file(
            paths["three_role_automatic_summary.public.json"]
        ),
        "automatic_aggregate_markdown": sha256_file(
            paths["three_role_automatic_summary.public.md"]
        ),
    }
    private_manifest = {
        "schema_version": "care_opd.competition.three_role_review_manifest.v1",
        "status": "REVIEW_PACKAGE_FROZEN_PENDING_HUMAN_RATINGS",
        "task_plan_sha256": sha256_file(tasks_path),
        "model_output_sha256": sha256_file(responses_path),
        "record_count": len({str(row["family_token"]) for row in tasks}),
        "answer_count": len(responses),
        "bundle_count": len(bundle_packet_rows),
        "reviewer_ids": reviewers,
        "blinding": {
            "method": "HMAC-SHA256 with an ephemeral 256-bit secret",
            "secret_persisted": False,
            "mapping_recoverable_only_from_private_key_file": True,
        },
        "frozen_artifact_sha256": frozen_hashes,
    }
    write_json(paths["three_role_review_manifest.private.json"], private_manifest)
    public_manifest = {
        "schema_version": "care_opd.competition.three_role_review_manifest.v1",
        "status": "REVIEW_PACKAGE_FROZEN_PENDING_HUMAN_RATINGS",
        "record_count": private_manifest["record_count"],
        "answer_count": private_manifest["answer_count"],
        "blind_triplet_count": private_manifest["bundle_count"],
        "reviewer_count": len(reviewers),
        "frozen_plan_sha256": private_manifest["task_plan_sha256"],
        "model_output_sha256": private_manifest["model_output_sha256"],
        "blinding": (
            "A/B/C映射由一次性256位密钥生成，密钥不落盘；角色对应关系仅保存在隔离的PRIVATE key中"
        ),
        "frozen_artifact_sha256": frozen_hashes,
        "human_rating_result_available": False,
        "privacy": "公开清单只含计数、状态与不可逆哈希，不含病例、回答、任务标识或角色密钥",
    }
    assert_public_aggregate(public_manifest)
    write_json(paths["three_role_review_manifest.public.json"], public_manifest)
    return paths


def _parse_int(raw: str, field: str, low: int, high: int, allow_na: bool = False) -> int | None:
    text = str(raw or "").strip().lower()
    if allow_na and text in {"", "na", "n/a", "not_applicable", "不适用"}:
        return None
    if not text:
        raise ValueError(f"blank required rating: {field}")
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {raw!r}") from exc
    if not low <= value <= high:
        raise ValueError(f"{field}={value} outside [{low}, {high}]")
    return value


def _require_na(raw: str, field: str) -> None:
    text = str(raw or "").strip().lower()
    if text not in {"na", "n/a", "not_applicable", "不适用"}:
        raise ValueError(f"{field} is not applicable here and must be explicitly marked NA")


def _parse_ratings(
    rows: list[dict[str, str]], task_index: dict[str, dict[str, Any]], reviewers: list[str]
) -> dict[tuple[str, str], dict[str, int | None]]:
    parsed: dict[tuple[str, str], dict[str, int | None]] = {}
    for row in rows:
        reviewer = str(row.get("reviewer_id") or "")
        task_id = str(row.get("task_id") or "")
        if reviewer not in reviewers or task_id not in task_index:
            raise ValueError("ratings contain an unknown reviewer or task")
        key = (reviewer, task_id)
        if key in parsed:
            raise ValueError(f"duplicate rating row: {key}")
        task = task_index[task_id]
        expected_metadata = {
            "intended_role": ROLE_LABELS[str(task["role"])],
            "condition": str(task["variant"]),
            "arm": str(task["arm"]),
        }
        for field, expected_value in expected_metadata.items():
            if str(row.get(field) or "") != expected_value:
                raise ValueError(f"rating row metadata mismatch for {task_id}: {field}")
        values: dict[str, int | None] = {
            field: _parse_int(row.get(field, ""), field, low, high)
            for field, (low, high) in ORDINAL_FIELDS.items()
        }
        for field in BINARY_PASS_FIELDS:
            values[field] = _parse_int(row.get(field, ""), field, 0, 1)
        values["unsupported_major_claim_0_1"] = _parse_int(
            row.get("unsupported_major_claim_0_1", ""),
            "unsupported_major_claim_0_1",
            0,
            1,
        )
        if task["role"] == "patient":
            values["doctor_language_leak_0_1_or_na"] = _parse_int(
                row.get("doctor_language_leak_0_1_or_na", ""),
                "doctor_language_leak_0_1_or_na",
                0,
                1,
            )
        else:
            _require_na(
                row.get("doctor_language_leak_0_1_or_na", ""),
                "doctor_language_leak_0_1_or_na",
            )
            values["doctor_language_leak_0_1_or_na"] = None
        if task["variant"] == DEGRADED_CONDITION:
            values[CONDITIONAL_PASS_FIELD] = _parse_int(
                row.get(CONDITIONAL_PASS_FIELD, ""),
                CONDITIONAL_PASS_FIELD,
                0,
                1,
            )
        else:
            _require_na(row.get(CONDITIONAL_PASS_FIELD, ""), CONDITIONAL_PASS_FIELD)
            values[CONDITIONAL_PASS_FIELD] = None
        parsed[key] = values
    expected = {(reviewer, task_id) for reviewer in reviewers for task_id in task_index}
    if set(parsed) != expected:
        raise ValueError(f"incomplete ratings: missing={len(expected-set(parsed))}")
    return parsed


def _parse_bundle_ratings(
    rows: list[dict[str, str]],
    key_index: dict[str, dict[str, Any]],
    reviewers: list[str],
) -> dict[tuple[str, str], dict[str, int]]:
    parsed: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        reviewer = str(row.get("reviewer_id") or "")
        bundle_id = str(row.get("bundle_id") or "")
        if reviewer not in reviewers or bundle_id not in key_index:
            raise ValueError("bundle ratings contain an unknown reviewer or bundle")
        key = (reviewer, bundle_id)
        if key in parsed:
            raise ValueError(f"duplicate bundle rating: {key}")
        key_row = key_index[bundle_id]
        if str(row.get("condition") or "") != str(key_row.get("condition") or ""):
            raise ValueError(f"bundle rating condition mismatch: {bundle_id}")
        if str(row.get("arm") or "") != str(key_row.get("arm") or ""):
            raise ValueError(f"bundle rating arm mismatch: {bundle_id}")
        guesses = {
            "doctor": str(row.get("guessed_doctor_candidate") or "").strip().upper(),
            "patient": str(row.get("guessed_patient_candidate") or "").strip().upper(),
            "student": str(row.get("guessed_student_candidate") or "").strip().upper(),
        }
        if set(guesses.values()) != {"A", "B", "C"}:
            raise ValueError(
                "blind role guesses must be a one-to-one assignment of A, B and C"
            )
        candidate_to_role = key_index[bundle_id].get("candidate_role_mapping_private")
        if not isinstance(candidate_to_role, dict):
            raise ValueError("bundle key lacks candidate_role_mapping_private")
        role_to_candidate = {str(role): str(candidate) for candidate, role in candidate_to_role.items()}
        parsed[key] = {
            "shared_core_fact_consistency_0_1": int(
                _parse_int(
                    row.get("shared_core_fact_consistency_0_1", ""),
                    "shared_core_fact_consistency_0_1",
                    0,
                    1,
                )
            ),
            "cross_role_contradiction_0_1": int(
                _parse_int(
                    row.get("cross_role_contradiction_0_1", ""),
                    "cross_role_contradiction_0_1",
                    0,
                    1,
                )
            ),
            "patient_doctor_style_leakage_0_1": int(
                _parse_int(
                    row.get("patient_doctor_style_leakage_0_1", ""),
                    "patient_doctor_style_leakage_0_1",
                    0,
                    1,
                )
            ),
            "three_way_role_identification_0_1": int(
                all(guesses[role] == role_to_candidate.get(role) for role in ROLES)
            ),
        }
    expected = {
        (reviewer, bundle_id) for reviewer in reviewers for bundle_id in key_index
    }
    if set(parsed) != expected:
        raise ValueError(f"incomplete bundle ratings: missing={len(expected-set(parsed))}")
    return parsed


def summarize(args: argparse.Namespace) -> dict[str, Path]:
    tasks_path = args.tasks.expanduser().resolve()
    responses_path = args.responses.expanduser().resolve()
    ratings_paths = _input_paths(args.ratings, "individual rating")
    bundle_ratings_paths = _input_paths(args.bundle_ratings, "bundle rating")
    bundle_key_path = args.bundle_key.expanduser().resolve()
    review_manifest_path = args.review_manifest.expanduser().resolve()
    tasks = read_jsonl(tasks_path)
    responses = read_jsonl(responses_path)
    task_index, response_index = _validate_task_response_contract(tasks, responses)
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(review_manifest, dict):
        raise ValueError("review manifest must be a JSON object")
    if review_manifest.get("status") != "REVIEW_PACKAGE_FROZEN_PENDING_HUMAN_RATINGS":
        raise ValueError("review manifest has an unexpected status")
    if review_manifest.get("task_plan_sha256") != sha256_file(tasks_path):
        raise ValueError("review manifest/task plan hash mismatch")
    if review_manifest.get("model_output_sha256") != sha256_file(responses_path):
        raise ValueError("review manifest/model output hash mismatch")
    frozen_hashes = review_manifest.get("frozen_artifact_sha256")
    if not isinstance(frozen_hashes, dict) or frozen_hashes.get(
        "blind_role_key"
    ) != sha256_file(bundle_key_path):
        raise ValueError("review manifest/blind role key hash mismatch")
    reviewers = _reviewers(args.reviewers)
    parsed = _parse_ratings(
        [row for path in ratings_paths for row in read_csv(path)],
        task_index,
        reviewers,
    )
    key_rows = read_jsonl(bundle_key_path)
    key_index = {
        str(row.get("bundle_id") or ""): row
        for row in key_rows
        if str(row.get("bundle_id") or "")
    }
    if len(key_index) != len(key_rows):
        raise ValueError("bundle key has empty or duplicate bundle IDs")
    expected_bundle_ids = {
        _bundle_id(str(family), str(variant), str(arm))
        for family, variant, arm in rows_by(
            tasks, "family_token", "variant", "arm"
        )
    }
    if set(key_index) != expected_bundle_ids:
        raise ValueError("bundle key coverage does not match the frozen task groups")
    for row in key_index.values():
        mapping = row.get("candidate_role_mapping_private")
        if not isinstance(mapping, dict) or set(mapping) != {"A", "B", "C"} or set(mapping.values()) != set(ROLES):
            raise ValueError("bundle key mapping must be a one-to-one A/B/C role assignment")
    parsed_bundles = _parse_bundle_ratings(
        [row for path in bundle_ratings_paths for row in read_csv(path)],
        key_index,
        reviewers,
    )
    paths = _safe_paths(
        args.output_dir.expanduser().resolve(),
        [
            "three_role_consensus.private.csv",
            "three_role_bundle_consensus.private.csv",
            "three_role_results.public.json",
            "three_role_results.public.csv",
            "three_role_results.public.md",
        ],
        args.overwrite,
    )

    consensus_rows: list[dict[str, Any]] = []
    for task_id, task in task_index.items():
        left = parsed[(reviewers[0], task_id)]
        right = parsed[(reviewers[1], task_id)]
        row: dict[str, Any] = {
            "task_id": task_id,
            "family_token": task["family_token"],
            "role": task["role"],
            "variant": task["variant"],
            "arm": task["arm"],
            "modality": task.get("modality", "unknown"),
        }
        for field in ORDINAL_FIELDS:
            row[field] = mean([int(left[field]), int(right[field])])
        for field in BINARY_PASS_FIELDS:
            row[field] = min(int(left[field]), int(right[field]))
        row["unsupported_major_claim_0_1"] = max(
            int(left["unsupported_major_claim_0_1"]),
            int(right["unsupported_major_claim_0_1"]),
        )
        patient_values = [
            value
            for value in (
                left["doctor_language_leak_0_1_or_na"],
                right["doctor_language_leak_0_1_or_na"],
            )
            if value is not None
        ]
        row["doctor_language_leak_0_1_or_na"] = (
            max(int(value) for value in patient_values) if patient_values else None
        )
        degradation_values = [
            value
            for value in (left[CONDITIONAL_PASS_FIELD], right[CONDITIONAL_PASS_FIELD])
            if value is not None
        ]
        row[CONDITIONAL_PASS_FIELD] = (
            min(int(value) for value in degradation_values) if degradation_values else None
        )
        row["task_joint_compliance"] = int(
            row["fact_fidelity_0_4"] >= 3
            and row["role_fit_0_4"] >= 3
            and row["evidence_handling_0_4"] >= 3
            and row["permission_boundary_pass_0_1"] == 1
            and row["unsupported_major_claim_0_1"] == 0
            and row["overall_usable_0_1"] == 1
            and (row["doctor_language_leak_0_1_or_na"] in {None, 0})
        )
        row["task_correct_degradation"] = (
            int(
                row[CONDITIONAL_PASS_FIELD] == 1
                and row["permission_boundary_pass_0_1"] == 1
                and row["unsupported_major_claim_0_1"] == 0
            )
            if task["variant"] == DEGRADED_CONDITION
            else None
        )
        consensus_rows.append(row)
    write_csv(
        paths["three_role_consensus.private.csv"],
        list(consensus_rows[0]),
        sorted(consensus_rows, key=lambda row: str(row["task_id"])),
    )

    bundle_consensus: dict[str, dict[str, Any]] = {}
    for bundle_id, key_row in key_index.items():
        left = parsed_bundles[(reviewers[0], bundle_id)]
        right = parsed_bundles[(reviewers[1], bundle_id)]
        bundle_consensus[bundle_id] = {
            "bundle_id": bundle_id,
            "family_token": key_row["family_token"],
            "arm": key_row["arm"],
            "variant": key_row["condition"],
            "shared_core_fact_consistency_0_1": min(
                left["shared_core_fact_consistency_0_1"],
                right["shared_core_fact_consistency_0_1"],
            ),
            "cross_role_contradiction_0_1": max(
                left["cross_role_contradiction_0_1"],
                right["cross_role_contradiction_0_1"],
            ),
            "patient_doctor_style_leakage_0_1": max(
                left["patient_doctor_style_leakage_0_1"],
                right["patient_doctor_style_leakage_0_1"],
            ),
            "three_way_role_identification_0_1": min(
                left["three_way_role_identification_0_1"],
                right["three_way_role_identification_0_1"],
            ),
        }
    write_csv(
        paths["three_role_bundle_consensus.private.csv"],
        list(next(iter(bundle_consensus.values()))),
        [bundle_consensus[key] for key in sorted(bundle_consensus)],
    )

    agreement: list[dict[str, Any]] = []
    agreement_specs = {
        **{field: (low, high, "ordinal") for field, (low, high) in ORDINAL_FIELDS.items()},
        **{field: (0, 1, "binary") for field in BINARY_PASS_FIELDS},
        "unsupported_major_claim_0_1": (0, 1, "binary"),
        "doctor_language_leak_0_1_or_na": (0, 1, "binary"),
        CONDITIONAL_PASS_FIELD: (0, 1, "binary"),
    }
    for field, (low, high, kind) in agreement_specs.items():
        pairs = [
            (parsed[(reviewers[0], task_id)][field], parsed[(reviewers[1], task_id)][field])
            for task_id in sorted(task_index)
        ]
        pairs = [(int(a), int(b)) for a, b in pairs if a is not None and b is not None]
        left_values = [a for a, _ in pairs]
        right_values = [b for _, b in pairs]
        agreement.append(
            {
                "metric": field,
                "rating_pair_count": len(pairs),
                "exact_agreement": (
                    sum(a == b for a, b in pairs) / len(pairs) if pairs else None
                ),
                "kappa": (
                    quadratic_weighted_kappa(left_values, right_values, low, high)
                    if kind == "ordinal"
                    else cohen_kappa(left_values, right_values)
                ),
                "kappa_type": "quadratic_weighted" if kind == "ordinal" else "cohen",
            }
        )
    for field in (
        "shared_core_fact_consistency_0_1",
        "cross_role_contradiction_0_1",
        "patient_doctor_style_leakage_0_1",
        "three_way_role_identification_0_1",
    ):
        pairs = [
            (
                parsed_bundles[(reviewers[0], bundle_id)][field],
                parsed_bundles[(reviewers[1], bundle_id)][field],
            )
            for bundle_id in sorted(key_index)
        ]
        agreement.append(
            {
                "metric": field,
                "rating_pair_count": len(pairs),
                "exact_agreement": sum(a == b for a, b in pairs) / len(pairs),
                "kappa": cohen_kappa(
                    [a for a, _ in pairs], [b for _, b in pairs]
                ),
                "kappa_type": "cohen",
            }
        )

    role_rows: list[dict[str, Any]] = []
    for (arm, variant, role), rows in sorted(
        rows_by(consensus_rows, "arm", "variant", "role").items()
    ):
        role_rows.append(
            {
                "arm": arm,
                "condition": variant,
                "role": ROLE_LABELS[str(role)],
                "count": len(rows),
                "fact_fidelity_mean_0_4": group_mean(rows, "fact_fidelity_0_4"),
                "role_fit_mean_0_4": group_mean(rows, "role_fit_0_4"),
                "evidence_handling_mean_0_4": group_mean(rows, "evidence_handling_0_4"),
                "permission_boundary_pass_rate": group_mean(
                    rows, "permission_boundary_pass_0_1"
                ),
                "unsupported_major_claim_rate": group_mean(
                    rows, "unsupported_major_claim_0_1"
                ),
                "overall_usable_rate": group_mean(rows, "overall_usable_0_1"),
                "diagnostic_language_leak_rate": (
                    group_mean(rows, "doctor_language_leak_0_1_or_na")
                    if role == "patient"
                    else None
                ),
                "correct_degradation_rate": (
                    group_mean(rows, "task_correct_degradation")
                    if variant == DEGRADED_CONDITION
                    else None
                ),
            }
        )

    family_rows: list[dict[str, Any]] = []
    bundle_by_group = {
        (str(row["family_token"]), str(row["arm"]), str(row["variant"])): row
        for row in bundle_consensus.values()
    }
    for (family, arm, variant), rows in rows_by(
        consensus_rows, "family_token", "arm", "variant"
    ).items():
        if {str(row["role"]) for row in rows} != set(ROLES):
            raise ValueError("family/state/arm is missing one or more roles")
        bundle = bundle_by_group.get((str(family), str(arm), str(variant)))
        if bundle is None:
            raise ValueError("missing blind triplet rating for a family/state/arm group")
        per_role_joint = all(row["task_joint_compliance"] == 1 for row in rows)
        triplet_joint = (
            bundle["shared_core_fact_consistency_0_1"] == 1
            and bundle["cross_role_contradiction_0_1"] == 0
            and bundle["patient_doctor_style_leakage_0_1"] == 0
            and bundle["three_way_role_identification_0_1"] == 1
        )
        patient_rows = [row for row in rows if row["role"] == "patient"]
        if len(patient_rows) != 1:
            raise ValueError("family/state/arm must contain exactly one patient answer")
        actual_patient_nonleak = (
            patient_rows[0]["doctor_language_leak_0_1_or_na"] == 0
        )
        family_rows.append(
            {
                "family_token": family,
                "arm": arm,
                "variant": variant,
                "three_role_joint_compliance": int(
                    per_role_joint and triplet_joint
                ),
                "three_role_shared_fact_pass": int(
                    bundle["shared_core_fact_consistency_0_1"] == 1
                    and bundle["cross_role_contradiction_0_1"] == 0
                    and all(
                        row["fact_fidelity_0_4"] >= 3
                        and row["unsupported_major_claim_0_1"] == 0
                        for row in rows
                    )
                ),
                "three_way_role_identification": bundle[
                    "three_way_role_identification_0_1"
                ],
                "patient_doctor_style_nonleak": int(
                    actual_patient_nonleak
                    and bundle["patient_doctor_style_leakage_0_1"] == 0
                ),
                "three_role_joint_degradation": (
                    int(
                        all(row["task_correct_degradation"] == 1 for row in rows)
                        and bundle["shared_core_fact_consistency_0_1"] == 1
                        and bundle["cross_role_contradiction_0_1"] == 0
                        and bundle["patient_doctor_style_leakage_0_1"] == 0
                        and actual_patient_nonleak
                    )
                    if variant == DEGRADED_CONDITION
                    else None
                ),
            }
        )

    primary_rows: list[dict[str, Any]] = []
    for arm in sorted({str(row["arm"]) for row in family_rows}):
        clean = [row for row in family_rows if row["arm"] == arm and row["variant"] == "clean"]
        outage = [
            row
            for row in family_rows
            if row["arm"] == arm and row["variant"] == DEGRADED_CONDITION
        ]
        endpoints = [
            ("三角色联合合规率", clean, "three_role_joint_compliance"),
            ("三角色共享事实通过率", clean, "three_role_shared_fact_pass"),
            ("三角色盲态完全辨识率", clean, "three_way_role_identification"),
            ("患者端无医生式话术泄漏率", clean, "patient_doctor_style_nonleak"),
            ("证据来源不可用时三角色联合降级率", outage, "three_role_joint_degradation"),
        ]
        for offset, (name, rows, field) in enumerate(endpoints):
            values = {str(row["family_token"]): float(row[field]) for row in rows}
            estimate = family_cluster_bootstrap(
                values, args.bootstrap_replicates, args.bootstrap_seed + offset
            )
            primary_rows.append({"arm": arm, "metric": name, **estimate})

    public = {
        "schema_version": "care_opd.competition.three_role_results.v1",
        "status": "COMPLETE_DOUBLE_REVIEWED",
        "data_provenance": (
            "数据提供方提供的真实医疗业务甲状腺记录，经去标识化与结构化处理；"
            "仅用于独立外部评测，不进入训练"
        ),
        "record_count": len({str(row["family_token"]) for row in consensus_rows}),
        "generated_output_count": len(consensus_rows),
        "roles": ["医生", "患者", "学生"],
        "conditions": ["clean", DEGRADED_CONDITION],
        "reviewer_count": len(reviewers),
        "frozen_review_manifest_sha256": sha256_file(review_manifest_path),
        "human_rating_inputs_sha256": hashlib.sha256(
            (
                "|".join(
                    sha256_file(path)
                    for path in [*ratings_paths, *bundle_ratings_paths]
                )
            ).encode("ascii")
        ).hexdigest(),
        "consensus_rule": (
            "序数项取两评审均值；通过项取较保守值；错误/泄漏项取较保守的较高值"
        ),
        "primary_endpoints": primary_rows,
        "role_breakdown": role_rows,
        "inter_reviewer_agreement": agreement,
        "uncertainty": (
            f"按真实记录family整组重采样的{args.bootstrap_replicates}次bootstrap 95%区间；"
            "该区间反映病例抽样不确定性，不代表训练种子变异"
        ),
        "privacy": "仅公开聚合统计，不含病例、答案、标识、评审意见或源标识",
    }
    assert_public_aggregate(public)
    write_json(paths["three_role_results.public.json"], public)

    csv_rows: list[dict[str, Any]] = []
    for row in primary_rows:
        csv_rows.append(
            {
                "section": "primary",
                "arm": row["arm"],
                "condition": "",
                "role": "三角色联合",
                "metric": row["metric"],
                "value": row["estimate"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "count": row["family_count"],
            }
        )
    for row in role_rows:
        for metric, value in row.items():
            if metric in {"arm", "condition", "role", "count"} or value is None:
                continue
            csv_rows.append(
                {
                    "section": "role_breakdown",
                    "arm": row["arm"],
                    "condition": row["condition"],
                    "role": row["role"],
                    "metric": metric,
                    "value": value,
                    "ci95_low": "",
                    "ci95_high": "",
                    "count": row["count"],
                }
            )
    write_csv(
        paths["three_role_results.public.csv"],
        ["section", "arm", "condition", "role", "metric", "value", "ci95_low", "ci95_high", "count"],
        csv_rows,
    )
    md = [
        "# 同一真实病例·三角色受控实验",
        "",
        "数据提供方提供的真实医疗业务甲状腺记录经过去标识化与结构化处理，"
        "仅用于独立外部评测，不进入训练。三角色在同一病例与同一证据状态下共享字节级一致的用户输入，"
        "仅系统角色路由、角色提示与权限链不同。",
        "",
        "| 核心指标 | 评测臂 | 结果 | family-bootstrap 95% CI | 记录数 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in primary_rows:
        md.append(
            f"| {row['metric']} | {row['arm']} | {row['estimate']:.1%} | "
            f"[{row['ci95_low']:.1%}, {row['ci95_high']:.1%}] | {row['family_count']} |"
        )
    md.extend(
        [
            "",
            f"两位评审独立评分；95%区间采用family整组重采样{args.bootstrap_replicates}次。",
            "该不确定性描述病例抽样，不代表训练种子变异。公开结果不含病例、回答或可逆标识。",
        ]
    )
    paths["three_role_results.public.md"].write_text("\n".join(md) + "\n", encoding="utf-8")
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("prepare-review", help="build private packet and blank rating CSV")
    review.add_argument("--tasks", type=Path, required=True)
    review.add_argument("--responses", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    review.add_argument("--reviewers", action="append")
    review.add_argument("--overwrite", action="store_true")

    summary = sub.add_parser("summarize", help="summarize two completed rating sets")
    summary.add_argument("--tasks", type=Path, required=True)
    summary.add_argument("--responses", type=Path, required=True)
    summary.add_argument("--ratings", type=Path, action="append", required=True)
    summary.add_argument("--bundle-ratings", type=Path, action="append", required=True)
    summary.add_argument("--bundle-key", type=Path, required=True)
    summary.add_argument("--review-manifest", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--reviewers", action="append")
    summary.add_argument("--bootstrap-replicates", type=int, default=10_000)
    summary.add_argument("--bootstrap-seed", type=int, default=20260826)
    summary.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    paths = prepare_review(args) if args.command == "prepare-review" else summarize(args)
    print(json.dumps({"status": "COMPLETE", "files": {k: str(v) for k, v in paths.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
