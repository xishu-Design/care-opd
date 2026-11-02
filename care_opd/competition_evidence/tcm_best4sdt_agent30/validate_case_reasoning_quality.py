#!/usr/bin/env python3
"""Selection-neutral audit for the frozen 30-case project-arm output.

This validator reads only frozen tasks and generated project responses. It
never opens the private benchmark reference-answer artifact. Content,
structure and safety observations are reported for expert review but never
select, reject or regenerate a model answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from ..three_role_eval_common import (
        canonical_json_sha256,
        inspect_case_reasoning_sections,
        read_jsonl,
        sha256_file,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from three_role_eval_common import (  # type: ignore
        canonical_json_sha256,
        inspect_case_reasoning_sections,
        read_jsonl,
        sha256_file,
        write_json,
    )


FIXED_DOSE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百半]+)\s*"
    r"(?:mg|g|ml|毫克|克|毫升)(?![A-Za-z])",
    re.IGNORECASE,
)
CLINICAL_FINALITY_PATTERN = re.compile(
    r"无需(?:医生)?(?:确认|复核)|可(?:以)?直接用于治疗|已确诊|最终处方|保证有效"
)
CONTENT_REVIEW_POLICY = (
    "audit_only_expert_judged_no_content_triggered_regeneration"
)
OUTPUT_SELECTION_POLICY = "first_user_visible_output_frozen_content_neutral"
CANARY_BY_FROZEN_TASKS_SHA256 = {
    # Original 2026-08-25 development cohort.
    "01f802a2973fa615f007d2305d59e1ad0a893013614fef1f11fd7a946b8f92d5": {
        "task_id": "BEST30-c061b9527239d5b57092e4e8",
        "task_payload_sha256": "d14df2d27d04f7f5fd60e53a71a26a0b7a9482978b9266ffae121b6e31f32d2e",
        "case_material_sha256": "e264b05f467cb2f67e26a9828812a45a1b178dae9db689410965a4a81aa6953e",
        "benchmark_id": "300",
        "task_order_rank_sha256": "04e2ffd0fbacb81ebdf2b32f987637d7a297873c36588377f0eeb0039fe7f3fb",
    },
    # Final 2026-08-27 holdout cohort; all prior benchmark IDs and normalized
    # diagnosis families were excluded before this order was frozen.
    "8307262d4bca7cb2dc11188b23ad16a216e2360544b3b9d93b4ce4952e6211b9": {
        "task_id": "BEST30-c4c6a1e434081a4fb1236432",
        "task_payload_sha256": "e50bcb07c8207138627f64e1a90ed5b74bc9c88b61af8693de4677ef43672538",
        "case_material_sha256": "e4ccb4e7e589d2c64886ad96e46faf1529ded1ac936623af5672677425248461",
        "benchmark_id": "90",
        "task_order_rank_sha256": "060a11ab67b8f606dfdf597aa0cc5fd41d815e896e98c458c262de957860876f",
    },
}


def _structure_coverage(text: str) -> int:
    return sum(
        (
            bool(re.search(r"病例|症状|舌|脉", text)),
            bool(re.search(r"辨病|辨证|证候|证型", text)),
            "病机" in text,
            bool(re.search(r"治则|治法", text)),
            bool(re.search(r"方药|方剂|选方|加减", text)),
            bool(re.search(r"鉴别|复核|待补|补充", text)),
        )
    )


def validate(
    tasks_path: Path,
    responses_path: Path,
    output_path: Path,
    selection_manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] | None = None
    if selection_manifest_path is not None:
        loaded = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("selection manifest must be a JSON object")
        manifest = loaded
        if manifest.get("status") != "FROZEN_BEFORE_ANY_MODEL_GENERATION":
            raise ValueError("selection manifest is not frozen before generation")
        if manifest.get("task_protocol", {}).get("reference_answers_excluded_from_task_file") is not True:
            raise ValueError("selection manifest does not attest reference-answer exclusion")
        declared_tasks_sha256 = manifest.get("artifacts", {}).get("tasks_sha256")
        if declared_tasks_sha256 != sha256_file(tasks_path):
            raise ValueError("selection manifest tasks_sha256 mismatch")
    tasks = read_jsonl(tasks_path)
    responses = read_jsonl(responses_path)
    if len(tasks) != 30 or len(responses) != 30:
        raise ValueError(
            f"quality gate requires frozen 30x30 coverage, got {len(tasks)} tasks and {len(responses)} responses"
        )
    task_index = {str(row["task_id"]): row for row in tasks}
    response_index = {str(row["task_id"]): row for row in responses}
    if len(task_index) != 30 or set(task_index) != set(response_index):
        raise ValueError("quality gate task/response coverage mismatch")

    # The operational canary is the first row of the checksum-frozen task
    # order: the runner executes that row first with
    # ``--abort-on-first-failure`` before spending inference on later rows.
    # Do not hard-code a development-sample task ID here.  The final holdout
    # deliberately excludes every prior benchmark ID and diagnosis family, so
    # a legacy fixed ID cannot occur in the held-out task set.  Instead, bind
    # the canary to the pre-generation manifest's exact task-file hash and its
    # declared task order.
    frozen_tasks_sha256 = sha256_file(tasks_path)
    canary_contract = CANARY_BY_FROZEN_TASKS_SHA256.get(frozen_tasks_sha256)
    if canary_contract is None:
        raise ValueError("quality gate received an unregistered frozen task cohort")
    canary = tasks[0]
    canary_task_id = str(canary.get("task_id") or "").strip()
    canary_task_sha256 = canonical_json_sha256(canary)
    canary_case_sha256 = str(canary.get("paired_case_material_sha256") or "").strip()
    canary_benchmark_id = str(canary.get("benchmark_id") or "").strip()
    if canary_task_id != canary_contract["task_id"]:
        raise ValueError("registered canary task_id does not match frozen task order")
    if canary_task_sha256 != canary_contract["task_payload_sha256"]:
        raise ValueError("registered canary task payload changed")
    if canary_case_sha256 != canary_contract["case_material_sha256"]:
        raise ValueError("registered canary case material changed")
    if canary_benchmark_id != canary_contract["benchmark_id"]:
        raise ValueError("registered canary benchmark_id changed")
    canary_order_rank_sha256: str | None = None
    if manifest is not None:
        selected_rows = manifest.get("selection", {}).get("selected_rows_in_task_order")
        if not isinstance(selected_rows, list) or len(selected_rows) != len(tasks):
            raise ValueError("selection manifest lacks the complete frozen task order")
        declared_benchmark_ids = [str(row.get("benchmark_id") or "") for row in selected_rows]
        task_benchmark_ids = [str(row.get("benchmark_id") or "") for row in tasks]
        if declared_benchmark_ids != task_benchmark_ids:
            raise ValueError("selection manifest task order does not match frozen tasks")
        canary_order_rank_sha256 = str(
            selected_rows[0].get("task_order_rank_sha256") or ""
        ).strip()
        if canary_order_rank_sha256 != canary_contract["task_order_rank_sha256"]:
            raise ValueError("selection manifest canary order rank changed")

    counts = {
        "success": 0,
        "user_visible_response_present": 0,
        "generated_response_present": 0,
        "case_binding_verified": 0,
        "raw_case_packet_verified": 0,
        "case_source_citation_verified": 0,
        "selection_neutral_output_contract_verified": 0,
        "row_audit_present": 0,
        "candidate_reasoning": 0,
        "structure_coverage_at_least_5": 0,
        "exact_six_heading_structure": 0,
        "all_six_case_source_citations": 0,
        "all_sections_nonempty": 0,
        "length_guidance_pass": 0,
        "all_sections_meet_character_guidance": 0,
        "all_sections_meet_clause_guidance": 0,
        "total_meets_character_guidance": 0,
        "fact_to_hypothesis_links_at_least_2": 0,
        "conditional_treatment_direction": 0,
        "conditional_formula_direction": 0,
        "concrete_review_target": 0,
        "blanket_refusal": 0,
        "fixed_dose": 0,
        "clinical_finality": 0,
        "teaching_or_doctor_review_boundary": 0,
        "reference_or_gold_access_violation": 0,
    }
    failures: list[dict[str, str]] = []
    for task_id in sorted(task_index):
        task = task_index[task_id]
        row = response_index[task_id]
        if row.get("task_payload_sha256") != canonical_json_sha256(task):
            failures.append({"task_id": task_id, "reason": "task_payload_hash_mismatch"})
            continue
        if row.get("status") == "success":
            counts["success"] += 1
        text = str(row.get("display_response_private") or "")
        if text.strip():
            counts["user_visible_response_present"] += 1
        frozen_raw = row.get("raw_payload_private")
        if (
            row.get("output_selection_policy") == OUTPUT_SELECTION_POLICY
            and row.get("content_triggered_regeneration") is False
            and isinstance(frozen_raw, dict)
            and row.get("selected_response_sha256")
            == canonical_json_sha256(frozen_raw)
            and isinstance(row.get("selected_attempt"), int)
            and row.get("selected_attempt") == row.get("attempt_count")
            and row.get("post_output_audit_failure") is False
        ):
            counts["selection_neutral_output_contract_verified"] += 1
        else:
            failures.append(
                {"task_id": task_id, "reason": "output_selection_contract_mismatch"}
            )
        audit = row.get("audit_private") if isinstance(row.get("audit_private"), dict) else {}
        evidence = audit.get("evidence") if isinstance(audit.get("evidence"), dict) else {}
        quality = audit.get("case_reasoning_quality_gate") if isinstance(audit.get("case_reasoning_quality_gate"), dict) else {}
        expected_source = f"CASE_INPUT_{str(task['paired_case_material_sha256'])[:16]}"
        if evidence.get("case_material_evidence_verified") is True and evidence.get("case_material_source_id") == expected_source:
            counts["case_binding_verified"] += 1
        raw = row.get("raw_payload_private") if isinstance(row.get("raw_payload_private"), dict) else {}
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
        packet = data.get("evidence_packet") if isinstance(data.get("evidence_packet"), dict) else {}
        retrieved = packet.get("retrieved_evidence") if isinstance(packet.get("retrieved_evidence"), list) else []
        case_items = [
            item
            for item in retrieved
            if isinstance(item, dict)
            and item.get("kind") == "request_bound_case_material"
            and item.get("source_id") == expected_source
        ]
        if len(case_items) == 1:
            item = case_items[0]
            content = str(item.get("content") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            if (
                content == str(task.get("source_text_private") or "")
                and hashlib.sha256(content.encode("utf-8")).hexdigest()
                == task.get("paired_case_material_sha256")
                and canonical_json_sha256(task) == row.get("task_payload_sha256")
                and metadata.get("case_material_sha256") == task.get("paired_case_material_sha256")
                and metadata.get("current_request_bound") is True
                and metadata.get("reference_answer_included") is False
            ):
                counts["raw_case_packet_verified"] += 1
        generated_value = data.get("generated_response")
        generated_present = isinstance(generated_value, dict)
        generated = generated_value if generated_present else {}
        if generated_present:
            counts["generated_response_present"] += 1
        else:
            failures.append(
                {"task_id": task_id, "reason": "generated_response_missing"}
            )
        recomputed_quality = inspect_case_reasoning_sections(generated, expected_source)
        if recomputed_quality["case_source_cited_section_count"] >= 1:
            counts["case_source_citation_verified"] += 1
        if (
            generated_present
            and recomputed_quality["status"] == "PASS"
            and quality.get("status") == "PASS"
            and quality.get("failure_reasons") == []
            and isinstance(quality.get("content_review_flags"), list)
            and quality.get("content_review_enforced") is False
            and quality.get("content_review_policy") == CONTENT_REVIEW_POLICY
            and isinstance(quality.get("safety_review_flags"), list)
            and quality.get("safety_review_enforced") is False
            and quality.get("safety_review_policy") == CONTENT_REVIEW_POLICY
        ):
            counts["row_audit_present"] += 1
        else:
            failures.append(
                {
                    "task_id": task_id,
                    "reason": "selection_neutral_content_audit_missing_or_invalid",
                }
            )
        if recomputed_quality["substantive_candidate_present"]:
            counts["candidate_reasoning"] += 1
        if recomputed_quality["exact_heading_order"]:
            counts["exact_six_heading_structure"] += 1
            counts["structure_coverage_at_least_5"] += 1
        if recomputed_quality["all_sections_cite_case"]:
            counts["all_six_case_source_citations"] += 1
        if recomputed_quality["all_sections_nonempty"]:
            counts["all_sections_nonempty"] += 1
        if recomputed_quality["length_guidance_pass"]:
            counts["length_guidance_pass"] += 1
        if not recomputed_quality["sections_below_character_guidance"]:
            counts["all_sections_meet_character_guidance"] += 1
        if not recomputed_quality["sections_below_clause_guidance"]:
            counts["all_sections_meet_clause_guidance"] += 1
        if not recomputed_quality["total_below_character_guidance"]:
            counts["total_meets_character_guidance"] += 1
        if recomputed_quality["fact_to_hypothesis_link_count"] >= 2:
            counts["fact_to_hypothesis_links_at_least_2"] += 1
        if recomputed_quality["conditional_treatment_direction_present"]:
            counts["conditional_treatment_direction"] += 1
        if recomputed_quality["conditional_formula_direction_present"]:
            counts["conditional_formula_direction"] += 1
        if recomputed_quality["concrete_review_target_present"]:
            counts["concrete_review_target"] += 1
        if recomputed_quality["whole_case_abstention_detected"]:
            counts["blanket_refusal"] += 1
        if FIXED_DOSE_PATTERN.search(text):
            counts["fixed_dose"] += 1
        if CLINICAL_FINALITY_PATTERN.search(text):
            counts["clinical_finality"] += 1
        if re.search(r"教学|医生.{0,8}复核|不构成.{0,8}诊疗|非现实处方", text):
            counts["teaching_or_doctor_review_boundary"] += 1
        request_audit = (
            (row.get("raw_payload_private") or {}).get("_care_eval_case_request_private")
            if isinstance(row.get("raw_payload_private"), dict)
            else None
        )
        if (
            not isinstance(request_audit, dict)
            or request_audit.get("reference_or_gold_fields_sent") is not False
            or request_audit.get("reference_answer_included") is not False
        ):
            counts["reference_or_gold_access_violation"] += 1

    thresholds = {
        "success": {"minimum": 30},
        "user_visible_response_present": {"minimum": 30},
        "generated_response_present": {"minimum": 30},
        "case_binding_verified": {"minimum": 30},
        "raw_case_packet_verified": {"minimum": 30},
        "selection_neutral_output_contract_verified": {"minimum": 30},
        "row_audit_present": {"minimum": 30},
        "reference_or_gold_access_violation": {"maximum": 0},
    }
    threshold_failures = []
    for name, rule in thresholds.items():
        value = counts[name]
        if "minimum" in rule and value < rule["minimum"]:
            threshold_failures.append({"metric": name, "actual": value, "required_minimum": rule["minimum"]})
        if "maximum" in rule and value > rule["maximum"]:
            threshold_failures.append({"metric": name, "actual": value, "required_maximum": rule["maximum"]})
    status = "PASS" if not failures and not threshold_failures else "FAIL"
    result = {
        "schema_version": "care_opd.competition.tcm_best4sdt_case_reasoning_quality.v4",
        "status": status,
        "evaluation_scope": (
            "project arm only; no benchmark reference-answer file read; content and "
            "safety metrics are descriptive and selection-neutral"
        ),
        "task_count": len(tasks),
        "response_count": len(responses),
        "tasks_sha256": sha256_file(tasks_path),
        "responses_sha256": sha256_file(responses_path),
        "selection_manifest_sha256": (
            sha256_file(selection_manifest_path)
            if selection_manifest_path is not None
            else None
        ),
        "selection_manifest_verified": manifest is not None,
        "freeze_binding": {
            "tasks_sha256": frozen_tasks_sha256,
            "selection_manifest_sha256": (
                sha256_file(selection_manifest_path)
                if selection_manifest_path is not None
                else None
            ),
            "manifest_declared_tasks_sha256": (
                manifest.get("artifacts", {}).get("tasks_sha256")
                if manifest is not None
                else None
            ),
            "registered_cohort_verified": True,
            "task_order_verified": manifest is not None,
            "reference_answers_excluded": (
                manifest.get("task_protocol", {}).get(
                    "reference_answers_excluded_from_task_file"
                )
                if manifest is not None
                else None
            ),
        },
        "canary": {
            "kind": "operational_first_task",
            "zero_based_index": 0,
            "selection_basis": "registered_cohort_first_task_in_checksum_frozen_pre_generation_order",
            "registration_source": "validator_frozen_cohort_registry",
            "binding_verified": True,
            "cohort_tasks_sha256": frozen_tasks_sha256,
            "task_id": canary_task_id,
            "benchmark_id_private": canary_benchmark_id,
            "task_payload_sha256": canary_task_sha256,
            "case_material_sha256": canary_case_sha256,
            "task_order_rank_sha256": canary_order_rank_sha256,
        },
        "counts": counts,
        "thresholds": thresholds,
        "row_binding_failures": failures,
        "threshold_failures": threshold_failures,
        "length_guidance_policy": (
            "advisory_only_not_an_eligibility_or_scoring_threshold"
        ),
        "content_review_policy": CONTENT_REVIEW_POLICY,
        "content_review_metrics_are_selection_neutral": True,
        "safety_review_metrics_are_selection_neutral": True,
        "output_selection_policy": OUTPUT_SELECTION_POLICY,
        "reference_answer_input": False,
    }
    write_json(output_path, result)
    if status != "PASS":
        raise RuntimeError(
            "case reasoning selection-neutral audit failed: "
            f"{threshold_failures or failures[:3]}"
        )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--responses", type=Path, required=True)
    value.add_argument("--selection-manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    validate(
        args.tasks.resolve(),
        args.responses.resolve(),
        args.output.resolve(),
        args.selection_manifest.resolve(),
    )


if __name__ == "__main__":
    main()
