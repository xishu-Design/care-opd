#!/usr/bin/env python3
"""Validate the four local response sets before any expert-review blinding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .three_role_eval_common import (
        canonical_json_sha256,
        read_jsonl,
        sha256_file,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from three_role_eval_common import (  # type: ignore
        canonical_json_sha256,
        read_jsonl,
        sha256_file,
        write_json,
    )


SCHEMA_VERSION = "care_opd.competition.local_four_agent_execution.v2"
LOCK_SCHEMA = "care_opd.competition.local_model_suite.selection_lock.v1"
PRODUCT_PROFILE_KNOWLEDGE_BACKENDS = {
    "live_product": "lightrag",
    "local_product": "local_knowledge",
}
OUTPUT_SELECTION_POLICY = "first_user_visible_output_frozen_content_neutral"
CONTENT_REVIEW_POLICY = (
    "audit_only_expert_judged_no_content_triggered_regeneration"
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_response_set(
    *,
    system_id: str,
    path: Path,
    task_index: dict[str, dict[str, Any]],
    project_system_api_profile: str = "",
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    if len(rows) != len(task_index):
        raise ValueError(
            f"{system_id} response count mismatch: "
            f"expected={len(task_index)}, actual={len(rows)}"
        )
    response_index: dict[str, dict[str, Any]] = {}
    providers: set[str] = set()
    models: set[str] = set()
    product_audit_count = 0
    explicit_selection_contract_count = 0
    legacy_first_attempt_count = 0
    is_project_system = system_id == "shennong_sanyan_full"
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in response_index:
            raise ValueError(f"{system_id} has empty or duplicate task_id")
        task = task_index.get(task_id)
        if task is None:
            raise ValueError(f"{system_id} has unknown task_id: {task_id}")
        if row.get("status") != "success":
            raise ValueError(f"{system_id} has non-success response: {task_id}")
        expected_hash = canonical_json_sha256(task)
        if row.get("task_payload_sha256") != expected_hash:
            raise ValueError(f"{system_id} task hash mismatch: {task_id}")
        if not str(row.get("display_response_private") or "").strip():
            raise ValueError(f"{system_id} has empty display response: {task_id}")
        explicit_row_contract = (
            row.get("output_selection_policy") == OUTPUT_SELECTION_POLICY
            and row.get("content_triggered_regeneration") is False
        )
        if not explicit_row_contract:
            # The three already-frozen baseline files predate the explicit
            # policy fields.  Reuse is fair only when each row is provably its
            # first and only attempt; project-arm rows never receive this
            # compatibility exception.
            if not is_project_system and row.get("attempt_count") == 1:
                legacy_first_attempt_count += 1
            else:
                raise ValueError(
                    f"{system_id} did not freeze a selection-neutral first response: {task_id}"
                )
        else:
            frozen_raw = row.get("raw_payload_private")
            if (
                not isinstance(frozen_raw, dict)
                or row.get("selected_response_sha256")
                != canonical_json_sha256(frozen_raw)
                or not isinstance(row.get("selected_attempt"), int)
                or row.get("selected_attempt") != row.get("attempt_count")
                or row.get("post_output_audit_failure") is not False
            ):
                raise ValueError(
                    f"{system_id} explicit first-response binding is invalid: {task_id}"
                )
            explicit_selection_contract_count += 1
        providers.add(str(row.get("response_provider") or "missing"))
        models.add(str(row.get("response_model") or "missing"))
        if system_id == "shennong_sanyan_full" and project_system_api_profile:
            _validate_project_row_audit(
                row=row,
                task=task,
                expected_profile=project_system_api_profile,
            )
            product_audit_count += 1
        response_index[task_id] = row
    if set(response_index) != set(task_index):
        raise ValueError(f"{system_id} task coverage mismatch")
    engine_manifest = path.with_name("three_role_execution.private.json")
    if not engine_manifest.is_file():
        raise FileNotFoundError(engine_manifest)
    engine = load_object(engine_manifest)
    if (
        engine.get("status") != "COMPLETE"
        or engine.get("success_count") != len(task_index)
        or engine.get("failure_count") != 0
    ):
        raise ValueError(f"{system_id} engine manifest is not COMPLETE")
    if engine.get("response_sha256") != sha256_file(path):
        raise ValueError(f"{system_id} response hash does not match engine manifest")
    explicit_engine_contract = (
        engine.get("output_selection_policy") == OUTPUT_SELECTION_POLICY
        and engine.get("content_triggered_regeneration") is False
    )
    legacy_first_attempt_attested = (
        not is_project_system
        and legacy_first_attempt_count == len(task_index)
        and explicit_selection_contract_count == 0
    )
    if not explicit_engine_contract and not legacy_first_attempt_attested:
        raise ValueError(
            f"{system_id} engine manifest has no selection-neutral output contract"
        )
    if explicit_selection_contract_count not in {0, len(task_index)}:
        raise ValueError(
            f"{system_id} mixes explicit and legacy output-selection contracts"
        )
    if system_id == "shennong_sanyan_full" and project_system_api_profile:
        if engine.get("system_api_profile") != project_system_api_profile:
            raise ValueError("project engine manifest system API profile mismatch")
        if product_audit_count != len(task_index):
            raise ValueError("project product-profile audit coverage is incomplete")
    return {
        "system_id": system_id,
        "status": "COMPLETE",
        "response_count": len(rows),
        "response_sha256": sha256_file(path),
        "engine_manifest_sha256": sha256_file(engine_manifest),
        "response_providers_private": sorted(providers),
        "response_models_private": sorted(models),
        "project_system_api_profile": (
            project_system_api_profile
            if system_id == "shennong_sanyan_full" and project_system_api_profile
            else "not_applicable"
        ),
        "qualified_project_row_count": product_audit_count,
        "output_selection_policy": OUTPUT_SELECTION_POLICY,
        "content_triggered_regeneration": False,
        "legacy_first_attempt_attested": legacy_first_attempt_attested,
        "response_path_private": str(path),
    }


def _validate_project_row_audit(
    *, row: dict[str, Any], task: dict[str, Any], expected_profile: str
) -> None:
    task_id = str(task.get("task_id") or "")
    if row.get("execution_mode") != "system_api":
        raise ValueError(f"project row did not use system_api: {task_id}")
    if row.get("system_api_profile") != expected_profile:
        raise ValueError(f"project row profile mismatch: {task_id}")
    audit = row.get("audit_private")
    if not isinstance(audit, dict) or audit.get("profile") != expected_profile:
        raise ValueError(f"project row has no qualified private audit: {task_id}")
    if expected_profile == "locked_evidence":
        return
    expected_knowledge_backend = PRODUCT_PROFILE_KNOWLEDGE_BACKENDS.get(
        expected_profile
    )
    if expected_knowledge_backend is None:
        raise ValueError(f"unsupported project product profile: {expected_profile}")
    auth = audit.get("auth")
    llm = audit.get("llm")
    orchestration = audit.get("orchestration")
    evidence = audit.get("evidence")
    memory = audit.get("memory")
    setup = audit.get("setup")
    if not all(
        isinstance(value, dict)
        for value in (auth, llm, orchestration, evidence, memory, setup)
    ):
        raise ValueError(f"project product audit is structurally incomplete: {task_id}")
    role = task.get("role")
    if (
        auth.get("endpoint_role") != role
        or auth.get("authenticated_role") != role
        or auth.get("cross_role_violation") is not False
    ):
        raise ValueError(f"project product audit role scope mismatch: {task_id}")
    if (
        llm.get("fallback_used") is not False
        or not str(llm.get("provider") or "")
        or not str(llm.get("model") or "")
    ):
        raise ValueError(f"project product audit LLM qualification failed: {task_id}")
    required_orchestration = {
        "response_path_verified": True,
        "packet_path_verified": True,
        "knowledge_backend": expected_knowledge_backend,
        "graph_backend": "graphiti",
        "memory_backend": "mem0",
        "adapter_fallback_used": False,
        "context_strategy": "recent_window+mem0+graphiti",
    }
    for key, expected in required_orchestration.items():
        if orchestration.get(key) != expected:
            raise ValueError(
                f"project product audit orchestration mismatch for {key}: {task_id}"
            )
    if int(orchestration.get("available_recent_turn_count") or 0) < 1:
        raise ValueError(f"project product audit has no observed prior turn: {task_id}")
    if (
        int(orchestration.get("recent_turn_count") or 0) < 1
        or orchestration.get("context_mode") != "topic_continuation"
    ):
        raise ValueError(f"project product audit did not inject prior context: {task_id}")
    evidence_count = int(evidence.get("evidence_count") or 0)
    knowledge_evidence_count = int(
        evidence.get("knowledge_evidence_count")
        if evidence.get("knowledge_evidence_count") is not None
        else evidence_count
    )
    if (
        evidence.get("visibility") != "doctor_internal"
        or len(str(evidence.get("source_ids_sha256") or "")) != 64
        or not str(evidence.get("packet_id") or "")
        or not str(evidence.get("trace_id") or "")
    ):
        raise ValueError(f"project product audit evidence qualification failed: {task_id}")
    degraded_tools = orchestration.get("degraded_tools")
    degraded_tools = degraded_tools if isinstance(degraded_tools, list) else []
    if expected_profile == "live_product" and evidence_count < 1:
        raise ValueError(f"project live_product has no retrieved evidence: {task_id}")
    if expected_profile == "local_product":
        if knowledge_evidence_count < 1 and "search_evidence" not in degraded_tools:
            raise ValueError(
                f"project local_product omitted evidence without a degraded retrieval audit: {task_id}"
            )
        if knowledge_evidence_count > 0 and "search_evidence" in degraded_tools:
            raise ValueError(
                f"project local_product evidence/degraded audit is contradictory: {task_id}"
            )
    if task.get("modality") == "tcm_open_case_reasoning":
        expected_case_source = (
            f"CASE_INPUT_{str(task.get('paired_case_material_sha256') or '')[:16]}"
        )
        if (
            evidence.get("case_material_evidence_verified") is not True
            or evidence.get("case_material_source_id") != expected_case_source
        ):
            raise ValueError(
                f"project case material evidence binding failed: {task_id}"
            )
        raw_payload = row.get("raw_payload_private")
        response_data = (
            raw_payload.get("data") if isinstance(raw_payload, dict) else None
        )
        generated_response = (
            response_data.get("generated_response")
            if isinstance(response_data, dict)
            else None
        )
        if not isinstance(generated_response, dict):
            raise ValueError(
                f"project case reasoning generated response is missing: {task_id}"
            )
        quality = audit.get("case_reasoning_quality_gate")
        if (
            not isinstance(quality, dict)
            or quality.get("status") != "PASS"
            or quality.get("failure_reasons") != []
            or not isinstance(quality.get("content_review_flags"), list)
            or quality.get("content_review_enforced") is not False
            or quality.get("content_review_policy") != CONTENT_REVIEW_POLICY
            or quality.get("length_guidance_enforced") is not False
            or quality.get("length_guidance_policy")
            != "advisory_only_not_an_eligibility_or_scoring_threshold"
            or not isinstance(quality.get("safety_review_flags"), list)
            or quality.get("safety_review_enforced") is not False
            or quality.get("safety_review_policy") != CONTENT_REVIEW_POLICY
        ):
            raise ValueError(
                f"project case reasoning selection-neutral audit contract failed: {task_id}"
            )
    if (
        memory.get("scope_role") != role
        or memory.get("context_mode") != "topic_continuation"
        or int(memory.get("recent_turn_count") or 0) < 1
        or int(memory.get("available_recent_turn_count") or 0) < 1
        or not str(memory.get("scope_identity_sha256") or "")
    ):
        raise ValueError(f"project product audit memory scope failed: {task_id}")
    if (
        setup.get("protocol") != "isolated_neutral_longitudinal_state_v1"
        or setup.get("seed_contains_case_material") is not False
        or setup.get("warmup_chat_id_present") is not True
    ):
        raise ValueError(f"project product audit setup contract failed: {task_id}")
    if setup.get("profile", expected_profile) != expected_profile:
        raise ValueError(f"project product audit setup profile mismatch: {task_id}")
    if setup.get("knowledge_backend", expected_knowledge_backend) != expected_knowledge_backend:
        raise ValueError(
            f"project product audit setup knowledge backend mismatch: {task_id}"
        )
    probes = setup.get("probes")
    required_probes = {
        "knowledge": expected_knowledge_backend,
        "memory": "mem0",
        "graph": "graphiti",
    }
    if not isinstance(probes, dict):
        raise ValueError(f"project product audit setup probes are missing: {task_id}")
    for name, backend in required_probes.items():
        probe = probes.get(name)
        if (
            not isinstance(probe, dict)
            or probe.get("backend") != backend
            or probe.get("fallback_used") is not False
        ):
            raise ValueError(
                f"project product audit setup probe mismatch for {name}: {task_id}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks_path = args.tasks.expanduser().resolve()
    spec_path = args.model_spec.expanduser().resolve()
    lock_path = args.selection_lock.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (tasks_path, spec_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tasks = read_jsonl(tasks_path)
    task_index: dict[str, dict[str, Any]] = {}
    for row in tasks:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in task_index:
            raise ValueError("tasks contain an empty or duplicate task_id")
        task_index[task_id] = row
    if not task_index:
        raise ValueError("task package is empty")

    spec = load_object(spec_path)
    models = spec.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("model spec must contain exactly three baselines")
    baseline_ids = [str(row.get("system_id") or "") for row in models]
    if len(set(baseline_ids)) != 3 or any(not value for value in baseline_ids):
        raise ValueError("model spec baseline system_id values are invalid")

    lock = load_object(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("unexpected model selection-lock schema")
    if lock.get("status") != "FROZEN_BEFORE_ANY_BLIND_EVAL_GENERATION":
        raise ValueError("model selection was not frozen before generation")
    if lock.get("model_suite_sha256") != canonical_hash(spec):
        raise ValueError("model selection lock does not match model suite")
    locked_ids = [str(row.get("system_id") or "") for row in lock.get("models", [])]
    if locked_ids != baseline_ids:
        raise ValueError("selection-lock model order differs from model spec")
    if any(len(str(row.get("resolved_revision") or "")) != 40 for row in lock["models"]):
        raise ValueError("every baseline must have an immutable 40-character revision")

    systems = ["shennong_sanyan_full", *baseline_ids]
    audits = []
    for system_id in systems:
        path = run_root / system_id / "responses.private.jsonl"
        audits.append(
            validate_response_set(
                system_id=system_id,
                path=path,
                task_index=task_index,
                project_system_api_profile=(
                    args.project_system_api_profile
                    if system_id == "shennong_sanyan_full"
                    else ""
                ),
            )
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "comparison_scope": (
            "Shennong Sanyan declared product deployment profile versus three frozen local open-weight "
            "role-conditioned model baselines; not CARE-OPD causal attribution and "
            "not clinical efficacy"
        ),
        "project_system_api_profile": args.project_system_api_profile,
        "privacy_scope": "all inference endpoints were loopback-local on the controlled server",
        "output_selection_policy": OUTPUT_SELECTION_POLICY,
        "content_triggered_regeneration": False,
        "content_review_policy": CONTENT_REVIEW_POLICY,
        "legacy_first_attempt_attested": any(
            bool(system.get("legacy_first_attempt_attested")) for system in audits
        ),
        "legacy_first_attempt_system_ids": [
            str(system["system_id"])
            for system in audits
            if system.get("legacy_first_attempt_attested") is True
        ],
        "task_count_per_system": len(task_index),
        "total_response_count": len(task_index) * len(systems),
        "tasks_sha256": sha256_file(tasks_path),
        "model_spec_sha256": sha256_file(spec_path),
        "selection_lock_sha256": sha256_file(lock_path),
        "systems": audits,
        "ready_for_blind_review_packaging": True,
    }
    write_json(output_path, result)
    return {**result, "manifest": str(output_path)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--model-spec", type=Path, required=True)
    value.add_argument("--selection-lock", type=Path, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument(
        "--project-system-api-profile",
        choices=("", "locked_evidence", "live_product", "local_product"),
        default="",
    )
    return value


def main() -> None:
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
