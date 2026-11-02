from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import EvidencePacketLog
from app.harness.trace import record_event
from app.mcp.contracts import ToolResult, utc_iso
from app.services.conversation_context_service import context_memory_items


def build_evidence_packet(role: str, task_type: str, case_summary: str, rule_findings: list[dict]) -> dict:
    return {
        "packet_id": str(uuid4()),
        "role": role,
        "task_type": task_type,
        "visibility": _visibility_for(role),
        "case_summary": case_summary,
        "retrieved_evidence": [
            {
                "source": "mock_lightrag",
                "summary": "第一阶段使用 Mock LightRAG 证据，后续可替换为真实检索结果。",
            }
        ],
        "rule_findings": rule_findings,
        "graph_reasoning_path": [
            {"source": "mock_graphiti", "path": ["patient", "prescription", "risk_card"]}
        ],
        "memory_used": [{"source": "mock_mem0", "summary": "未读取跨角色敏感记忆。"}],
        "safety_decision": {
            "allow_answer": True,
            "need_human_confirm": True,
            "refusal_reason": "",
        },
        "created_at": utc_iso(),
    }


def build_evidence_packet_v2(
    db: Session,
    *,
    role: str,
    task_type: str,
    visibility: str,
    case_summary: str,
    tool_results: list[ToolResult],
    created_by_user_id: int,
    safety_decision: dict | None = None,
    trace_id: str | None = None,
    graph_node_path: list[str] | None = None,
    human_confirm: dict | None = None,
    conversation_context: dict | None = None,
    request_bound_evidence: list[dict] | None = None,
    request_bound_contract: dict | None = None,
) -> dict:
    packet_id = str(uuid4())
    retrieved_evidence: list[dict] = list(request_bound_evidence or [])
    graph_reasoning_path: list[dict] = []
    rule_findings: list[dict] = []
    memory_used: list[dict] = []
    structured_context: dict = {}
    tool_calls: list[dict] = []
    degraded_reasons: list[str] = []

    for result in tool_results:
        data = result.data or {}
        if data.get("evidence"):
            retrieved_evidence.extend(data["evidence"])
        if data.get("graph_reasoning_path"):
            graph_reasoning_path.extend(data["graph_reasoning_path"])
        if data.get("rule_findings"):
            rule_findings.extend(data["rule_findings"])
        if data.get("memory_used"):
            memory_used.extend(data["memory_used"])
        if data.get("parsed_items"):
            structured_context.setdefault("prescription", {})["items"] = data["parsed_items"]
        if data.get("prescription_snapshot"):
            structured_context["prescription"] = data["prescription_snapshot"]
        if "patient_allergy_summary" in data:
            structured_context["patient_allergy_summary"] = data.get("patient_allergy_summary") or ""
        if "allergy_matched_herbs" in data:
            structured_context["allergy_matched_herbs"] = data.get("allergy_matched_herbs") or []
        if "herb_incompatibility_matches" in data:
            structured_context["herb_incompatibility_matches"] = data.get("herb_incompatibility_matches") or []
        if data.get("patient_profile"):
            structured_context["patient_self"] = {
                "profile": data["patient_profile"],
                "medical_history": data.get("medical_history") or "",
                "allergy_history": data.get("allergy_history") or "",
            }
        if "published_prescriptions" in data:
            structured_context["published_prescriptions"] = data.get("published_prescriptions") or []
        if "doctor_patient_roster" in data:
            structured_context["doctor_patient_roster"] = data.get("doctor_patient_roster") or []
        if data.get("learning_case"):
            structured_context["learning_case"] = data["learning_case"]
        if data.get("standard_analysis"):
            structured_context["learning_standard_analysis"] = data["standard_analysis"]
        if result.tool_name in {"run_prescription_review", "run_mock_prescription_review"}:
            structured_context["rule_review"] = {
                "risk_level": data.get("risk_level"),
                "review_summary": data.get("review_summary"),
            }
        tool_calls.append(
            {
                "request_id": result.request_id,
                "tool_name": result.tool_name,
                "allowed": result.allowed,
                "success": result.success,
                "risk_level": result.risk_level,
                "requires_confirmation": result.requires_confirmation,
                "error_code": result.error_code,
                "metadata": result.metadata,
            }
        )
        if result.metadata.get("fallback_used"):
            degraded_reasons.append(f"fallback:{result.tool_name}:{result.metadata.get('fallback_reason') or 'adapter_fallback'}")
        if result.metadata.get("degraded"):
            degraded_reasons.append(result.metadata.get("degraded_reason") or f"degraded:{result.tool_name}")

    if rule_findings and not retrieved_evidence:
        degraded_reasons.append("rule_triggered_without_evidence")

    if request_bound_contract:
        structured_context["case_reasoning_contract"] = dict(request_bound_contract)
    conversation_context = conversation_context or {}
    memory_used.extend(context_memory_items(conversation_context))
    backend_summary = _backend_summary(tool_results)
    if request_bound_contract:
        backend_summary["case_reasoning_contract"] = dict(request_bound_contract)
    backend_summary["context_strategy"] = conversation_context.get("strategy")
    backend_summary["recent_turn_count"] = int(conversation_context.get("turn_count") or 0)
    backend_summary["available_recent_turn_count"] = int(conversation_context.get("available_turn_count") or 0)
    backend_summary["follow_up_detected"] = bool(conversation_context.get("follow_up_detected"))
    backend_summary["context_mode"] = conversation_context.get("context_mode")
    business_context_types = [
        key
        for key in ("patient_self", "published_prescriptions", "doctor_patient_roster")
        if key in structured_context
    ]
    backend_summary["business_context_types"] = business_context_types
    backend_summary["business_context_count"] = sum(
        len(structured_context.get(key) or []) if isinstance(structured_context.get(key), list) else 1
        for key in business_context_types
    )
    degraded = bool(degraded_reasons)
    visible_evidence = _filter_for_visibility(retrieved_evidence, visibility)
    private_control = backend_summary.get("private_eval_evidence_control")
    if isinstance(private_control, dict):
        expected_count = int(private_control.get("expected_retrieved_evidence_count", -1))
        actual_count = len(visible_evidence)
        private_control = {
            **private_control,
            "actual_retrieved_evidence_count": actual_count,
            "evidence_count_verified": actual_count == expected_count,
        }
        if actual_count != expected_count:
            raise RuntimeError(
                "private evaluation evidence count failed closed: "
                f"expected {expected_count}, got {actual_count}"
            )
        backend_summary["private_eval_evidence_control"] = private_control

    packet = {
        "packet_id": packet_id,
        "trace_id": trace_id,
        "role": role,
        "task_type": task_type,
        "visibility": visibility,
        "case_summary": case_summary,
        "retrieved_evidence": visible_evidence,
        "graph_reasoning_path": _graph_for_visibility(graph_reasoning_path, visibility),
        "graph_node_path": graph_node_path or [],
        "rule_findings": rule_findings if visibility == "doctor_internal" else [],
        "memory_used": _memory_for_visibility(memory_used, visibility),
        "conversation_context": conversation_context,
        "structured_context": structured_context,
        "tool_calls": tool_calls,
        "backend_summary": backend_summary,
        "degraded": degraded,
        "degraded_reasons": sorted(set(reason for reason in degraded_reasons if reason)),
        "safety_decision": safety_decision
        or {
            "allow_answer": True,
            "need_human_confirm": visibility == "doctor_internal",
            "refusal_reason": "",
        },
        "human_confirm": human_confirm or {},
        "created_at": utc_iso(),
    }
    db.add(
        EvidencePacketLog(
            packet_id=packet_id,
            trace_id=trace_id,
            role=role,
            task_type=task_type,
            visibility=visibility,
            case_summary=case_summary,
            retrieved_evidence_json=packet["retrieved_evidence"],
            graph_reasoning_path_json=packet["graph_reasoning_path"],
            graph_node_path_json=packet["graph_node_path"],
            rule_findings_json=packet["rule_findings"],
            memory_used_json=packet["memory_used"],
            tool_calls_json=packet["tool_calls"],
            backend_summary_json=packet["backend_summary"],
            degraded=packet["degraded"],
            degraded_reasons_json=packet["degraded_reasons"],
            safety_decision_json=packet["safety_decision"],
            human_confirm_json=packet["human_confirm"],
            created_by_user_id=created_by_user_id,
        )
    )
    record_event(
        db,
        trace_id,
        node_name="EvidencePacketNode",
        event_type="evidence_packet",
        status="success",
        output_summary={
            "packet_id": packet_id,
            "visibility": visibility,
            "evidence_count": len(packet["retrieved_evidence"]),
            "degraded": packet["degraded"],
            "fallback_used": packet["backend_summary"].get("fallback_used", False),
            "graph_node_path": packet["graph_node_path"],
        },
        evidence_packet_id=packet_id,
    )
    db.flush()
    return packet


def update_evidence_packet_human_confirm(db: Session, packet_id: str | None, human_confirm: dict) -> dict:
    if not packet_id:
        return {}
    packet_log = db.query(EvidencePacketLog).filter(EvidencePacketLog.packet_id == packet_id).first()
    if not packet_log:
        return {}
    current = dict(packet_log.human_confirm_json or {})
    current.update(human_confirm)
    packet_log.human_confirm_json = current
    db.flush()
    return current


def _visibility_for(role: str) -> str:
    return {"doctor": "doctor_internal", "patient": "patient_safe", "student": "student_learning"}.get(role, "doctor_internal")


def _backend_summary(tool_results: list[ToolResult]) -> dict:
    summary = {
        "knowledge_backend": None,
        "knowledge_fallback_used": False,
        "fallback_backend": None,
        "lightrag_mode": None,
        "lightrag_degraded": False,
        "retrieval_mode": None,
        "rule_engine_backend": None,
        "graph_backend": None,
        "memory_backend": None,
        "fallback_used": False,
        "graph_retrieval_count": 0,
        "qa_retrieval_count": 0,
    }
    for result in tool_results:
        backend = result.metadata.get("backend")
        if result.tool_name in {
            "search_evidence",
            "search_herb",
            "search_formula",
            "search_contraindication",
            "search_diet_knowledge",
            "public_tcm_education",
            "private_eval_locked_evidence",
            "mock_search_evidence",
            "mock_search_herb",
            "mock_search_formula",
            "mock_search_contraindication",
            "mock_search_diet_knowledge",
            "mock_public_tcm_education",
        }:
            summary["knowledge_backend"] = backend
            if result.metadata.get("configured_backend") == "lightrag":
                summary["knowledge_backend"] = result.metadata.get("knowledge_backend") or backend
            if result.metadata.get("fallback_used"):
                summary["knowledge_fallback_used"] = True
                summary["fallback_backend"] = result.metadata.get("fallback_backend") or backend
            if result.metadata.get("lightrag_mode"):
                summary["lightrag_mode"] = result.metadata.get("lightrag_mode")
            if result.metadata.get("retrieval_mode"):
                summary["retrieval_mode"] = result.metadata.get("retrieval_mode")
            summary["qa_retrieval_count"] += int(result.metadata.get("qa_retrieval_count") or 0)
            if result.metadata.get("degraded") and (backend == "lightrag" or result.metadata.get("configured_backend") == "lightrag"):
                summary["lightrag_degraded"] = True
            summary["graph_retrieval_count"] += sum(1 for item in result.evidence if item.get("graph_entities") or item.get("graph_relations") or item.get("graph_path"))
        elif result.tool_name in {"run_prescription_review", "run_mock_prescription_review"}:
            summary["rule_engine_backend"] = backend
        elif result.tool_name in {
            "create_case_graph",
            "update_case_graph",
            "add_prescription_to_graph",
            "add_risk_to_graph",
            "get_reasoning_path",
            "export_deidentified_graph",
            "mock_get_reasoning_path",
            "mock_export_deidentified_graph",
        }:
            summary["graph_backend"] = backend
        elif "memory" in result.tool_name or result.tool_name in {"get_doctor_preferences", "get_patient_explanation_preference", "get_student_progress_memory"}:
            summary["memory_backend"] = backend
        if result.metadata.get("fallback_used"):
            summary["fallback_used"] = True
        private_control = result.metadata.get("private_eval_evidence_control")
        if isinstance(private_control, dict):
            summary["private_eval_evidence_control"] = dict(private_control)
    return summary


def _filter_for_visibility(items: list[dict], visibility: str) -> list[dict]:
    if visibility == "patient_safe":
        return [{**item, "visibility": "patient_safe"} for item in items if _evidence_visible(item, {"patient", "patient_safe"})]
    if visibility == "student_learning":
        return [{**item, "visibility": "student_learning"} for item in items if _evidence_visible(item, {"student", "student_learning"})]
    return items


def _evidence_visible(item: dict, allowed: set[str]) -> bool:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    visibility = metadata.get("visibility") or item.get("visibility") or []
    values = visibility if isinstance(visibility, list) else [visibility]
    visible_to = {str(value) for value in values if value}
    return not visible_to or bool(visible_to & allowed)


def _graph_for_visibility(items: list[dict], visibility: str) -> list[dict]:
    if visibility == "patient_safe":
        return []
    if visibility == "student_learning":
        return [item for item in items if item.get("visibility") == "student_learning"]
    return items


def _memory_for_visibility(items: list[dict], visibility: str) -> list[dict]:
    if visibility == "patient_safe":
        return [item for item in items if item.get("owner_role") == "patient"]
    if visibility == "student_learning":
        return [item for item in items if item.get("owner_role") == "student"]
    return items
