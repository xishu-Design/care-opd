from sqlalchemy.orm import Session

from app.agents.nodes.common import add_error, append_node, copy_state
from app.agents.state import AgentState
from app.mcp.contracts import ResourceScope, ToolContext, ToolRequest, ToolResult, new_request_id
from app.mcp.gateway import MCPGateway


PATIENT_ALLOWED_TOOLS = {"public_tcm_education", "search_diet_knowledge", "search_role_memory"}


class MCPToolNode:
    node_name = "MCPToolNode"

    def __init__(self, db: Session):
        self.db = db

    def __call__(self, state: AgentState) -> AgentState:
        state = copy_state(state)
        context = ToolContext(
            user_id=state["user_id"],
            role=state["role"],
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
            task_type=state["task_type"],
            patient_id=state.get("patient_id"),
            prescription_id=state.get("prescription_id"),
            learning_case_id=state.get("learning_case_id"),
            resource_scope=ResourceScope.model_validate(state.get("resource_scope") or {}),
        )
        if state.get("private_eval_evidence_control"):
            results = self._execute_private_eval_evidence_control(state)
            state["tool_results"] = [result.model_dump() for result in results]
        elif state["role"] == "doctor" and state["task_type"] == "prescription_review":
            results = self._execute_doctor_review(state, context)
        else:
            requests = self._tool_requests_for_state(state, context)
            state["tool_requests"] = [
                {"tool_name": request.tool_name, "arguments": request.arguments, "request_id": request.request_id}
                for request in requests
            ]

            gateway = MCPGateway(self.db)
            results = [gateway.execute(request) for request in requests]
            state["tool_results"] = [result.model_dump() for result in results]
            self._extract_structured_outputs(state, results)

        failed = [result for result in results if not result.success]
        request_count = len(state.get("tool_requests", []))
        state["critical_error"] = bool(request_count) and len(failed) == len(results)
        state["degraded"] = bool(state.get("degraded")) or any(result.metadata.get("degraded") for result in results)
        state["fallback_used"] = bool(state.get("fallback_used")) or any(result.metadata.get("fallback_used") for result in results)
        for result in failed:
            state = add_error(state, result.error_code or "TOOL_FAILED", result.message or f"{result.tool_name} failed")

        return append_node(
            self.db,
            state,
            self.node_name,
            status="blocked" if state["critical_error"] else "success",
            output_summary={
                "tools": [result.tool_name for result in results],
                "failed": [result.tool_name for result in failed],
                "critical_error": state["critical_error"],
                "degraded": state["degraded"],
                "fallback_used": state["fallback_used"],
            },
            error_code="MCP_TOOL_CRITICAL_ERROR" if state["critical_error"] else None,
        )

    def _execute_private_eval_evidence_control(self, state: AgentState) -> list[ToolResult]:
        """Bypass every live adapter and inject only the pre-authorized evidence state."""

        control = dict(state.get("private_eval_evidence_control") or {})
        condition = str(control.get("condition") or "")
        if condition not in {"clean", "evidence_unavailable_counterfactual"}:
            raise RuntimeError("invalid private evaluation evidence condition in AgentState")
        expected_count = 1 if condition == "clean" else 0
        if int(control.get("expected_retrieved_evidence_count", -1)) != expected_count:
            raise RuntimeError("private evaluation expected evidence count is inconsistent")

        request_id = new_request_id()
        audit = {
            "protocol": control["protocol"],
            "condition": condition,
            "task_token": control["task_token"],
            "case_material_sha256": control["case_material_sha256"],
            "expected_retrieved_evidence_count": expected_count,
            "live_adapter_calls_bypassed": True,
            "graphiti_bypassed": True,
            "mem0_bypassed": True,
        }
        state["tool_requests"] = [
            {
                "request_id": request_id,
                "tool_name": "private_eval_locked_evidence",
                "arguments": {
                    "condition": condition,
                    "case_material_sha256": control["case_material_sha256"],
                },
            }
        ]
        evidence: list[dict] = []
        if condition == "clean":
            evidence.append(
                {
                    "source_id": f"PRIVATE_EVAL_LOCKED_{control['case_material_sha256'][:16]}",
                    "kind": "locked_case_material",
                    "content": control["case_material"],
                    "summary": control["case_material"],
                    "snippet": control["case_material"],
                    "visibility": ["doctor", "patient", "student"],
                    "metadata": {
                        "visibility": ["doctor", "patient", "student"],
                        "private_eval_locked": True,
                        "case_material_sha256": control["case_material_sha256"],
                    },
                }
            )
        metadata = {
            "backend": "private_eval_locked_evidence",
            "configured_backend": "private_eval_locked_evidence",
            "retrieval_mode": "locked_injection" if evidence else "forced_zero_evidence",
            "degraded": not evidence,
            "degraded_reason": (
                "evidence_unavailable_counterfactual" if not evidence else None
            ),
            "fallback_used": False,
            "qa_retrieval_count": 0,
            "private_eval_evidence_control": audit,
        }
        return [
            ToolResult(
                request_id=request_id,
                tool_name="private_eval_locked_evidence",
                allowed=True,
                success=True,
                data={"evidence": evidence},
                evidence=evidence,
                metadata=metadata,
            )
        ]

    def _tool_requests_for_state(self, state: AgentState, context: ToolContext) -> list[ToolRequest]:
        if state["role"] == "doctor" and state["task_type"] == "prescription_review":
            return self._doctor_review_requests(state, context)
        if state["role"] == "patient" and state["task_type"] == "patient_education":
            return self._patient_education_requests(state, context)
        if state["role"] == "student" and state["task_type"] == "student_training":
            return self._student_training_requests(state, context)
        if state["task_type"] in {"doctor_ai_chat", "student_ai_chat"}:
            return self._role_ai_chat_requests(state, context)
        return []

    def _doctor_review_requests(self, state: AgentState, context: ToolContext) -> list[ToolRequest]:
        del state, context
        return []

    def _execute_doctor_review(self, state: AgentState, context: ToolContext) -> list[ToolResult]:
        prescription_id = int(state["prescription_id"])
        initial = [
            ToolRequest(tool_name="parse_prescription", arguments={"prescription_id": prescription_id}, context=context),
            ToolRequest(tool_name="run_prescription_review", arguments={"prescription_id": prescription_id}, context=context),
        ]
        # Evidence queries depend on rule output, so they are appended after the first two tools run.
        gateway = MCPGateway(self.db)
        initial_results = [gateway.execute(request) for request in initial]
        state["tool_requests"] = [
            {"tool_name": request.tool_name, "arguments": request.arguments, "request_id": request.request_id}
            for request in initial
        ]
        state["tool_results"] = [result.model_dump() for result in initial_results]
        self._extract_structured_outputs(state, initial_results)

        rule_findings = state.get("input_payload", {}).get("rule_findings", [])
        evidence_queries = [finding.get("evidence_query") for finding in rule_findings if finding.get("evidence_query")]
        if not evidence_queries:
            title = state.get("input_payload", {}).get("prescription_title") or f"Prescription {prescription_id}"
            evidence_queries = [f"{title} 审方证据 用法用量 服药 注意事项"]

        follow_up: list[ToolRequest] = [
            ToolRequest(
                tool_name="search_evidence",
                arguments={"query": query, "prescription_id": prescription_id},
                context=context,
            )
            for query in evidence_queries
        ]
        for item in state.get("memory_plan", []):
            tool_name = item["tool_name"]
            if tool_name == "search_evidence":
                continue
            follow_up.append(ToolRequest(tool_name=tool_name, arguments=self._planned_prescription_arguments(tool_name, state), context=context))

        follow_up_results = [gateway.execute(request) for request in follow_up]
        all_requests = initial + follow_up
        all_results = initial_results + follow_up_results
        state["tool_requests"] = [
            {"tool_name": request.tool_name, "arguments": request.arguments, "request_id": request.request_id}
            for request in all_requests
        ]
        state["tool_results"] = [result.model_dump() for result in all_results]
        self._extract_structured_outputs(state, all_results)
        return all_results

    def _patient_education_requests(self, state: AgentState, context: ToolContext) -> list[ToolRequest]:
        # These are first-party, self-scoped business records. They must be
        # present on every patient question so phrases such as “当前处方” can
        # be resolved without guessing or relying on vector search.
        requests = [
            ToolRequest(tool_name="get_my_profile", arguments={}, context=context),
            ToolRequest(tool_name="get_my_published_prescriptions", arguments={}, context=context),
        ]
        for item in state.get("memory_plan", []):
            tool_name = item["tool_name"]
            if tool_name not in PATIENT_ALLOWED_TOOLS:
                continue
            arguments = {"query": state.get("retrieval_query") or state.get("input_text", "")}
            if tool_name == "search_role_memory":
                arguments.update({"owner_role": state["role"], "owner_user_id": state["user_id"]})
            requests.append(ToolRequest(tool_name=tool_name, arguments=arguments, context=context))
        return requests

    def _student_training_requests(self, state: AgentState, context: ToolContext) -> list[ToolRequest]:
        learning_case_id = int(state["learning_case_id"])
        title = state.get("input_payload", {}).get("learning_case_title") or f"Learning case {learning_case_id}"
        requests = [
            ToolRequest(tool_name="get_learning_case", arguments={"learning_case_id": learning_case_id}, context=context),
            ToolRequest(tool_name="get_standard_analysis", arguments={"learning_case_id": learning_case_id}, context=context),
            ToolRequest(tool_name="generate_mock_knowledge_cards", arguments={"learning_case_id": learning_case_id}, context=context),
            ToolRequest(tool_name="generate_mock_quiz", arguments={"learning_case_id": learning_case_id}, context=context),
        ]
        for item in state.get("memory_plan", []):
            tool_name = item["tool_name"]
            arguments = {"learning_case_id": learning_case_id, "query": f"{title} 学习证据 用法用量 服药 注意事项"}
            if tool_name == "get_reasoning_path":
                arguments["visibility"] = "student_learning"
            if tool_name == "search_role_memory":
                arguments.update({"owner_role": state["role"], "owner_user_id": state["user_id"]})
            requests.append(ToolRequest(tool_name=tool_name, arguments=arguments, context=context))
        return requests

    def _role_ai_chat_requests(self, state: AgentState, context: ToolContext) -> list[ToolRequest]:
        query = state.get("retrieval_query") or state.get("input_text", "")
        requests: list[ToolRequest] = []
        if state["role"] == "doctor":
            # The adapter resolves only bindings owned by the authenticated
            # doctor. No patient id supplied by the model is trusted here.
            requests.append(ToolRequest(tool_name="get_bound_patient_roster", arguments={}, context=context))
        for item in state.get("memory_plan", []):
            tool_name = item["tool_name"]
            arguments: dict = {"query": query}
            if tool_name == "get_reasoning_path":
                arguments["visibility"] = "doctor_internal" if state["role"] == "doctor" else "student_learning"
            if tool_name == "search_role_memory":
                # Long-term memory contains stable role/user preferences, not
                # raw clinical facts.  Include a preference-oriented semantic
                # anchor so a long case narrative cannot drown out the
                # relevant Mem0 item; the current question is retained for
                # task-specific relevance.
                role_anchor = {
                    "doctor": "医生长期回答格式偏好 专业关注 先给结论 专业依据 鉴别复核",
                    "student": "学生长期学习偏好 知识框架 推理依据 复习建议",
                }.get(state["role"], "长期回答偏好")
                arguments.update(
                    {
                        "owner_role": state["role"],
                        "owner_user_id": state["user_id"],
                        "query": f"{role_anchor}。当前问题：{query[:1200]}",
                    }
                )
            requests.append(ToolRequest(tool_name=tool_name, arguments=arguments, context=context))
        return requests

    def _planned_prescription_arguments(self, tool_name: str, state: AgentState) -> dict:
        if tool_name == "get_reasoning_path":
            return {
                "prescription_id": state.get("prescription_id"),
                "patient_id": state.get("patient_id"),
                "visibility": "doctor_internal",
            }
        if tool_name == "search_role_memory":
            return {
                "owner_role": state["role"],
                "owner_user_id": state["user_id"],
                "query": "doctor prescription review preference",
            }
        return {"prescription_id": state.get("prescription_id")}

    def _extract_structured_outputs(self, state: AgentState, results: list[ToolResult]) -> None:
        for result in results:
            data = result.data or {}
            if result.tool_name == "run_prescription_review" and data:
                rule_findings = data.get("rule_findings") or data.get("risk_cards") or []
                state["input_payload"].update(
                    {
                        "risk_level": data.get("risk_level", "red"),
                        "review_summary": data.get("review_summary", "结构化规则引擎未返回结果，必须由医生人工复核。"),
                        "rule_findings": rule_findings,
                    }
                )
            if result.tool_name == "public_tcm_education" and data.get("answer"):
                response = dict(state.get("final_response") or state.get("safety_decision") or {})
                response["safe_reply"] = data["answer"]
                state["final_response"] = response
            if data.get("knowledge_cards"):
                state["input_payload"]["knowledge_cards"] = data["knowledge_cards"]
            if data.get("quiz"):
                state["input_payload"]["quiz"] = data["quiz"]
