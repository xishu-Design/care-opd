from time import perf_counter

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import MCPToolCall, MemoryAccessLog, User
from app.harness.trace import record_event
from app.mcp.adapter_registry import AdapterRegistry
from app.mcp.contracts import ToolContext, ToolRequest, ToolResult
from app.mcp.registry import get_tool_spec
from app.services.safety_service import post_check_tool_result, pre_check_tool_request, sanitize_patient_memory_context


SENSITIVE_ARGUMENT_KEYS = {"password", "id_card", "phone", "address", "full_medical_record"}
ROLE_MEMORY_TOOLS = {
    "mock_search_role_memory",
    "search_role_memory",
    "mock_write_role_memory",
    "write_role_memory",
    "mock_get_doctor_preferences",
    "get_doctor_preferences",
    "mock_get_patient_explanation_preference",
    "get_patient_explanation_preference",
    "mock_get_student_progress_memory",
    "get_student_progress_memory",
    "mock_update_student_weakness",
    "update_student_weakness",
}


class MCPGateway:
    def __init__(self, db: Session):
        self.db = db

    def call(self, user: User, tool_name: str, arguments: dict | None = None) -> dict:
        arguments = arguments or {}
        spec = get_tool_spec(tool_name)
        task_type = arguments.pop("task_type", None) or (spec.allowed_task_types[0] if spec else "generic")
        context = ToolContext(
            user_id=user.id,
            role=user.role,
            task_type=task_type,
            patient_id=arguments.get("patient_id"),
            prescription_id=arguments.get("prescription_id"),
            learning_case_id=arguments.get("learning_case_id"),
        )
        result = self.execute(ToolRequest(tool_name=tool_name, arguments=arguments, context=context))
        payload = result.model_dump()
        payload["result"] = result.result
        payload["reason"] = result.error_code or ("allowed" if result.allowed else "tool_not_allowed_for_role")
        return payload

    def execute(self, request: ToolRequest) -> ToolResult:
        started = perf_counter()
        spec = get_tool_spec(request.tool_name)
        if not spec:
            result = ToolResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                allowed=False,
                success=False,
                data=None,
                risk_level="high",
                error_code="TOOL_NOT_REGISTERED",
                message=f"Tool {request.tool_name} is not registered",
                metadata={"blocked_by": "mcp_gateway"},
            )
            self._log(request, result, started)
            return result

        denied = self._validate_request(request, spec)
        if denied:
            self._log(request, denied, started)
            return denied

        precheck = pre_check_tool_request(spec, request)
        if precheck:
            self._log(request, precheck, started)
            return precheck

        try:
            data, adapter_metadata = AdapterRegistry(self.db).execute(spec, request)
            if (
                request.context.role == "patient"
                and request.tool_name in ROLE_MEMORY_TOOLS
                and isinstance(data, dict)
                and not data.get("error_code")
            ):
                data, filtered_count = sanitize_patient_memory_context(data)
                adapter_metadata = {**adapter_metadata, "patient_memory_filtered_count": filtered_count}
            if isinstance(data, dict) and data.get("error_code"):
                result = ToolResult(
                    request_id=request.request_id,
                    tool_name=request.tool_name,
                    allowed=False,
                    success=False,
                    data=None,
                    risk_level=spec.risk_level,
                    requires_confirmation=spec.requires_confirmation,
                    error_code=data["error_code"],
                    message=data.get("message", "tool returned a guarded error"),
                    metadata={**adapter_metadata, "blocked_by": "adapter"},
                )
            else:
                evidence = data.get("evidence", []) if isinstance(data, dict) else []
                metadata = {**adapter_metadata, "latency_ms": int((perf_counter() - started) * 1000)}
                result = ToolResult(
                    request_id=request.request_id,
                    tool_name=request.tool_name,
                    allowed=True,
                    success=True,
                    data=data,
                    evidence=evidence,
                    risk_level=data.get("risk_level", spec.risk_level) if isinstance(data, dict) else spec.risk_level,
                    requires_confirmation=spec.requires_confirmation,
                    message="",
                    metadata=metadata,
                )
                output_error = self._validate_output(result, spec)
                if output_error:
                    result = output_error
        except HTTPException as exc:
            result = ToolResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                allowed=True,
                success=False,
                data=None,
                risk_level=spec.risk_level,
                requires_confirmation=spec.requires_confirmation,
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
                metadata={"blocked_by": "adapter", "degraded": True, "degraded_reason": "adapter_http_error"},
            )
        except Exception as exc:
            exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
            result = ToolResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                allowed=True,
                success=False,
                data=None,
                risk_level=spec.risk_level,
                requires_confirmation=spec.requires_confirmation,
                error_code="TOOL_EXECUTION_ERROR",
                message=str(exc).strip() or exception_type,
                metadata={
                    "blocked_by": "adapter",
                    "degraded": True,
                    "degraded_reason": "adapter_execution_error",
                    "exception_type": exception_type,
                },
            )

        postcheck = post_check_tool_result(result, request)
        if postcheck:
            result = postcheck
        self._log(request, result, started)
        return result

    def _validate_request(self, request: ToolRequest, spec) -> ToolResult | None:
        if request.context.role not in spec.allowed_roles:
            return self._deny(request, spec.risk_level, "TOOL_NOT_ALLOWED_FOR_ROLE", f"{request.context.role} role cannot call {request.tool_name}")
        if request.context.task_type not in spec.allowed_task_types:
            return self._deny(request, spec.risk_level, "TASK_TYPE_NOT_ALLOWED", f"{request.context.task_type} cannot call {request.tool_name}")
        for key in spec.input_schema.get("required", []):
            if key not in request.arguments and getattr(request.context, key, None) is None:
                return self._deny(request, spec.risk_level, "INPUT_SCHEMA_VALIDATION_FAILED", f"Missing required argument: {key}")
        if request.tool_name in ROLE_MEMORY_TOOLS and not self._memory_scope_allowed(request):
            self._log_memory_access(request, False, "MEMORY_SCOPE_VIOLATION")
            return self._deny(request, "high", "MEMORY_SCOPE_VIOLATION", "cross-role memory access is not allowed")
        if request.context.role == "doctor" and request.context.patient_id and request.context.resource_scope.is_doctor_bound_to_patient is False:
            return self._deny(request, "high", "DOCTOR_NOT_BOUND_TO_PATIENT", "doctor is not bound to this patient")
        if request.context.role == "patient" and request.context.patient_id and request.context.resource_scope.is_patient_self is False:
            return self._deny(request, "high", "PATIENT_RESOURCE_SCOPE_VIOLATION", "patient can only access own resources")
        if request.context.role == "student" and request.context.learning_case_id and request.context.resource_scope.is_learning_case_deidentified is False:
            return self._deny(request, "high", "LEARNING_CASE_NOT_DEIDENTIFIED", "student can only access deidentified learning cases")
        return None

    def _validate_output(self, result: ToolResult, spec) -> ToolResult | None:
        data = result.data or {}
        for key in spec.output_schema.get("required", []):
            if key not in data:
                return ToolResult(
                    request_id=result.request_id,
                    tool_name=result.tool_name,
                    allowed=True,
                    success=False,
                    data=None,
                    risk_level=spec.risk_level,
                    requires_confirmation=spec.requires_confirmation,
                    error_code="OUTPUT_SCHEMA_VALIDATION_FAILED",
                    message=f"Missing required output: {key}",
                    metadata={**result.metadata, "blocked_by": "mcp_gateway"},
                )
        return None

    def _deny(self, request: ToolRequest, risk_level: str, error_code: str, message: str) -> ToolResult:
        return ToolResult(
            request_id=request.request_id,
            tool_name=request.tool_name,
            allowed=False,
            success=False,
            data=None,
            risk_level=risk_level,
            requires_confirmation=False,
            error_code=error_code,
            message=message,
            metadata={"blocked_by": "mcp_gateway"},
        )

    def _log(self, request: ToolRequest, result: ToolResult, started: float) -> None:
        latency_ms = int((perf_counter() - started) * 1000)
        self.db.add(
            MCPToolCall(
                request_id=request.request_id,
                user_id=request.context.user_id,
                role=request.context.role,
                session_id=request.context.session_id,
                trace_id=request.context.trace_id,
                tool_name=request.tool_name,
                allowed=result.allowed,
                success=result.success,
                risk_level=result.risk_level,
                requires_confirmation=result.requires_confirmation,
                reason=result.error_code or ("allowed" if result.allowed else "denied"),
                arguments_summary=_summarize(request.arguments),
                result_summary=_summarize(result.data or {}),
                error_code=result.error_code,
                blocked_by=result.metadata.get("blocked_by"),
                backend=result.metadata.get("backend"),
                degraded=bool(result.metadata.get("degraded", False)),
                fallback_used=bool(result.metadata.get("fallback_used", False)),
                fallback_reason=result.metadata.get("fallback_reason"),
                latency_ms=result.metadata.get("latency_ms", latency_ms),
            )
        )
        record_event(
            self.db,
            request.context.trace_id,
            node_name="MCPToolNode",
            event_type="tool_call",
            status="success" if result.success else "blocked",
            input_summary=_summarize(request.arguments),
            output_summary={
                "allowed": result.allowed,
                "success": result.success,
                "backend": result.metadata.get("backend"),
                "degraded": bool(result.metadata.get("degraded", False)),
                "fallback_used": bool(result.metadata.get("fallback_used", False)),
            },
            tool_name=request.tool_name,
            error_code=result.error_code,
            duration_ms=result.metadata.get("latency_ms", latency_ms),
            metadata=result.metadata,
        )
        self.db.commit()

    def _memory_scope_allowed(self, request: ToolRequest) -> bool:
        owner_role = request.arguments.get("owner_role", request.context.role)
        owner_user_id = int(request.arguments.get("owner_user_id", request.context.user_id))
        return owner_role == request.context.role and owner_user_id == request.context.user_id

    def _log_memory_access(self, request: ToolRequest, allowed: bool, reason: str) -> None:
        self.db.add(
            MemoryAccessLog(
                requester_user_id=request.context.user_id,
                requester_role=request.context.role,
                owner_user_id=request.arguments.get("owner_user_id"),
                owner_role=request.arguments.get("owner_role"),
                tool_name=request.tool_name,
                allowed=allowed,
                reason=reason,
            )
        )
        self.db.flush()


def _summarize(payload: dict) -> dict:
    summary = {}
    for key, value in payload.items():
        if key in SENSITIVE_ARGUMENT_KEYS:
            summary[key] = "<redacted>"
        elif isinstance(value, str) and len(value) > 160:
            summary[key] = value[:160] + "..."
        else:
            summary[key] = value
    return summary
