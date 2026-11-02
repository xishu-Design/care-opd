from app.evaluation.runners.base import BaseEvaluationRunner
from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult
from app.mcp.contracts import ResourceScope, ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway


class MCPEvaluationRunner(BaseEvaluationRunner):
    def run(self, cases: list[EvaluationCase]) -> list[EvaluationCaseResult]:
        return [self.timed(case, self._case) for case in cases]

    def _case(self, case: EvaluationCase) -> EvaluationCaseResult:
        if "mcp_permission_check" in self.disabled:
            return self.result(case, False, {"allowed": True, "logged": False, "ablation": "mcp_permission_check"})
        role = str(case.input["role"])
        user = self.seed[role]
        context = ToolContext(
            user_id=user.id,
            role=role,
            task_type=str(case.input["task_type"]),
            patient_id=self.seed["patient"].id,
            prescription_id=case.input.get("arguments", {}).get("prescription_id"),
            resource_scope=ResourceScope(is_patient_self=role == "patient"),
        )
        arguments = dict(case.input.get("arguments", {}))
        arguments["patient_id"] = self.seed["patient"].id
        result = MCPGateway(self.db).execute(ToolRequest(tool_name=str(case.input["tool_name"]), arguments=arguments, context=context))
        passed = result.allowed is False and bool(result.error_code)
        return self.result(case, passed, {"allowed": result.allowed, "logged": True, "error_code": result.error_code, "blocked_by": result.metadata.get("blocked_by")})
