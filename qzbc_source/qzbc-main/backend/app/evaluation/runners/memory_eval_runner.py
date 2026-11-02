from app.evaluation.runners.base import BaseEvaluationRunner
from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult
from app.mcp.contracts import ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway


class MemoryEvaluationRunner(BaseEvaluationRunner):
    def run(self, cases: list[EvaluationCase]) -> list[EvaluationCaseResult]:
        return [self.timed(case, self._case) for case in cases]

    def _case(self, case: EvaluationCase) -> EvaluationCaseResult:
        if "role_memory" in self.disabled:
            return self.result(case, False, {"cross_role_blocked": False, "same_role_available": False, "ablation": "role_memory"})
        owner_role = str(case.input["owner_role"])
        requester_role = str(case.input["requester_role"])
        owner = self.seed[owner_role]
        requester = self.seed[requester_role]
        gateway = MCPGateway(self.db)
        write = gateway.execute(
            ToolRequest(
                tool_name="write_role_memory",
                arguments={"owner_role": owner_role, "owner_user_id": owner.id, "memory_type": case.input["memory_type"], "content": {"case_id": case.case_id}},
                context=ToolContext(user_id=owner.id, role=owner_role, task_type=self._task(owner_role)),
            )
        )
        own_read = gateway.execute(
            ToolRequest(
                tool_name="search_role_memory",
                arguments={"owner_role": owner_role, "owner_user_id": owner.id},
                context=ToolContext(user_id=owner.id, role=owner_role, task_type=self._task(owner_role)),
            )
        )
        cross_read = gateway.execute(
            ToolRequest(
                tool_name="search_role_memory",
                arguments={"owner_role": owner_role, "owner_user_id": owner.id},
                context=ToolContext(user_id=requester.id, role=requester_role, task_type=self._task(requester_role)),
            )
        )
        own_text = str(own_read.data)
        own_available = write.success and own_read.success and case.case_id in own_text
        cross_blocked = cross_read.success is False and cross_read.error_code == "MEMORY_SCOPE_VIOLATION"
        return self.result(case, own_available and cross_blocked, {"same_role_available": own_available, "cross_role_blocked": cross_blocked, "cross_error_code": cross_read.error_code})

    @staticmethod
    def _task(role: str) -> str:
        return "prescription_review" if role == "doctor" else "patient_education" if role == "patient" else "student_training"
