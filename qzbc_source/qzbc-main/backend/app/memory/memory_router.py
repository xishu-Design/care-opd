from pydantic import BaseModel, Field


class MemoryRouteRequest(BaseModel):
    role: str
    task_type: str
    risk_level: str | None = None
    token_budget: int = 4000
    patient_id: int | None = None
    prescription_id: int | None = None
    learning_case_id: int | None = None


class MemoryPlanItem(BaseModel):
    memory_layer: str
    tool_name: str
    reason: str


class MemoryPlan(BaseModel):
    memory_plan: list[MemoryPlanItem] = Field(default_factory=list)


class MemoryRouter:
    def plan(self, request: MemoryRouteRequest) -> MemoryPlan:
        if request.role == "doctor" and request.task_type == "prescription_review":
            return MemoryPlan(
                memory_plan=[
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_evidence", reason="doctor prescription review requires evidence retrieval"),
                    MemoryPlanItem(memory_layer="graph", tool_name="get_reasoning_path", reason="doctor review requires context reasoning path"),
                    MemoryPlanItem(memory_layer="memory", tool_name="search_role_memory", reason="doctor output preference may affect format"),
                ]
            )
        if request.role == "patient" and request.task_type == "patient_education":
            return MemoryPlan(
                memory_plan=[
                    MemoryPlanItem(memory_layer="knowledge", tool_name="public_tcm_education", reason="patient education requires public knowledge only"),
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_diet_knowledge", reason="patient education may need public diet precautions"),
                    MemoryPlanItem(memory_layer="memory", tool_name="search_role_memory", reason="patient explanation preference may affect wording"),
                ]
            )
        if request.role == "student" and request.task_type == "student_training":
            return MemoryPlan(
                memory_plan=[
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_evidence", reason="student training requires teaching evidence"),
                    MemoryPlanItem(memory_layer="graph", tool_name="get_reasoning_path", reason="student can only inspect deidentified graph paths"),
                    MemoryPlanItem(memory_layer="memory", tool_name="search_role_memory", reason="student progress memory supports review guidance"),
                ]
            )
        if request.role == "doctor" and request.task_type == "doctor_ai_chat":
            return MemoryPlan(
                memory_plan=[
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_evidence", reason="doctor questions require traceable professional evidence"),
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_herb", reason="doctor questions may involve structured herb knowledge"),
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_formula", reason="doctor questions may involve structured formula knowledge"),
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_contraindication", reason="doctor questions require safety boundaries"),
                    MemoryPlanItem(memory_layer="graph", tool_name="get_reasoning_path", reason="doctor questions benefit from knowledge relationship paths"),
                    MemoryPlanItem(memory_layer="memory", tool_name="search_role_memory", reason="doctor long-term preferences and professional focus support continuity"),
                ]
            )
        if request.role == "student" and request.task_type == "student_ai_chat":
            return MemoryPlan(
                memory_plan=[
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_evidence", reason="student questions require teaching evidence"),
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_herb", reason="student questions may involve herb concepts"),
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_formula", reason="student questions may involve formula concepts"),
                    MemoryPlanItem(memory_layer="knowledge", tool_name="search_contraindication", reason="student learning should include safety concepts"),
                    MemoryPlanItem(memory_layer="graph", tool_name="get_reasoning_path", reason="student questions benefit from deidentified knowledge paths"),
                    MemoryPlanItem(memory_layer="memory", tool_name="search_role_memory", reason="student long-term goals and weaknesses support adaptive teaching"),
                ]
            )
        return MemoryPlan()
