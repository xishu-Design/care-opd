from app.mcp.contracts import ToolSpec


def _spec(
    name: str,
    description: str,
    server: str,
    roles: list[str],
    tasks: list[str],
    risk_level: str = "low",
    requires_confirmation: bool = False,
    input_required: list[str] | None = None,
    output_required: list[str] | None = None,
    deprecated: bool = False,
    remove_after_phase: int | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        server=server,
        allowed_roles=roles,
        allowed_task_types=tasks,
        input_schema={"required": input_required or []},
        output_schema={"required": output_required or []},
        risk_level=risk_level,
        requires_confirmation=requires_confirmation,
        deprecated=deprecated,
        remove_after_phase=remove_after_phase,
    )


TOOL_SPECS: dict[str, ToolSpec] = {
    # Auth tools
    "verify_user_role": _spec("verify_user_role", "Verify user role", "AuthMockServer", ["doctor", "patient", "student"], ["generic", "auth"]),
    "check_doctor_patient_binding": _spec("check_doctor_patient_binding", "Check doctor-patient binding", "AuthMockServer", ["doctor"], ["patient_access", "prescription_review"], input_required=["patient_id"]),
    "check_patient_ownership": _spec("check_patient_ownership", "Check patient self ownership", "AuthMockServer", ["patient"], ["patient_profile", "patient_prescription", "patient_education"]),
    "check_learning_case_access": _spec("check_learning_case_access", "Check learning case access", "AuthMockServer", ["student"], ["student_training"], input_required=["learning_case_id"]),
    # Patient record tools
    "get_patient_profile": _spec("get_patient_profile", "Get bound patient profile", "PatientRecordMockServer", ["doctor"], ["patient_access", "prescription_review"], input_required=["patient_id"]),
    "get_my_profile": _spec("get_my_profile", "Get own patient profile", "PatientRecordMockServer", ["patient"], ["patient_profile", "patient_education"]),
    "get_patient_medical_history": _spec("get_patient_medical_history", "Get patient medical history", "PatientRecordMockServer", ["doctor"], ["patient_access", "prescription_review"], input_required=["patient_id"]),
    "get_patient_allergy_history": _spec("get_patient_allergy_history", "Get patient allergy history", "PatientRecordMockServer", ["doctor"], ["patient_access", "prescription_review"], input_required=["patient_id"]),
    "get_bound_patient_roster": _spec(
        "get_bound_patient_roster",
        "Get the current doctor's actively bound patient roster",
        "PatientRecordMockServer",
        ["doctor"],
        ["doctor_ai_chat"],
        output_required=["doctor_patient_roster"],
    ),
    # Prescription tools
    "create_prescription_draft": _spec("create_prescription_draft", "Create prescription draft", "PrescriptionMockServer", ["doctor"], ["prescription_draft"], "medium", True, ["patient_id"]),
    "parse_prescription": _spec("parse_prescription", "Parse prescription items", "PrescriptionMockServer", ["doctor"], ["prescription_review"], "medium", False, ["prescription_id"], ["parsed_items"]),
    "get_my_published_prescriptions": _spec("get_my_published_prescriptions", "Get own published prescriptions", "PrescriptionMockServer", ["patient"], ["patient_prescription", "patient_education"]),
    "get_prescription_detail": _spec("get_prescription_detail", "Get prescription detail", "PrescriptionMockServer", ["doctor", "patient"], ["prescription_review", "patient_prescription"], input_required=["prescription_id"]),
    "generate_patient_instruction": _spec("generate_patient_instruction", "Generate patient instruction", "PrescriptionMockServer", ["doctor"], ["prescription_review"], "medium", True, ["prescription_id"]),
    "generate_diet_precaution": _spec("generate_diet_precaution", "Generate diet precaution", "PrescriptionMockServer", ["doctor"], ["prescription_review"], "medium", True, ["prescription_id"]),
    # Rule engine tools
    "run_prescription_review": _spec("run_prescription_review", "Run structured prescription safety review", "RuleEngineServer", ["doctor"], ["prescription_review"], "high", True, ["prescription_id"], ["rule_findings"]),
    "run_mock_prescription_review": _spec("run_mock_prescription_review", "Run mock prescription safety review", "RuleEngineMockServer", ["doctor"], ["prescription_review"], "high", True, ["prescription_id"], ["rule_findings"], True, 3),
    "check_mock_dosage_risk": _spec("check_mock_dosage_risk", "Check mock dosage risk", "RuleEngineMockServer", ["doctor"], ["prescription_review"], "medium", True, deprecated=True, remove_after_phase=3),
    "check_mock_missing_information": _spec("check_mock_missing_information", "Check missing information", "RuleEngineMockServer", ["doctor"], ["prescription_review"], "medium", True, deprecated=True, remove_after_phase=3),
    # Local knowledge tools plus Phase 2 mock aliases
    "search_evidence": _spec("search_evidence", "Search grounded local evidence", "KnowledgeServer", ["doctor", "student"], ["prescription_review", "student_training", "doctor_ai_chat", "student_ai_chat"], input_required=["query"], output_required=["evidence"]),
    "search_herb": _spec("search_herb", "Search local herb knowledge", "KnowledgeServer", ["doctor", "patient", "student"], ["prescription_review", "patient_education", "student_training", "doctor_ai_chat", "student_ai_chat"], input_required=["query"], output_required=["evidence"]),
    "search_formula": _spec("search_formula", "Search local formula knowledge", "KnowledgeServer", ["doctor", "patient", "student"], ["prescription_review", "patient_education", "student_training", "doctor_ai_chat", "student_ai_chat"], input_required=["query"], output_required=["evidence"]),
    "search_contraindication": _spec("search_contraindication", "Search local contraindication knowledge", "KnowledgeServer", ["doctor", "student"], ["prescription_review", "student_training", "doctor_ai_chat", "student_ai_chat"], input_required=["query"], output_required=["evidence"]),
    "search_diet_knowledge": _spec("search_diet_knowledge", "Search local diet precaution knowledge", "KnowledgeServer", ["doctor", "patient"], ["prescription_review", "patient_education"], input_required=["query"], output_required=["evidence"]),
    "public_tcm_education": _spec("public_tcm_education", "Public TCM education with safe boundaries", "KnowledgeServer", ["patient"], ["patient_education", "generic"], input_required=["query"], output_required=["evidence", "answer"]),
    "mock_search_herb": _spec("mock_search_herb", "Search mock herb knowledge", "LightRAGMockServer", ["doctor", "patient", "student"], ["prescription_review", "patient_education", "student_training"], input_required=["query"], output_required=["evidence"], deprecated=True, remove_after_phase=3),
    "mock_search_formula": _spec("mock_search_formula", "Search mock formula knowledge", "LightRAGMockServer", ["doctor", "patient", "student"], ["prescription_review", "patient_education", "student_training"], input_required=["query"], output_required=["evidence"], deprecated=True, remove_after_phase=3),
    "mock_search_contraindication": _spec("mock_search_contraindication", "Search mock contraindication knowledge", "LightRAGMockServer", ["doctor", "student"], ["prescription_review", "student_training"], input_required=["query"], output_required=["evidence"], deprecated=True, remove_after_phase=3),
    "mock_search_diet_knowledge": _spec("mock_search_diet_knowledge", "Search mock diet knowledge", "LightRAGMockServer", ["doctor", "patient"], ["prescription_review", "patient_education"], input_required=["query"], output_required=["evidence"], deprecated=True, remove_after_phase=3),
    "mock_search_evidence": _spec("mock_search_evidence", "Search mock evidence", "LightRAGMockServer", ["doctor", "student"], ["prescription_review", "student_training"], input_required=["query"], output_required=["evidence"], deprecated=True, remove_after_phase=3),
    "mock_public_tcm_education": _spec("mock_public_tcm_education", "Public TCM education", "LightRAGMockServer", ["patient"], ["patient_education"], input_required=["query"], output_required=["evidence", "answer"], deprecated=True, remove_after_phase=3),
    # Graphiti mock tools
    "create_case_graph": _spec("create_case_graph", "Create persistent case graph", "GraphServer", ["doctor"], ["prescription_review"]),
    "update_case_graph": _spec("update_case_graph", "Update persistent case graph", "GraphServer", ["doctor"], ["prescription_review"]),
    "add_prescription_to_graph": _spec("add_prescription_to_graph", "Add prescription to graph", "GraphServer", ["doctor"], ["prescription_review"], input_required=["prescription_id"]),
    "add_risk_to_graph": _spec("add_risk_to_graph", "Add risk to graph", "GraphServer", ["doctor"], ["prescription_review"]),
    "get_reasoning_path": _spec("get_reasoning_path", "Get persistent reasoning path", "GraphServer", ["doctor", "student"], ["prescription_review", "student_training", "doctor_ai_chat", "student_ai_chat"], output_required=["graph_reasoning_path"]),
    "export_deidentified_graph": _spec("export_deidentified_graph", "Export deidentified graph", "GraphServer", ["student", "doctor"], ["student_training", "prescription_review"], output_required=["graph_reasoning_path"]),
    "mock_create_case_graph": _spec("mock_create_case_graph", "Create mock case graph", "GraphitiMockServer", ["doctor"], ["prescription_review"], deprecated=True, remove_after_phase=3),
    "mock_update_case_graph": _spec("mock_update_case_graph", "Update mock case graph", "GraphitiMockServer", ["doctor"], ["prescription_review"], deprecated=True, remove_after_phase=3),
    "mock_add_prescription_to_graph": _spec("mock_add_prescription_to_graph", "Add prescription to mock graph", "GraphitiMockServer", ["doctor"], ["prescription_review"], input_required=["prescription_id"], deprecated=True, remove_after_phase=3),
    "mock_add_risk_to_graph": _spec("mock_add_risk_to_graph", "Add risk to mock graph", "GraphitiMockServer", ["doctor"], ["prescription_review"], deprecated=True, remove_after_phase=3),
    "mock_get_reasoning_path": _spec("mock_get_reasoning_path", "Get mock reasoning path", "GraphitiMockServer", ["doctor", "student"], ["prescription_review", "student_training"], output_required=["graph_reasoning_path"], deprecated=True, remove_after_phase=3),
    "mock_export_deidentified_graph": _spec("mock_export_deidentified_graph", "Export deidentified graph", "GraphitiMockServer", ["student", "doctor"], ["student_training", "prescription_review"], output_required=["graph_reasoning_path"], deprecated=True, remove_after_phase=3),
    # Persistent memory tools plus Phase 2 mock aliases
    "write_role_memory": _spec("write_role_memory", "Write role-scoped persistent memory", "MemoryServer", ["doctor", "patient", "student"], ["prescription_review", "patient_education", "student_training", "doctor_ai_chat", "student_ai_chat"], input_required=["memory_type"]),
    "search_role_memory": _spec("search_role_memory", "Search role-scoped persistent memory", "MemoryServer", ["doctor", "patient", "student"], ["prescription_review", "patient_education", "student_training", "doctor_ai_chat", "student_ai_chat"], input_required=["owner_role"]),
    "get_doctor_preferences": _spec("get_doctor_preferences", "Get doctor preferences", "MemoryServer", ["doctor"], ["prescription_review"], output_required=["memory_used"]),
    "get_patient_explanation_preference": _spec("get_patient_explanation_preference", "Get patient explanation preference", "MemoryServer", ["patient"], ["patient_education"], output_required=["memory_used"]),
    "get_student_progress_memory": _spec("get_student_progress_memory", "Get student progress memory", "MemoryServer", ["student"], ["student_training"], output_required=["memory_used"]),
    "update_student_weakness": _spec("update_student_weakness", "Update student weakness memory", "MemoryServer", ["student"], ["student_training"], input_required=["weakness"]),
    "mock_write_role_memory": _spec("mock_write_role_memory", "Write role-scoped mock memory", "Mem0MockServer", ["doctor", "patient", "student"], ["prescription_review", "patient_education", "student_training"], input_required=["memory_type"], deprecated=True, remove_after_phase=3),
    "mock_search_role_memory": _spec("mock_search_role_memory", "Search role-scoped mock memory", "Mem0MockServer", ["doctor", "patient", "student"], ["prescription_review", "patient_education", "student_training"], input_required=["owner_role"], deprecated=True, remove_after_phase=3),
    "mock_get_doctor_preferences": _spec("mock_get_doctor_preferences", "Get doctor preferences", "Mem0MockServer", ["doctor"], ["prescription_review"], output_required=["memory_used"], deprecated=True, remove_after_phase=3),
    "mock_get_patient_explanation_preference": _spec("mock_get_patient_explanation_preference", "Get patient explanation preference", "Mem0MockServer", ["patient"], ["patient_education"], output_required=["memory_used"], deprecated=True, remove_after_phase=3),
    "mock_get_student_progress_memory": _spec("mock_get_student_progress_memory", "Get student progress memory", "Mem0MockServer", ["student"], ["student_training"], output_required=["memory_used"], deprecated=True, remove_after_phase=3),
    "mock_update_student_weakness": _spec("mock_update_student_weakness", "Update student weakness memory", "Mem0MockServer", ["student"], ["student_training"], input_required=["weakness"], deprecated=True, remove_after_phase=3),
    # Learning tools
    "create_deidentified_learning_case": _spec("create_deidentified_learning_case", "Create deidentified learning case", "LearningMockServer", ["doctor"], ["prescription_review"], "medium", True, ["prescription_id"]),
    "list_learning_cases": _spec("list_learning_cases", "List learning cases", "LearningMockServer", ["student"], ["student_training"]),
    "get_learning_case": _spec("get_learning_case", "Get learning case", "LearningMockServer", ["student"], ["student_training"], input_required=["learning_case_id"]),
    "submit_review_answer": _spec("submit_review_answer", "Submit review answer", "LearningMockServer", ["student"], ["student_training"], input_required=["learning_case_id"]),
    "get_standard_analysis": _spec("get_standard_analysis", "Get standard analysis", "LearningMockServer", ["student"], ["student_training"], input_required=["learning_case_id"]),
    "generate_mock_knowledge_cards": _spec("generate_mock_knowledge_cards", "Generate mock knowledge cards", "LearningMockServer", ["student"], ["student_training"], input_required=["learning_case_id"]),
    "generate_mock_quiz": _spec("generate_mock_quiz", "Generate mock quiz", "LearningMockServer", ["student"], ["student_training"], input_required=["learning_case_id"]),
}


def get_tool_spec(tool_name: str) -> ToolSpec | None:
    return TOOL_SPECS.get(tool_name)


def all_tool_specs() -> list[ToolSpec]:
    return sorted(TOOL_SPECS.values(), key=lambda spec: spec.name)


def registered_tools() -> dict[str, list[str]]:
    tools_by_role: dict[str, list[str]] = {"doctor": [], "patient": [], "student": []}
    for spec in TOOL_SPECS.values():
        for role in spec.allowed_roles:
            tools_by_role.setdefault(role, []).append(spec.name)
    return {role: sorted(set(tools)) for role, tools in tools_by_role.items()}
