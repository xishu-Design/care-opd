from sqlalchemy.orm import Session

from app.agents.graphs.doctor_review_graph import DoctorReviewGraph
from app.agents.graphs.patient_education_graph import PatientEducationGraph
from app.agents.graphs.role_ai_chat_graph import RoleAIChatGraph
from app.agents.graphs.student_training_graph import StudentTrainingGraph


GRAPH_DESCRIPTIONS = {
    "DoctorReviewGraph": {
        "name": "DoctorReviewGraph",
        "role": "doctor",
        "task_type": "prescription_review",
        "version": "0.8.0",
        "nodes": [
            "AuthNode",
            "RouterNode",
            "MemoryRouterNode",
            "MCPToolNode",
            "EvidencePacketNode",
            "LLMGenerateNode",
            "ReviewPersistNode",
            "SafetyPostNode",
            "HumanConfirmNode",
            "ResponseNode",
            "TraceNode",
        ],
    },
    "PatientEducationGraph": {
        "name": "PatientEducationGraph",
        "role": "patient",
        "task_type": "patient_education",
        "version": "0.8.0",
        "nodes": [
            "AuthNode",
            "RouterNode",
            "SafetyPreNode",
            "MemoryRouterNode",
            "MCPToolNode",
            "EvidencePacketNode",
            "LLMGenerateNode",
            "SafetyPostNode",
            "ResponseNode",
            "TraceNode",
        ],
    },
    "StudentTrainingGraph": {
        "name": "StudentTrainingGraph",
        "role": "student",
        "task_type": "student_training",
        "version": "0.8.0",
        "nodes": [
            "AuthNode",
            "RouterNode",
            "MemoryRouterNode",
            "MCPToolNode",
            "EvidencePacketNode",
            "LLMGenerateNode",
            "SafetyPostNode",
            "ResponseNode",
            "TraceNode",
        ],
    },
    "RoleAIChatGraph": {
        "name": "RoleAIChatGraph",
        "role": "doctor|student",
        "task_type": "doctor_ai_chat|student_ai_chat",
        "version": "0.8.0",
        "nodes": [
            "AuthNode",
            "RouterNode",
            "MemoryRouterNode",
            "MCPToolNode",
            "EvidencePacketNode",
            "LLMGenerateNode",
            "SafetyPostNode",
            "ResponseNode",
            "TraceNode",
        ],
    },
}


def list_graphs() -> list[dict]:
    return list(GRAPH_DESCRIPTIONS.values())


def graph_detail(graph_name: str) -> dict:
    return GRAPH_DESCRIPTIONS.get(graph_name, {})


def build_graph(db: Session, graph_name: str):
    if graph_name == "DoctorReviewGraph":
        return DoctorReviewGraph(db)
    if graph_name == "PatientEducationGraph":
        return PatientEducationGraph(db)
    if graph_name == "StudentTrainingGraph":
        return StudentTrainingGraph(db)
    if graph_name == "RoleAIChatGraph":
        return RoleAIChatGraph(db)
    raise KeyError(graph_name)
