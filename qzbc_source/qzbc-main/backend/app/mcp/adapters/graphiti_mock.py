from sqlalchemy.orm import Session

from app.db.models import MockGraphPath
from app.mcp.contracts import ToolRequest


def _path(request: ToolRequest, visibility: str) -> list[dict]:
    if visibility == "student_learning":
        nodes = ["脱敏病例", "脱敏处方", "风险点", "教学证据"]
    else:
        nodes = ["病例A", "处方A", "药材A", "风险规则A", "证据A"]
    return [
        {
            "nodes": nodes,
            "edges": ["contains", "includes", "triggers", "supported_by"],
            "explanation": "Mock Graphiti path for prescription review.",
            "visibility": visibility,
        }
    ]


def execute(db: Session, request: ToolRequest) -> dict:
    visibility = "student_learning" if request.tool_name in {"mock_export_deidentified_graph", "export_deidentified_graph"} or request.context.role == "student" else "doctor_internal"
    path = _path(request, visibility)
    db.add(
        MockGraphPath(
            session_id=request.context.session_id,
            patient_id=request.context.patient_id,
            prescription_id=request.context.prescription_id,
            learning_case_id=request.context.learning_case_id,
            path_json=path,
            visibility=visibility,
        )
    )
    return {"graph_reasoning_path": path}
