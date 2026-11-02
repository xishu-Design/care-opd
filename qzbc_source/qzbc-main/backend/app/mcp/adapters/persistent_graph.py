from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import GraphPath
from app.mcp.contracts import ToolRequest


def health(db: Session) -> dict:
    del db
    return {"backend": "persistent_graph", "status": "ok"}


def execute(db: Session, request: ToolRequest) -> dict:
    visibility = _visibility_for(request)
    existing = _latest_path(db, request, visibility)
    path = existing.path_json if existing else _build_path(request, visibility)
    if request.tool_name in {
        "create_case_graph",
        "update_case_graph",
        "add_prescription_to_graph",
        "add_risk_to_graph",
        "get_reasoning_path",
        "export_deidentified_graph",
    }:
        db.add(
            GraphPath(
                session_id=request.context.session_id,
                patient_id=request.context.patient_id if visibility == "doctor_internal" else None,
                prescription_id=request.context.prescription_id if visibility == "doctor_internal" else None,
                learning_case_id=request.context.learning_case_id,
                visibility=visibility,
                path_json=path,
                created_by_user_id=request.context.user_id,
            )
        )
        db.flush()
    return {
        "graph_reasoning_path": path,
        "__metadata__": {
            "backend": "persistent_graph",
            "degraded": not path,
            "fallback_used": False,
            "degraded_reason": "graph_path_unavailable" if not path else None,
        },
    }


def _visibility_for(request: ToolRequest) -> str:
    if request.context.role == "student" or request.tool_name == "export_deidentified_graph":
        return "student_learning"
    return request.arguments.get("visibility") or "doctor_internal"


def _latest_path(db: Session, request: ToolRequest, visibility: str) -> GraphPath | None:
    query = select(GraphPath).where(GraphPath.visibility == visibility)
    if visibility == "student_learning" and request.context.learning_case_id:
        query = query.where(GraphPath.learning_case_id == request.context.learning_case_id)
    elif request.context.prescription_id:
        query = query.where(GraphPath.prescription_id == request.context.prescription_id)
    else:
        query = query.where(GraphPath.session_id == request.context.session_id)
    return db.scalar(query.order_by(GraphPath.created_at.desc()))


def _build_path(request: ToolRequest, visibility: str) -> list[dict]:
    if request.context.task_type == "student_ai_chat":
        return [
            {
                "nodes": ["学生问题", "脱敏教学知识", "证候—治法—方药关系", "教学解析"],
                "edges": ["检索", "关联", "组织"],
                "explanation": "学生问答仅使用 student_learning 可见的脱敏知识关系。",
                "visibility": "student_learning",
            }
        ]
    if visibility == "student_learning":
        return [
            {
                "nodes": ["脱敏教学案例", "脱敏处方", "规则触发点", "教学证据"],
                "edges": ["contains", "includes", "triggers", "supported_by"],
                "explanation": "学生端仅可查看脱敏学习图谱路径。",
                "visibility": "student_learning",
            }
        ]
    if request.context.task_type == "doctor_ai_chat":
        return [
            {
                "nodes": ["医生问题", "规范中药知识", "证候—治法—方剂—药材关系", "专业辅助回答"],
                "edges": ["检索", "关联", "组织"],
                "explanation": "医生问答根据可追溯知识证据和关系路径组织专业辅助回答。",
                "visibility": "doctor_internal",
            }
        ]
    return [
        {
            "nodes": [
                f"patient:{request.context.patient_id}",
                f"prescription:{request.context.prescription_id}",
                "structured_rule_findings",
                "local_knowledge_evidence",
            ],
            "edges": ["has_prescription", "triggers", "supported_by"],
            "explanation": "医生端审方图谱路径，包含内部规则与证据关联。",
            "visibility": "doctor_internal",
        }
    ]
