from sqlalchemy.orm import Session

from app.db.models import LightRAGRetrievalLog
from app.mcp.contracts import ToolRequest


def _evidence(query: str, title: str = "Mock 中药知识库") -> list[dict]:
    return [
        {
            "source_id": "SRC_MOCK_FALLBACK_001",
            "entry_id": "MOCK_EVIDENCE_001",
            "source_type": "mock_tcm_knowledge",
            "title": title,
            "snippet": f"{query or '该问题'} 的内容为第二阶段模拟证据，用于验证 Evidence Packet 链路。",
            "confidence": 0.8,
            "metadata": {"demo_only": True},
            "retrieved_by": "mock_search_evidence",
        }
    ]


def execute(db: Session, request: ToolRequest) -> dict:
    query = request.arguments.get("query") or request.arguments.get("message") or ""
    evidence = _evidence(query)
    db.add(
        LightRAGRetrievalLog(
            session_id=request.context.session_id,
            query=query,
            role=request.context.role,
            task_type=request.context.task_type,
            evidence_json=evidence,
        )
    )
    for item in evidence:
        item["retrieved_by"] = request.tool_name
    if request.tool_name in {"mock_public_tcm_education", "public_tcm_education"}:
        return {
            "answer": "这是一般中医药科普信息，不能替代医生诊疗意见。如涉及用药调整，请咨询医生。",
            "evidence": evidence,
        }
    return {"evidence": evidence}
