from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.knowledge.ingestion import knowledge_health, search_knowledge
from app.knowledge.qa_knowledge_index import qa_knowledge_health, search_qa_knowledge
from app.mcp.contracts import ToolRequest


ENTRY_TYPE_BY_TOOL = {
    "search_herb": "herb",
    "search_formula": "formula",
    "search_contraindication": "contraindication",
    "search_diet_knowledge": "diet_precaution",
    "public_tcm_education": "public_education",
}


def health(db: Session) -> dict:
    return {**knowledge_health(db), "qa_knowledge": qa_knowledge_health()}


def execute(db: Session, request: ToolRequest) -> dict:
    settings = get_settings()
    query = request.arguments.get("query") or request.arguments.get("message") or ""
    entry_type = ENTRY_TYPE_BY_TOOL.get(request.tool_name)
    evidence = search_knowledge(
        db,
        query=query,
        entry_type=entry_type,
        retrieved_by=request.tool_name,
        log_request={
            "session_id": request.context.session_id,
            "role": request.context.role,
            "task_type": request.context.task_type,
        },
    )
    qa_enabled = entry_type is None and settings.qa_knowledge_enabled
    if qa_enabled:
        qa_evidence = search_qa_knowledge(query=query, role=request.context.role, limit=3)
        evidence = _merge_evidence(qa_evidence, evidence, limit=5)
    metadata = {
        "backend": "local_knowledge",
        "degraded": not evidence,
        "fallback_used": False,
        "degraded_reason": "no_evidence_found" if not evidence else None,
        "retrieval_mode": "local_structured+qa_knowledge" if qa_enabled else "local_structured",
        "qa_retrieval_count": sum(1 for item in evidence if str(item.get("retrieved_by", "")).startswith("qa_knowledge")),
    }
    if request.tool_name == "public_tcm_education":
        answer = _public_answer(evidence)
        return {"answer": answer, "evidence": evidence, "__metadata__": metadata}
    return {"evidence": evidence, "__metadata__": metadata}


def _merge_evidence(primary: list[dict], secondary: list[dict], *, limit: int) -> list[dict]:
    result = []
    seen: set[tuple[str, str]] = set()
    for item in [*primary, *secondary]:
        key = (str(item.get("source_id") or ""), str(item.get("entry_id") or item.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _public_answer(evidence: list[dict]) -> str:
    if not evidence:
        return "我目前只能提供一般性科普，具体用药请以医生说明为准。"
    first = evidence[0]
    snippet = first.get("snippet") or "中医药科普可以帮助理解常见概念。"
    return f"{snippet} 以上仅为一般科普，不能替代医生诊疗意见。"
