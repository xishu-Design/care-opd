from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import Prescription
from app.mcp.contracts import ToolRequest
from app.rules.rule_engine import run_structured_prescription_review
from app.rules.rule_loader import get_rules


def health(db: Session) -> dict:
    return {"backend": "local_rules", "status": "ok", "rules_count": len(get_rules(db))}


def execute(db: Session, request: ToolRequest) -> dict:
    prescription_id = request.arguments.get("prescription_id") or request.context.prescription_id
    prescription = db.get(Prescription, prescription_id) if prescription_id else None
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")
    result = run_structured_prescription_review(db, prescription, request.context.session_id)
    result["__metadata__"] = {
        "backend": "local_rules",
        "degraded": False,
        "fallback_used": False,
    }
    return result
