from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.db.models import Prescription
from app.mcp.contracts import ToolRequest
from app.rules.mock_rule_engine import run_mock_prescription_review


def execute(db: Session, request: ToolRequest) -> dict:
    prescription_id = request.arguments.get("prescription_id") or request.context.prescription_id
    prescription = db.get(Prescription, prescription_id) if prescription_id else None
    if request.tool_name in {"run_mock_prescription_review", "run_prescription_review"}:
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")
        result = run_mock_prescription_review(prescription)
        if not result["risk_cards"]:
            result["risk_cards"] = [
                {
                    "risk_type": "mock_review_completed",
                    "risk_level": "green",
                    "message": "Mock 审方未发现阻断性风险，仍需医生最终确认。",
                    "evidence": "Mock rule evidence: 第二阶段用于验证规则发现链路。",
                }
            ]
        return {
            "risk_level": result["risk_level"],
            "rule_findings": result["risk_cards"],
            "review_summary": result["review_summary"],
        }
    if request.tool_name == "check_mock_missing_information":
        missing = bool(prescription and not prescription.usage_method)
        return {"rule_findings": [{"rule_id": "MOCK_RULE_MISSING_USAGE", "triggered": missing}]}
    return {"rule_findings": [{"rule_id": "MOCK_RULE_DOSAGE", "triggered": False}]}
