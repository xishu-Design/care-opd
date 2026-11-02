from app.db.models import Prescription


def run_mock_prescription_review(prescription: Prescription) -> dict:
    has_usage = bool(prescription.usage_method and prescription.usage_method.strip())
    risk_cards = []
    if not has_usage:
        complete_dose_count = sum(bool(item.dose and item.unit) for item in prescription.items)
        if prescription.items and complete_dose_count == len(prescription.items):
            message = "各味药的剂量和单位已经填写；当前缺少的是处方级用法说明（如煎服方式、每日频次、分服方法）。"
        else:
            message = "当前处方仍有剂量信息不完整，且缺少处方级用法说明。"
        risk_cards.append(
            {
                "risk_type": "missing_information",
                "risk_level": "yellow",
                "message": message,
                "evidence": "Mock evidence: 用法用量信息是患者安全用药说明的重要组成部分。",
            }
        )
    if not prescription.items:
        risk_cards.append(
            {
                "risk_type": "empty_prescription",
                "risk_level": "yellow",
                "message": "当前处方未录入药材组成，无法完成结构化审方。",
                "evidence": "Mock evidence: 审方需要至少包含药材组成。",
            }
        )
    return {
        "risk_level": "medium" if risk_cards else "low",
        "risk_cards": risk_cards,
        "review_summary": "Mock 审方已完成，请医生结合临床信息确认后再发布。",
    }
