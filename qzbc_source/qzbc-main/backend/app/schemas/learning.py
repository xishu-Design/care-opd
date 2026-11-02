from pydantic import BaseModel


class StudentAnswerRequest(BaseModel):
    answer_text: str
    proposed_herbs: list[str] = []
    has_risk: bool = False
    risk_level: str | None = None


class LearningCaseResponse(BaseModel):
    id: int
    title: str
    anonymized_case: dict
    standard_analysis: str | None = None
    risk_points: list
    evidence_chain: list
