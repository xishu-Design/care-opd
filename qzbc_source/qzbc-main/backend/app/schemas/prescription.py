from pydantic import BaseModel, Field, field_validator


class PrescriptionItemIn(BaseModel):
    herb_name: str
    dose: str | None = None
    unit: str = "g"
    notes: str | None = None

    @field_validator("unit", mode="before")
    @classmethod
    def default_herb_unit(cls, value):
        return str(value).strip() if value is not None and str(value).strip() else "g"


class PrescriptionCreateRequest(BaseModel):
    patient_id: int
    title: str = "辅助拟方草稿"
    symptom_context: str = ""
    usage_method: str | None = None
    doctor_notes: str | None = None
    items: list[PrescriptionItemIn] = Field(default_factory=list)


class PrescriptionUpdateRequest(BaseModel):
    symptom_context: str | None = None
    usage_method: str | None = None
    doctor_notes: str | None = None


class PrescriptionSuggestionRequest(BaseModel):
    patient_id: int
    symptoms: str = ""
    query: str = ""


class PrescriptionResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    title: str
    symptom_context: str = ""
    patient_age_at_visit: int | None = None
    patient_sex_at_visit: str | None = None
    status: str
    usage_method: str | None = None
    doctor_notes: str | None = None
    items: list[dict]


class ReviewResponse(BaseModel):
    prescription_id: int
    risk_level: str
    review_summary: str
    risk_cards: list[dict]
    evidence_packet: dict
