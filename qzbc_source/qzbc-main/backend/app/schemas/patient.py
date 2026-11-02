from pydantic import BaseModel, Field


class PatientCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str | None = Field(default=None, min_length=6, max_length=128)
    display_name: str
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    allergy_history: str = ""
    medical_history: str = ""
    body_state: str = ""
    current_symptoms: str = ""


class PatientProfileResponse(BaseModel):
    id: int
    username: str
    display_name: str
    age: int | None = None
    sex: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    allergy_history: str = ""
    medical_history: str = ""
    body_state: str = ""
    current_symptoms: str = ""


class PatientUpdateRequest(BaseModel):
    display_name: str | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    allergy_history: str | None = None
    medical_history: str | None = None
    body_state: str | None = None
    current_symptoms: str | None = None
