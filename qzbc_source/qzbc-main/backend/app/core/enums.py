from enum import StrEnum


class UserRole(StrEnum):
    DOCTOR = "doctor"
    PATIENT = "patient"
    STUDENT = "student"
    ADMIN = "admin"


class PrescriptionStatus(StrEnum):
    DRAFT = "draft"
    REVIEWING = "reviewing"
    DOCTOR_CONFIRMED = "doctor_confirmed"
    PUBLISHED_TO_PATIENT = "published_to_patient"
    DEIDENTIFIED_FOR_LEARNING = "deidentified_for_learning"
