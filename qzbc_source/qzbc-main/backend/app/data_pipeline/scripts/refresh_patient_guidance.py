from __future__ import annotations

import argparse
import json

from sqlalchemy import select

from app.core.enums import PrescriptionStatus
from app.db.models import Prescription
from app.db.session import SessionLocal
from app.services.patient_guidance_service import upsert_patient_guidance


def refresh(*, prescription_id: int | None = None) -> dict:
    db = SessionLocal()
    try:
        query = select(Prescription).where(
            Prescription.status.in_(
                [
                    PrescriptionStatus.PUBLISHED_TO_PATIENT.value,
                    PrescriptionStatus.DEIDENTIFIED_FOR_LEARNING.value,
                ]
            )
        )
        if prescription_id is not None:
            query = query.where(Prescription.id == prescription_id)
        prescriptions = db.scalars(query.order_by(Prescription.id)).all()
        results = []
        for prescription in prescriptions:
            details = upsert_patient_guidance(db, prescription)
            results.append(
                {
                    "prescription_id": prescription.id,
                    "provider": details.get("llm", {}).get("provider"),
                    "model": details.get("llm", {}).get("model"),
                    "food_interactions": len(details.get("food_interactions", [])),
                    "herb_interactions": len(details.get("herb_interactions", [])),
                }
            )
        db.commit()
        return {"refreshed": len(results), "items": results}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate patient-visible instructions from published prescriptions.")
    parser.add_argument("--prescription-id", type=int)
    args = parser.parse_args()
    print(json.dumps(refresh(prescription_id=args.prescription_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
