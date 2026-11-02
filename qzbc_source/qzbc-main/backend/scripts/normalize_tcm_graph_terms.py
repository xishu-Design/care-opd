import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal
from app.knowledge.internal_knowledge import normalize_shennong_claims


def main() -> None:
    db = SessionLocal()
    try:
        report = normalize_shennong_claims(db)
        db.commit()
        print(report)
    finally:
        db.close()


if __name__ == "__main__":
    main()
