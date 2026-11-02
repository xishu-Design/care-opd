import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.init_db import init_db
from app.db.session import SessionLocal
from app.services.seed_service import seed_demo_data


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        result = seed_demo_data(db)
        print(result)
    finally:
        db.close()


if __name__ == "__main__":
    main()
