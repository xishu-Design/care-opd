from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal
from app.knowledge.qa_knowledge_index import build_qa_knowledge_index


def main() -> None:
    with SessionLocal() as db:
        report = build_qa_knowledge_index(db)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
