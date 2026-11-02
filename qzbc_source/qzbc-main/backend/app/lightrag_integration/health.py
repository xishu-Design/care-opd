from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.lightrag_integration.client import LightRAGClient
from app.lightrag_integration.config import DEFAULT_REPORTS_DIR, get_lightrag_config
from app.knowledge.qa_knowledge_index import qa_knowledge_health


def lightrag_health() -> dict[str, Any]:
    config = get_lightrag_config()
    health = LightRAGClient(config).health()
    report = _read_report(DEFAULT_REPORTS_DIR / "lightrag_index_report.json")
    health.update(
        {
            "working_dir": str(config.working_dir),
            "input_dir": str(config.input_dir),
            "last_index_at": report.get("ended_at", ""),
            "index_status": report.get("status", "not_run"),
            "docs_indexed": report.get("docs_indexed", 0),
            "qa_knowledge": qa_knowledge_health(),
        }
    )
    return health


def _read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}
