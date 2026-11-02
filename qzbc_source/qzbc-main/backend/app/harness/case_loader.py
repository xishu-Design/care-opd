from __future__ import annotations

import json
from pathlib import Path

from app.harness.case_schema import HarnessCase, PHASE4

CASES_DIR = Path(__file__).resolve().parent / "cases" / "phase4"
CASE_FILES = (
    "auth_permission_cases.json",
    "mcp_permission_cases.json",
    "safety_cases.json",
    "evidence_cases.json",
    "deidentification_cases.json",
    "memory_isolation_cases.json",
    "graph_consistency_cases.json",
    "fallback_cases.json",
    "demo_scenario_cases.json",
)


def load_phase4_cases(suite: str = "full") -> list[HarnessCase]:
    cases: list[HarnessCase] = []
    for file_name in CASE_FILES:
        path = CASES_DIR / file_name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            raw_cases = json.load(file)
        cases.extend(HarnessCase.model_validate(raw_case) for raw_case in raw_cases)

    cases = [case for case in cases if case.phase == PHASE4]
    if suite == "quick":
        quick_categories = {"role_permission", "mcp_permission", "safety", "evidence"}
        return [case for case in cases if case.category in quick_categories][:8]
    if suite == "demo":
        return [case for case in cases if case.category == "demo_scenario"]
    if suite == "regression":
        return [case for case in cases if "regression" in case.tags or case.severity in {"critical", "high"}]
    return cases
