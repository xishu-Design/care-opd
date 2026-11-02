from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.schemas import EvaluationCase


DATASET_DIR = Path(__file__).resolve().parent / "datasets"
DATASET_FILES = (
    "patient_safety_cases.json",
    "doctor_review_cases.json",
    "student_deid_cases.json",
    "mcp_permission_cases.json",
    "evidence_packet_cases.json",
    "lightrag_retrieval_cases.json",
    "mem0_isolation_cases.json",
    "human_confirm_cases.json",
    "llm_safety_cases.json",
)


def load_dataset(file_name: str) -> list[EvaluationCase]:
    if file_name not in DATASET_FILES:
        raise ValueError(f"unknown evaluation dataset: {file_name}")
    payload = json.loads((DATASET_DIR / file_name).read_text(encoding="utf-8"))
    dataset = str(payload.get("dataset") or Path(file_name).stem)
    return [EvaluationCase.model_validate({**item, "dataset": item.get("dataset", dataset)}) for item in payload.get("cases", [])]


def load_all_datasets(suite: str = "full") -> dict[str, list[EvaluationCase]]:
    datasets = {file_name: load_dataset(file_name) for file_name in DATASET_FILES}
    if suite == "quick":
        return {name: cases[: min(3, len(cases))] for name, cases in datasets.items()}
    return datasets


def dataset_manifest() -> dict:
    datasets = load_all_datasets("full")
    return {
        "version": "phase12-v1",
        "total_cases": sum(len(cases) for cases in datasets.values()),
        "datasets": {name: len(cases) for name, cases in datasets.items()},
    }
