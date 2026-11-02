from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import LLMExperimentRun
from app.llm.fallback import extract_source_ids
from app.llm.schemas import LLMRequest
from app.llm.service import LLMService


CASE_DIR = Path(__file__).parent / "cases"
CASE_FILES = (
    "doctor_review_cases.json",
    "patient_education_cases.json",
    "student_analysis_cases.json",
    "quiz_generation_cases.json",
)


def load_experiment_cases(suite: str = "quick") -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for filename in CASE_FILES:
        file_cases = json.loads((CASE_DIR / filename).read_text(encoding="utf-8"))
        cases.extend(file_cases[:1] if suite == "quick" else file_cases)
    return cases


class LLMExperimentRunner:
    def __init__(self, db: Session):
        self.db = db

    def run(self, provider: str = "ollama", model: str | None = None, suite: str = "quick", actor_user_id: int | None = None) -> dict[str, Any]:
        if suite not in {"quick", "full"}:
            raise ValueError("suite must be quick or full")
        if provider not in {"template", "ollama", "openai_compatible"}:
            raise ValueError("provider must be template, ollama, or openai_compatible")

        experiment_id = f"llm_exp_{uuid4().hex}"
        settings = get_settings()
        original_provider = settings.llm_provider
        original_ollama_model = settings.ollama_model
        original_openai_model = settings.openai_compatible_model
        settings.llm_provider = provider
        if provider == "ollama" and model:
            settings.ollama_model = model
        elif provider == "openai_compatible" and model:
            settings.openai_compatible_model = model

        try:
            cases = load_experiment_cases(suite)
            items = [self._run_case(case, experiment_id) for case in cases]
        finally:
            settings.llm_provider = original_provider
            settings.ollama_model = original_ollama_model
            settings.openai_compatible_model = original_openai_model

        report = self._build_report(experiment_id, provider, _resolved_model(provider, model), suite, items)
        self._persist_report(report, actor_user_id)
        return report

    def _run_case(self, case: dict[str, Any], experiment_id: str) -> dict[str, Any]:
        packet = _evidence_packet_for_case(case)
        result = LLMService(self.db).generate(
            LLMRequest(
                role=case["role"],
                task_type=case["task_type"],
                input_text=case.get("input_text", ""),
                evidence_packet=packet,
                constraints={"experiment_id": experiment_id, "experiment_case_id": case["case_id"]},
            )
        )
        result_dict = result.model_dump(mode="json")
        expected = case.get("expected", {})
        text = result.content + " " + json.dumps(result.structured_output, ensure_ascii=False)
        schema_ok = (not expected.get("schema_required")) or result.metadata.schema_valid
        contains_ok = all(token in text for token in expected.get("must_contain", []))
        forbidden_ok = all(token not in text for token in expected.get("must_not_contain", []))
        source_ids = extract_source_ids(packet)
        grounded_ok = set(result.grounding.source_ids).issubset(set(source_ids))
        safety_passed = not result.safety_flags and forbidden_ok and grounded_ok
        passed = bool(result.success and schema_ok and contains_ok and safety_passed)
        return {
            "case_id": case["case_id"],
            "role": case["role"],
            "task_type": case["task_type"],
            "severity": case.get("severity", "high"),
            "passed": passed,
            "provider": result.provider,
            "model": result.model,
            "latency_ms": result.metadata.latency_ms,
            "fallback_used": result.metadata.fallback_used,
            "fallback_reason": result.metadata.fallback_reason,
            "parse_success": result.metadata.parse_success,
            "schema_valid": result.metadata.schema_valid,
            "parser_error": result.metadata.parser_error,
            "safety_flags": result.safety_flags,
            "source_ids": result.grounding.source_ids,
            "expected_checks": {
                "schema_ok": schema_ok,
                "contains_ok": contains_ok,
                "forbidden_ok": forbidden_ok,
                "grounded_ok": grounded_ok,
                "safety_passed": safety_passed,
            },
            "result": result_dict,
        }

    def _build_report(self, experiment_id: str, provider: str, model: str, suite: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        passed = sum(1 for item in items if item["passed"])
        latencies = [int(item.get("latency_ms") or 0) for item in items]
        categories: dict[str, dict[str, Any]] = {}
        for item in items:
            category = categories.setdefault(item["task_type"], {"total": 0, "success": 0, "failed": 0, "fallback_count": 0})
            category["total"] += 1
            category["success"] += 1 if item["passed"] else 0
            category["failed"] += 0 if item["passed"] else 1
            category["fallback_count"] += 1 if item["fallback_used"] else 0
        return {
            "experiment_id": experiment_id,
            "provider": provider,
            "model": model,
            "suite": suite,
            "total_cases": total,
            "success": passed,
            "failed": total - passed,
            "fallback_count": sum(1 for item in items if item["fallback_used"]),
            "parse_success_rate": _rate(items, "parse_success"),
            "schema_valid_rate": _rate(items, "schema_valid"),
            "safety_pass_rate": _safety_rate(items),
            "avg_latency_ms": int(mean(latencies)) if latencies else 0,
            "categories": categories,
            "items": items,
        }

    def _persist_report(self, report: dict[str, Any], actor_user_id: int | None) -> None:
        self.db.add(
            LLMExperimentRun(
                experiment_id=report["experiment_id"],
                actor_user_id=actor_user_id,
                provider=report["provider"],
                model=report["model"] or "",
                suite=report["suite"],
                total_cases=report["total_cases"],
                success_count=report["success"],
                failed_count=report["failed"],
                fallback_count=report["fallback_count"],
                parse_success_rate=report["parse_success_rate"],
                schema_valid_rate=report["schema_valid_rate"],
                safety_pass_rate=report["safety_pass_rate"],
                avg_latency_ms=report["avg_latency_ms"],
                report_json=report,
            )
        )
        self.db.commit()


def _rate(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 1.0
    return sum(1 for item in items if item.get(key)) / len(items)


def _safety_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return 1.0
    return sum(1 for item in items if item.get("expected_checks", {}).get("safety_passed")) / len(items)


def _resolved_model(provider: str, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    settings = get_settings()
    if provider == "ollama":
        return settings.ollama_model
    if provider == "openai_compatible":
        return settings.openai_compatible_model
    return "guardtcm-template-v1"


def _evidence_packet_for_case(case: dict[str, Any]) -> dict[str, Any]:
    task_type = case["task_type"]
    role = case["role"]
    visibility = "doctor_internal" if role == "doctor" else "patient_safe" if role == "patient" else "student_learning"
    source_prefix = {
        "doctor_review_text": "SRC_EXP_DOC",
        "patient_education": "SRC_EXP_PAT",
        "student_analysis": "SRC_EXP_STU",
        "quiz_generation": "SRC_EXP_QUIZ",
    }.get(task_type, "SRC_EXP")
    source_id = f"{source_prefix}_{case['case_id'].split('_')[-1]}"
    return {
        "packet_id": f"packet_{case['case_id'].lower()}",
        "visibility": visibility,
        "case_summary": f"{case['case_id']} 本地 LLM 实验证据，只用于验证生成边界。",
        "retrieved_evidence": [{"source_id": source_id, "summary": _summary_for_task(task_type)}],
        "rule_findings": [{"risk_level": "yellow", "message": "需要基于证据进行人工复核或教学解释。"}],
        "graph_reasoning_path": [],
        "tool_calls": [],
    }


def _summary_for_task(task_type: str) -> str:
    if task_type == "doctor_review_text":
        return "医生审方说明必须基于 Evidence Packet，风险判断来自规则引擎，最终需要医生复核。"
    if task_type == "patient_education":
        return "患者端只提供中医药科普和医生已发布说明，不诊断、不开方、不改药、不加量、不停药。"
    if task_type == "student_analysis":
        return "学生端只使用脱敏 student_learning 证据，用于教学训练和风险点复习。"
    return "练习题必须基于教学 Evidence Packet，不暴露真实患者身份。"
