from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.graphs.patient_education_graph import PatientEducationGraph
from app.core.config import get_settings
from app.db.models import AgentTraceEvent, HarnessResult, HarnessRun, User, utcnow
from app.evidence.evidence_packet import build_evidence_packet_v2
from app.harness.case_schema import PHASE10, RunnerResult
from app.harness.phase5_runner import _category_summary
from app.lightrag_integration.client import LightRAGClient
from app.lightrag_integration.document_builder import build_lightrag_docs, load_lightrag_docs
from app.lightrag_integration.index_builder import build_lightrag_index
from app.mcp.contracts import ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway


class Phase10HarnessRunner:
    def __init__(self, db: Session, suite: str = "full", reset_demo_data: bool = True):
        self.db = db
        self.suite = suite
        self.reset_demo_data = reset_demo_data

    def run(self) -> HarnessRun:
        started = perf_counter()
        started_at = utcnow()
        original = _snapshot_settings()
        with TemporaryDirectory(prefix="guardtcm_phase10_") as tmp:
            self.tmp_dir = Path(tmp)
            self.content_dir = self.tmp_dir / "content" / "data"
            self.input_dir = self.tmp_dir / "input_docs"
            self.working_dir = self.tmp_dir / "working_dir"
            self.reports_dir = self.tmp_dir / "reports"
            _write_content_fixture(self.content_dir)
            _configure_lightrag(self.input_dir, self.working_dir)
            try:
                results = [
                    self._timed("P10_DOC_001", "lightrag_doc_build", "content/data can generate traceable LightRAG docs", "critical", self._doc_build),
                    self._timed("P10_INDEX_001", "lightrag_index", "index builder creates working dir and index report", "critical", self._index_build),
                    self._timed("P10_RETRIEVAL_001", "lightrag_retrieval", "LightRAG client returns source-bound GraphRAG results", "critical", self._retrieval),
                    self._timed("P10_VISIBILITY_001", "lightrag_visibility_filter", "patient role cannot see doctor/student-only evidence", "critical", self._visibility_filter),
                    self._timed("P10_FALLBACK_001", "lightrag_fallback", "unavailable LightRAG falls back to LocalKnowledgeAdapter", "critical", self._fallback),
                    self._timed("P10_PACKET_001", "evidence_packet_graph_retrieval", "Evidence Packet records LightRAG graph retrieval metadata", "critical", self._evidence_packet),
                    self._timed("P10_MCP_001", "mcp_lightrag_adapter", "MCP Gateway routes KnowledgeServer tools to LightRAGAdapter", "critical", self._mcp_adapter),
                    self._timed("P10_DOCTOR_001", "doctor_lightrag_review", "doctor prescription-review task can retrieve LightRAG evidence", "high", self._doctor_path),
                    self._timed("P10_PATIENT_001", "patient_lightrag_safety", "patient high-risk request stops before retrieval", "critical", self._patient_safety),
                    self._timed("P10_STUDENT_001", "student_lightrag_learning", "student task retrieves student-visible LightRAG evidence", "high", self._student_path),
                ]
                if self.suite == "quick":
                    results = results[:6]
            finally:
                _apply_settings(original)

        total = len(results)
        passed = sum(1 for result in results if result.passed)
        metrics = _metrics(results)
        run = HarnessRun(
            phase=PHASE10,
            suite=self.suite,
            status="completed",
            total_cases=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total else 0.0,
            categories=_category_summary(results),
            metrics_json=metrics,
            markdown_summary=_markdown_summary(total, passed, total - passed, metrics),
            started_at=started_at,
            ended_at=utcnow(),
            duration_ms=int((perf_counter() - started) * 1000),
        )
        self.db.add(run)
        self.db.flush()
        for result in results:
            self.db.add(
                HarnessResult(
                    run_id=run.id,
                    case_id=result.case_id,
                    category=result.category,
                    name=result.details,
                    severity=result.severity,
                    passed=result.passed,
                    actual_json=result.actual,
                    expected_json=result.expected,
                    details={"message": result.details},
                    trace_id=result.trace_id,
                    evidence_packet_id=result.related_evidence_packet_id,
                    failure_reason=result.failure_reason,
                    duration_ms=result.duration_ms,
                )
            )
        self.db.commit()
        self.db.refresh(run)
        return run

    def _timed(self, case_id: str, category: str, name: str, severity: str, handler: Callable[[str, str, str, str], RunnerResult]) -> RunnerResult:
        started = perf_counter()
        try:
            result = handler(case_id, category, name, severity)
        except Exception as exc:
            result = self._result(case_id, category, name, severity, False, {"error": str(exc)}, str(exc))
        result.duration_ms = int((perf_counter() - started) * 1000)
        return result

    def _result(self, case_id: str, category: str, name: str, severity: str, passed: bool, actual: dict[str, Any], failure_reason: str | None = None) -> RunnerResult:
        return RunnerResult(case_id=case_id, category=category, passed=passed, severity=severity, actual=actual, expected={"name": name}, details=name, failure_reason=None if passed else failure_reason or "assertion_failed")

    def _doc_build(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        report = self._build_docs()
        docs = load_lightrag_docs(self.input_dir / "guardtcm_lightrag_docs.jsonl")
        passed = report["docs_written"] >= 3 and all(doc.metadata.get("source_id") and doc.entry_id and doc.entry_type and doc.metadata.get("license") for doc in docs)
        return self._result(case_id, category, name, severity, passed, {"report": report, "docs": [doc.model_dump(mode="json") for doc in docs]})

    def _index_build(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._build_docs()
        report = build_lightrag_index(docs_path=self.input_dir / "guardtcm_lightrag_docs.jsonl", working_dir=self.working_dir, report_path=self.reports_dir / "lightrag_index_report.json", rebuild=True)
        passed = report["status"] == "success" and report["docs_indexed"] >= 3 and (self.working_dir / "guardtcm_lightrag_index.jsonl").exists()
        return self._result(case_id, category, name, severity, passed, {"report": report})

    def _retrieval(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_index()
        response = LightRAGClient().query("黄芪", mode="hybrid")
        first = response.results[0] if response.results else None
        passed = response.success and first is not None and first.source_id == "SRC_PHASE10_OPEN_DATA" and bool(first.graph_entities)
        return self._result(case_id, category, name, severity, passed, response.model_dump(mode="json"))

    def _visibility_filter(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_index()
        patient = self._user("phase10_harness_patient", "patient")
        result = self._call(patient, "search_herb", {"query": "黄芪"}, "patient_education")
        passed = result.success and result.evidence == [] and result.metadata.get("backend") == "lightrag" and result.metadata.get("filtered_count", 0) > 0
        return self._result(case_id, category, name, severity, passed, result.model_dump(mode="json"))

    def _fallback(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        settings = get_settings()
        settings.lightrag_enabled = False
        doctor = self._user("phase10_harness_fallback_doctor", "doctor")
        result = self._call(doctor, "search_herb", {"query": "黄芪"}, "prescription_review")
        settings.lightrag_enabled = True
        passed = result.success and result.metadata.get("backend") == "local_knowledge" and result.metadata.get("fallback_used") is True
        return self._result(case_id, category, name, severity, passed, result.model_dump(mode="json"))

    def _evidence_packet(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_index()
        doctor = self._user("phase10_harness_packet_doctor", "doctor")
        result = self._call(doctor, "search_herb", {"query": "黄芪"}, "prescription_review")
        packet = build_evidence_packet_v2(self.db, role="doctor", task_type="prescription_review", visibility="doctor_internal", case_summary="phase10 harness", tool_results=[result], created_by_user_id=doctor.id)
        passed = packet["backend_summary"].get("knowledge_backend") == "lightrag" and bool(packet["retrieved_evidence"][0].get("graph_entities"))
        return self._result(case_id, category, name, severity, passed, {"packet": packet})

    def _mcp_adapter(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_index()
        doctor = self._user("phase10_harness_mcp_doctor", "doctor")
        result = self._call(doctor, "search_formula", {"query": "黄芪"}, "prescription_review")
        passed = result.success and result.metadata.get("backend") == "lightrag" and result.evidence and result.evidence[0].get("entry_type") == "formula"
        return self._result(case_id, category, name, severity, passed, result.model_dump(mode="json"))

    def _doctor_path(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_index()
        doctor = self._user("phase10_harness_doctor", "doctor")
        result = self._call(doctor, "search_evidence", {"query": "黄芪"}, "prescription_review")
        passed = result.success and result.metadata.get("backend") == "lightrag" and bool(result.evidence)
        return self._result(case_id, category, name, severity, passed, result.model_dump(mode="json"))

    def _patient_safety(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        patient = self._user("phase10_harness_safe_patient", "patient")
        response = PatientEducationGraph(self.db).invoke(patient, "我能不能把医生开的药加量？")
        events = self.db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.trace_id == response["trace_id"])).all()
        nodes = {event.node_name for event in events}
        passed = response.get("allowed") is False and "MCPToolNode" not in nodes
        return self._result(case_id, category, name, severity, passed, {"response": response, "nodes": sorted(nodes)})

    def _student_path(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_index()
        student = self._user("phase10_harness_student", "student")
        result = self._call(student, "search_evidence", {"query": "黄芪"}, "student_training")
        evidence_text = json.dumps(result.evidence, ensure_ascii=False)
        passed = (
            result.success
            and result.metadata.get("backend") == "lightrag"
            and result.evidence
            and result.evidence[0].get("metadata", {}).get("visibility") == ["doctor", "student"]
            and "patient_name" not in evidence_text
            and "手机号" not in evidence_text
            and "身份证" not in evidence_text
        )
        return self._result(case_id, category, name, severity, passed, result.model_dump(mode="json"))

    def _build_docs(self) -> dict[str, Any]:
        return build_lightrag_docs(content_data_dir=self.content_dir, output_path=self.input_dir / "guardtcm_lightrag_docs.jsonl", report_path=self.reports_dir / "lightrag_doc_build_report.json")

    def _ensure_index(self) -> None:
        if not (self.input_dir / "guardtcm_lightrag_docs.jsonl").exists():
            self._build_docs()
        if not (self.working_dir / "guardtcm_lightrag_index.jsonl").exists():
            build_lightrag_index(docs_path=self.input_dir / "guardtcm_lightrag_docs.jsonl", working_dir=self.working_dir, report_path=self.reports_dir / "lightrag_index_report.json", rebuild=True)

    def _user(self, username: str, role: str) -> User:
        user = self.db.scalar(select(User).where(User.username == username))
        if user:
            return user
        user = User(username=username, hashed_password="harness", role=role, display_name=username)
        self.db.add(user)
        self.db.flush()
        return user

    def _call(self, user: User, tool_name: str, arguments: dict[str, Any], task_type: str):
        return MCPGateway(self.db).execute(ToolRequest(tool_name=tool_name, arguments=arguments, context=ToolContext(user_id=user.id, role=user.role, task_type=task_type)))


def _write_content_fixture(content_dir: Path) -> None:
    source = {
        "source_id": "SRC_PHASE10_OPEN_DATA",
        "source_title": "Phase10 Open Data Fixture",
        "source_type": "open_source_dataset",
        "source_version": "phase10abc",
        "source_year": "2026",
        "evidence_level": "open_dataset_extracted",
        "license": "MIT",
        "license_note": "Harness fixture derived from Phase 9 style content.",
        "processing_script": "phase10_harness_fixture.py",
        "processing_version": "1.0.0",
    }
    _write_json(content_dir / "sources.json", {"items": [source]})
    _write_json(content_dir / "source_catalog.json", {"sources": [{"source_id": source["source_id"], "name": source["source_title"], "license": "MIT", "commit_hash": "phase10abc", "processing_script": "phase10_harness_fixture.py", "processing_version": "1.0.0"}]})
    _write_json(content_dir / "herbs.json", {"items": [{"entry_id": "HERB_PHASE10_001", "name": "黄芪", "source_ids": [source["source_id"]], "source_records": [{"source_id": source["source_id"], "source_record_id": "row_1", "source_file": "fixture.csv"}], "functions": [], "cautions": [], "dose_range": None, "keywords": ["黄芪"], "visibility": ["doctor", "student"], "evidence_level": "open_dataset_extracted"}]})
    _write_json(content_dir / "formulas.json", {"items": [{"entry_id": "FORMULA_PHASE10_001", "name": "source_formula_phase10", "components": [{"herb_name": "黄芪", "normalized_name": "黄芪", "dosage": None, "unit": None}], "source_ids": [source["source_id"]], "source_records": [{"source_id": source["source_id"], "source_record_id": "row_2", "source_file": "fixture.csv"}], "functions": [], "cautions": [], "keywords": ["黄芪", "处方组成"], "visibility": ["doctor", "student"], "evidence_level": "open_dataset_extracted"}]})
    _write_json(content_dir / "public_education.json", {"items": [{"entry_id": "EDU_PHASE10_001", "topic": "中医药科普边界", "safe_answer": "中医药科普不能替代医生诊疗意见。", "source_id": source["source_id"], "keywords": ["科普", "边界"], "visibility": ["patient"]}]})
    for file_name in ("contraindications.json", "diet_precautions.json", "review_rules.json", "learning_cases.json", "quizzes.json"):
        _write_json(content_dir / file_name, {"items": []})


def _configure_lightrag(input_dir: Path, working_dir: Path) -> None:
    settings = get_settings()
    settings.knowledge_backend = "lightrag"
    settings.lightrag_enabled = True
    settings.lightrag_mode = "embedded"
    settings.lightrag_input_dir = str(input_dir)
    settings.lightrag_working_dir = str(working_dir)
    settings.lightrag_allow_fallback = True


def _snapshot_settings() -> dict[str, Any]:
    settings = get_settings()
    return {
        "knowledge_backend": settings.knowledge_backend,
        "lightrag_enabled": settings.lightrag_enabled,
        "lightrag_mode": settings.lightrag_mode,
        "lightrag_input_dir": settings.lightrag_input_dir,
        "lightrag_working_dir": settings.lightrag_working_dir,
        "lightrag_allow_fallback": settings.lightrag_allow_fallback,
    }


def _apply_settings(values: dict[str, Any]) -> None:
    settings = get_settings()
    for key, value in values.items():
        setattr(settings, key, value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rate(results: list[RunnerResult], category: str) -> float:
    rows = [result for result in results if result.category == category]
    if not rows:
        return 1.0
    return sum(1 for result in rows if result.passed) / len(rows)


def _metrics(results: list[RunnerResult]) -> dict[str, Any]:
    return {
        "lightrag_doc_build_rate": _rate(results, "lightrag_doc_build"),
        "lightrag_index_rate": _rate(results, "lightrag_index"),
        "lightrag_retrieval_rate": _rate(results, "lightrag_retrieval"),
        "lightrag_visibility_filter_rate": _rate(results, "lightrag_visibility_filter"),
        "lightrag_fallback_rate": _rate(results, "lightrag_fallback"),
        "evidence_packet_graph_retrieval_rate": _rate(results, "evidence_packet_graph_retrieval"),
        "mcp_lightrag_adapter_rate": _rate(results, "mcp_lightrag_adapter"),
        "patient_lightrag_safety_rate": _rate(results, "patient_lightrag_safety"),
        "student_lightrag_learning_rate": _rate(results, "student_lightrag_learning"),
    }


def _markdown_summary(total: int, passed: int, failed: int, metrics: dict[str, Any]) -> str:
    pass_rate = passed / total if total else 0.0
    return (
        "# GuardTCM Phase 10 Harness Report\n\n"
        f"本次 Harness 共运行 {total} 个 LightRAG/GraphRAG 接入用例，通过 {passed} 个，失败 {failed} 个，通过率 {pass_rate:.0%}。\n\n"
        "核心结论：LightRAG 只负责图谱增强检索，MCP Gateway、Evidence Packet、Safety Guard 和 LocalKnowledge fallback 仍保持边界。\n\n"
        f"- Doc build：{metrics['lightrag_doc_build_rate']:.0%}。\n"
        f"- Index：{metrics['lightrag_index_rate']:.0%}。\n"
        f"- Retrieval：{metrics['lightrag_retrieval_rate']:.0%}。\n"
        f"- Visibility：{metrics['lightrag_visibility_filter_rate']:.0%}。\n"
        f"- Fallback：{metrics['lightrag_fallback_rate']:.0%}。\n"
    )
