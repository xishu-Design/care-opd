from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.content.services.content_import_service import ContentImportService
from app.content.services.content_stats_service import content_stats
from app.content.validators.content_validator import ContentValidator
from app.data_pipeline.scripts.build_guardtcm_kb import build_guardtcm_kb
from app.data_pipeline.scripts.download_sources import build_download_report
from app.data_pipeline.validators.processed_item_validator import ProcessedItemValidator
from app.data_pipeline.validators.source_registry_validator import SourceRegistryValidator
from app.data_pipeline.validators.traceability_validator import validate_source_catalog_traceability
from app.db.models import HarnessResult, HarnessRun, LearningCase, Quiz, RuleDefinition, User, utcnow
from app.harness.case_schema import PHASE9, RunnerResult
from app.harness.phase5_runner import _category_summary
from app.mcp.contracts import ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway
from app.rules.rule_loader import get_rules, load_rules


class Phase9HarnessRunner:
    def __init__(self, db: Session, suite: str = "full", reset_demo_data: bool = True):
        self.db = db
        self.suite = suite
        self.reset_demo_data = reset_demo_data

    def run(self) -> HarnessRun:
        started = perf_counter()
        started_at = utcnow()
        with TemporaryDirectory(prefix="guardtcm_phase9_") as tmp_dir:
            self.tmp_dir = Path(tmp_dir)
            self.processed_dir = self.tmp_dir / "processed"
            self.content_dir = self.tmp_dir / "content" / "data"
            self.reports_dir = self.tmp_dir / "reports"
            _write_processed_fixture(self.processed_dir)
            results = [
                self._timed("P9_REGISTRY_001", "source_registry_validation", "Approved registry sources are explicit and downloadable only when approved", "critical", self._registry_validation),
                self._timed("P9_LICENSE_001", "license_validation", "Approved source licenses are accepted before processing", "critical", self._license_validation),
                self._timed("P9_DOWNLOAD_001", "download_pipeline", "Download report records version metadata and excludes pending/rejected sources", "critical", self._download_pipeline),
                self._timed("P9_PROCESSING_001", "data_processing", "Processed candidates preserve traceability and safety boundaries", "critical", self._processed_traceability),
                self._timed("P9_TRACE_001", "source_traceability", "Processed source catalog and candidates are mutually traceable", "critical", self._source_traceability),
                self._timed("P9_CONTENT_001", "content_build", "Content data builds from processed candidates without inferred medical fields", "critical", self._content_build),
                self._timed("P9_SCHEMA_001", "content_schema_validation", "ContentValidator validates generated content/data", "critical", self._content_schema_validation),
                self._timed("P9_SAFETY_001", "content_safety_validation", "Generated content keeps patient unsafe terms and open-data patient visibility out", "critical", self._content_safety_validation),
                self._timed("P9_IMPORT_001", "content_import", "Content import is idempotent and source covered", "critical", self._content_import),
                self._timed("P9_MCP_001", "mcp_search_open_data", "MCP search retrieves imported open data with original source metadata", "critical", self._mcp_search),
                self._timed("P9_RULE_001", "rule_source_binding", "Medical review rules require source binding before activation", "critical", self._rule_source_binding),
                self._timed("P9_PATIENT_001", "patient_safety_filter", "Patient role cannot retrieve doctor/student-only open data", "critical", self._patient_filter),
                self._timed("P9_LEARNING_001", "learning_case_deidentification", "Learning cases are deidentified and quizzes use extracted herb options", "critical", self._learning_deid),
                self._timed("P9_QUIZ_001", "quiz_quality", "Generated quizzes bind to cases and use extracted herb options", "high", self._quiz_quality),
                self._timed("P9_REPRO_001", "open_data_reproducibility", "Content build is stable across repeated runs", "high", self._reproducibility),
            ]
            if self.suite == "quick":
                results = results[:6]

        total = len(results)
        passed = sum(1 for result in results if result.passed)
        metrics = _phase9_metrics(results)
        run = HarnessRun(
            phase=PHASE9,
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
                    details={"message": result.details, "related_tool_calls": result.related_tool_calls},
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

    def _result(
        self,
        case_id: str,
        category: str,
        name: str,
        severity: str,
        passed: bool,
        actual: dict[str, Any],
        failure_reason: str | None = None,
    ) -> RunnerResult:
        return RunnerResult(
            case_id=case_id,
            category=category,
            passed=passed,
            severity=severity,
            actual=actual,
            expected={"name": name},
            details=name,
            failure_reason=None if passed else failure_reason or "assertion_failed",
        )

    def _registry_validation(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        report = SourceRegistryValidator().validate()
        passed = report.valid and "SRC_TCM_DATA_FOR_ML" in report.download_queue
        actual = report.to_dict()
        return self._result(case_id, category, name, severity, passed, actual)

    def _license_validation(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        validator = SourceRegistryValidator()
        approved_sources = validator.download_queue()
        passed = bool(approved_sources) and all(validator.license_status(source) == "approved" for source in approved_sources)
        actual = {"approved_sources": [source["source_id"] for source in approved_sources], "licenses": {source["source_id"]: source.get("license") for source in approved_sources}}
        return self._result(case_id, category, name, severity, passed, actual)

    def _download_pipeline(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        registry_dir = self.tmp_dir / "registry"
        raw_dir = self.tmp_dir / "raw"
        reports_dir = self.tmp_dir / "download_reports"
        _write_download_registry_fixture(registry_dir)
        commands: list[list[str]] = []

        def fake_runner(cmd, **kwargs):
            del kwargs
            commands.append(cmd)
            if cmd[:3] == ["git", "clone", "--depth"]:
                Path(cmd[-1]).mkdir(parents=True)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[-1] == "HEAD":
                return subprocess.CompletedProcess(cmd, 0, stdout="phase9abc123\n", stderr="")
            if cmd[-1] == "--abbrev-ref":
                return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        report = build_download_report(registry_dir=registry_dir, raw_dir=raw_dir, output_path=reports_dir / "download_report.json", runner=fake_runner)
        item = report["items"][0] if report.get("items") else {}
        passed = (
            report.get("downloaded") == 1
            and item.get("source_id") == "SRC_APPROVED_DOWNLOAD"
            and bool(item.get("commit_hash") or item.get("source_version"))
            and (raw_dir / "SRC_APPROVED_DOWNLOAD").exists()
            and not (raw_dir / "SRC_PENDING_DOWNLOAD").exists()
            and not (raw_dir / "SRC_REJECTED_DOWNLOAD").exists()
        )
        actual = {"report": report, "commands": commands}
        return self._result(case_id, category, name, severity, passed, actual)

    def _processed_traceability(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        processed = _processed_data(self.processed_dir)
        source_catalog = _read_json(self.processed_dir / "source_catalog.json")
        issues = validate_source_catalog_traceability(processed, source_catalog)
        validator = ProcessedItemValidator()
        item_issues = []
        for section, items in processed.items():
            candidate_id_field = "case_candidate_id" if section == "learning_case_candidates" else "candidate_id"
            for item in items:
                item_issues.extend(issue.to_dict() | {"section": section} for issue in validator.validate_candidate(item, candidate_id_field=candidate_id_field))
        passed = not issues and not item_issues
        actual = {"traceability_issues": issues, "item_issues": item_issues, "counts": {section: len(items) for section, items in processed.items()}}
        return self._result(case_id, category, name, severity, passed, actual)

    def _source_traceability(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        processed = _processed_data(self.processed_dir)
        source_catalog = _read_json(self.processed_dir / "source_catalog.json")
        catalog_sources = {source["source_id"]: source for source in source_catalog.get("sources", [])}
        issues = validate_source_catalog_traceability(processed, source_catalog)
        required_catalog_fields = {"source_id", "license", "commit_hash", "processed_at", "processing_script", "processing_version"}
        missing_catalog_fields = {
            source_id: sorted(field for field in required_catalog_fields if not source.get(field))
            for source_id, source in catalog_sources.items()
            if any(not source.get(field) for field in required_catalog_fields)
        }
        missing_records = []
        for section, items in processed.items():
            for item in items:
                if not (item.get("source_record_id") or item.get("source_file")):
                    missing_records.append({"section": section, "item_id": item.get("candidate_id") or item.get("case_candidate_id")})
        passed = not issues and not missing_catalog_fields and not missing_records
        actual = {"traceability_issues": issues, "missing_catalog_fields": missing_catalog_fields, "missing_records": missing_records}
        return self._result(case_id, category, name, severity, passed, actual)

    def _content_build(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        report = build_guardtcm_kb(processed_dir=self.processed_dir, content_data_dir=self.content_dir, reports_dir=self.reports_dir)
        validation = ContentValidator(self.content_dir).validate()
        herbs = _items(self.content_dir / "herbs.json")
        formulas = _items(self.content_dir / "formulas.json")
        passed = validation.valid and report["output_counts"]["herbs"] >= 2 and all(item["functions"] == [] and item["cautions"] == [] and item["dose_range"] is None for item in herbs) and all(item["functions"] == [] and item["cautions"] == [] for item in formulas)
        actual = {"build_report": report, "validation": validation.to_dict(), "herb_count": len(herbs), "formula_count": len(formulas)}
        return self._result(case_id, category, name, severity, passed, actual)

    def _content_schema_validation(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        build_guardtcm_kb(processed_dir=self.processed_dir, content_data_dir=self.content_dir, reports_dir=self.reports_dir)
        validation = ContentValidator(self.content_dir).validate()
        passed = validation.valid and validation.summary.get("sources", 0) >= 1 and validation.summary.get("herbs", 0) >= 2
        actual = validation.to_dict()
        return self._result(case_id, category, name, severity, passed, actual)

    def _content_safety_validation(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        build_guardtcm_kb(processed_dir=self.processed_dir, content_data_dir=self.content_dir, reports_dir=self.reports_dir)
        validation = ContentValidator(self.content_dir).validate()
        unsafe_codes = {"PATIENT_UNSAFE_TERM", "PHONE_NUMBER_DETECTED", "ID_CARD_DETECTED"}
        unsafe_errors = [issue.to_dict() for issue in validation.errors if issue.code in unsafe_codes]
        herbs = _items(self.content_dir / "herbs.json")
        formulas = _items(self.content_dir / "formulas.json")
        patient_visible_open_data = [
            item.get("entry_id")
            for item in [*herbs, *formulas]
            if "patient" in item.get("visibility", [])
        ]
        passed = not unsafe_errors and not patient_visible_open_data
        actual = {"unsafe_errors": unsafe_errors, "patient_visible_open_data": patient_visible_open_data}
        return self._result(case_id, category, name, severity, passed, actual)

    def _content_import(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        build_guardtcm_kb(processed_dir=self.processed_dir, content_data_dir=self.content_dir, reports_dir=self.reports_dir)
        first = ContentImportService(self.db, self.content_dir).import_content()
        second = ContentImportService(self.db, self.content_dir).import_content()
        stats = content_stats(self.db)
        passed = first["status"] == "completed" and second["status"] == "completed" and stats["coverage"]["source_coverage_rate"] == 1.0 and stats["knowledge_entries"] >= 2
        actual = {"first": first, "second": second, "stats": stats}
        return self._result(case_id, category, name, severity, passed, actual)

    def _mcp_search(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_imported()
        doctor = self._user("phase9_harness_doctor", "doctor")
        herb = self._call(doctor, "search_herb", {"query": "黄芪"}, "prescription_review")
        formula = self._call(doctor, "search_formula", {"query": "白术"}, "prescription_review")
        herb_evidence = herb.evidence[0] if herb.evidence else {}
        formula_evidence = formula.evidence[0] if formula.evidence else {}
        passed = (
            herb.success
            and formula.success
            and herb_evidence.get("original_source") == "TCMDataForML"
            and herb_evidence.get("metadata", {}).get("processing_script") == "extract_tcmdataforml.py"
            and formula_evidence.get("not_clinical_recommendation") is True
        )
        actual = {"herb": herb.model_dump(mode="json"), "formula": formula.model_dump(mode="json")}
        return self._result(case_id, category, name, severity, passed, actual)

    def _rule_source_binding(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        report = load_rules(self.db)
        rules = get_rules(self.db)
        active = get_rules(self.db, include_disabled=False)
        active_medical = [rule for rule in active if rule.rule_category == "medical_review"]
        rows = self.db.scalars(select(RuleDefinition)).all()
        passed = (
            report["enabled_count"] >= 2
            and all(rule.source_id for rule in active_medical)
            and all("patient" not in rule.applies_to_roles for rule in active_medical)
            and all(not (row.rule_category == "medical_review" and row.enabled and row.risk_level in {"red", "high"} and not row.source_id) for row in rows)
        )
        actual = {
            "report": report,
            "active_rules": [rule.model_dump() for rule in active],
            "all_rules": [rule.model_dump() for rule in rules],
        }
        return self._result(case_id, category, name, severity, passed, actual)

    def _patient_filter(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_imported()
        patient = self._user("phase9_harness_patient", "patient")
        result = self._call(patient, "search_herb", {"query": "黄芪"}, "patient_education")
        passed = result.success and result.evidence == [] and result.metadata.get("degraded") is True
        actual = result.model_dump(mode="json")
        return self._result(case_id, category, name, severity, passed, actual)

    def _learning_deid(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_imported()
        cases = self.db.scalars(select(LearningCase)).all()
        quizzes = self.db.scalars(select(Quiz)).all()
        herbs = {item["name"] for item in _items(self.content_dir / "herbs.json")}
        passed = (
            bool(cases)
            and all(isinstance(case.anonymized_case, dict) and case.anonymized_case.get("deidentified") is True and case.anonymized_case.get("visibility") == "student_learning" for case in cases)
            and bool(quizzes)
            and all(set(quiz.options_json).issubset(herbs) and quiz.answer in herbs for quiz in quizzes)
        )
        actual = {"learning_cases": len(cases), "quizzes": len(quizzes), "herb_pool": sorted(herbs)}
        return self._result(case_id, category, name, severity, passed, actual)

    def _quiz_quality(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        self._ensure_imported()
        quizzes = self.db.scalars(select(Quiz)).all()
        case_ids = {
            case.anonymized_case.get("case_id")
            for case in self.db.scalars(select(LearningCase)).all()
            if isinstance(case.anonymized_case, dict)
        }
        herbs = {item["name"] for item in _items(self.content_dir / "herbs.json")}
        invalid = [
            quiz.quiz_id
            for quiz in quizzes
            if quiz.case_id not in case_ids
            or quiz.answer not in quiz.options_json
            or not quiz.explanation
            or not quiz.knowledge_points_json
            or not set(quiz.options_json).issubset(herbs)
        ]
        passed = bool(quizzes) and not invalid
        actual = {"quiz_count": len(quizzes), "case_ids": sorted(case_ids), "herb_pool": sorted(herbs), "invalid_quizzes": invalid}
        return self._result(case_id, category, name, severity, passed, actual)

    def _reproducibility(self, case_id: str, category: str, name: str, severity: str) -> RunnerResult:
        first_dir = self.tmp_dir / "content_first"
        second_dir = self.tmp_dir / "content_second"
        build_guardtcm_kb(processed_dir=self.processed_dir, content_data_dir=first_dir, reports_dir=self.reports_dir / "first")
        build_guardtcm_kb(processed_dir=self.processed_dir, content_data_dir=second_dir, reports_dir=self.reports_dir / "second")
        file_names = ("sources.json", "herbs.json", "formulas.json", "learning_cases.json", "quizzes.json", "source_catalog.json")
        first = {name: (first_dir / name).read_text(encoding="utf-8") for name in file_names}
        second = {name: (second_dir / name).read_text(encoding="utf-8") for name in file_names}
        passed = first == second
        actual = {"stable_files": sorted(file_names), "changed_files": [file_name for file_name in file_names if first[file_name] != second[file_name]]}
        return self._result(case_id, category, name, severity, passed, actual)

    def _ensure_imported(self) -> None:
        if not (self.content_dir / "sources.json").exists():
            build_guardtcm_kb(processed_dir=self.processed_dir, content_data_dir=self.content_dir, reports_dir=self.reports_dir)
        ContentImportService(self.db, self.content_dir).import_content()

    def _user(self, username: str, role: str) -> User:
        user = self.db.scalar(select(User).where(User.username == username))
        if user:
            return user
        user = User(username=username, hashed_password="harness", role=role, display_name=username)
        self.db.add(user)
        self.db.flush()
        return user

    def _call(self, user: User, tool_name: str, arguments: dict[str, Any], task_type: str):
        context = ToolContext(user_id=user.id, role=user.role, task_type=task_type)
        return MCPGateway(self.db).execute(ToolRequest(tool_name=tool_name, arguments=arguments, context=context))


def _phase9_metrics(results: list[RunnerResult]) -> dict[str, Any]:
    by_category = {result.category: result for result in results}
    registry = by_category.get("source_registry_validation")
    registry_actual = registry.actual if registry else {}
    content_import = by_category.get("content_import")
    content_stats = (content_import.actual.get("stats") if content_import else {}) or {}
    return {
        "approved_source_count": registry_actual.get("approved_count", 0),
        "download_success_rate": _rate(results, "download_pipeline"),
        "license_approved_rate": _rate(results, "license_validation"),
        "processed_item_traceability_rate": _rate(results, "data_processing"),
        "source_traceability_rate": _rate(results, "source_traceability"),
        "content_source_coverage_rate": (content_stats.get("coverage") or {}).get("source_coverage_rate", _rate(results, "content_import")),
        "content_valid_rate": _rate(results, "content_schema_validation"),
        "patient_safe_rate": _rate(results, "content_safety_validation"),
        "open_data_search_coverage": _rate(results, "mcp_search_open_data"),
        "learning_case_deid_pass_rate": _rate(results, "learning_case_deidentification"),
        "deidentification_pass_rate": _rate(results, "learning_case_deidentification"),
        "rule_source_binding_pass_rate": _rate(results, "rule_source_binding"),
        "rule_valid_rate": _rate(results, "rule_source_binding"),
        "patient_safe_filter_pass_rate": _rate(results, "patient_safety_filter"),
        "quiz_valid_rate": _rate(results, "quiz_quality"),
        "open_data_reproducibility_rate": _rate(results, "open_data_reproducibility"),
    }


def _rate(results: list[RunnerResult], category: str) -> float:
    rows = [result for result in results if result.category == category]
    if not rows:
        return 1.0
    return sum(1 for result in rows if result.passed) / len(rows)


def _markdown_summary(total: int, passed: int, failed: int, metrics: dict[str, Any]) -> str:
    pass_rate = passed / total if total else 0.0
    return (
        "# GuardTCM Phase 9 Harness Report\n\n"
        f"本次 Harness 共运行 {total} 个开源数据流水线用例，通过 {passed} 个，失败 {failed} 个，通过率 {pass_rate:.0%}。\n\n"
        "核心结论：知识库内容必须由登记数据源、许可证检查、processed 候选、content 构建和导入链路形成，并可在 MCP 与 Evidence 中追溯来源。\n\n"
        f"- Source coverage：{metrics['content_source_coverage_rate']:.0%}。\n"
        f"- Content valid：{metrics['content_valid_rate']:.0%}。\n"
        f"- MCP open data search：{metrics['open_data_search_coverage']:.0%}。\n"
        f"- Rule source binding：{metrics['rule_source_binding_pass_rate']:.0%}。\n"
        f"- Patient safe filter：{metrics['patient_safe_filter_pass_rate']:.0%}。\n"
        f"- Quiz quality：{metrics['quiz_valid_rate']:.0%}。\n"
        f"- Reproducibility：{metrics['open_data_reproducibility_rate']:.0%}。\n"
    )


def _write_download_registry_fixture(registry_dir: Path) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "data_sources_registry.yaml").write_text(
        "\n".join(
            [
                "sources:",
                "  - source_id: SRC_APPROVED_DOWNLOAD",
                "    name: Approved download source",
                "    homepage: https://github.com/example/approved",
                "    download_url: https://github.com/example/approved.git",
                "    repository_url: https://github.com/example/approved",
                "    source_type: github_dataset",
                "    license: MIT",
                "    license_url: https://github.com/example/approved",
                "    allowed_use: import_allowed",
                "    data_format: csv_txt",
                "    expected_content:",
                "      - prescriptions",
                "    status: approved",
                "    notes: harness fixture",
                "    citation: harness fixture",
                "  - source_id: SRC_PENDING_DOWNLOAD",
                "    name: Pending download source",
                "    homepage: https://github.com/example/pending",
                "    download_url: https://github.com/example/pending.git",
                "    repository_url: https://github.com/example/pending",
                "    source_type: github_dataset",
                "    license: TO_BE_VERIFIED",
                "    license_url: ''",
                "    allowed_use: pending_review",
                "    data_format: csv_txt",
                "    expected_content:",
                "      - prescriptions",
                "    status: candidate",
                "    notes: harness fixture",
                "    citation: harness fixture",
                "  - source_id: SRC_REJECTED_DOWNLOAD",
                "    name: Rejected download source",
                "    homepage: https://github.com/example/rejected",
                "    download_url: https://github.com/example/rejected.git",
                "    repository_url: https://github.com/example/rejected",
                "    source_type: github_dataset",
                "    license: MIT",
                "    license_url: https://github.com/example/rejected",
                "    allowed_use: import_allowed",
                "    data_format: csv_txt",
                "    expected_content:",
                "      - prescriptions",
                "    status: rejected",
                "    notes: harness fixture",
                "    citation: harness fixture",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (registry_dir / "approved_sources.yaml").write_text("approved_sources:\n  - SRC_APPROVED_DOWNLOAD\n", encoding="utf-8")
    (registry_dir / "rejected_sources.yaml").write_text("rejected_sources:\n  - SRC_REJECTED_DOWNLOAD\n", encoding="utf-8")


def _write_processed_fixture(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        processed_dir / "source_catalog.json",
        {
            "sources": [
                {
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "name": "TCMDataForML",
                    "repository_url": "https://github.com/GuoxiangZu/TCMDataForML",
                    "license": "MIT",
                    "commit_hash": "abc123",
                    "downloaded_at": "2026-07-07T00:00:00+00:00",
                    "processed_at": "2026-07-07T00:00:00+00:00",
                    "processing_script": "extract_tcmdataforml.py",
                    "processing_version": "0.9.0",
                    "can_process": True,
                    "notes": "Parsed as open-source prescription/herb candidate data. Not treated as clinical authority.",
                }
            ]
        },
    )
    _write_json(
        processed_dir / "herb_candidates.json",
        {"items": [_herb_candidate(1, "黄芪", "row_1"), _herb_candidate(2, "白术", "row_2"), _herb_candidate(3, "茯苓", "row_3")]},
    )
    _write_json(
        processed_dir / "formula_candidates.json",
        {
            "items": [
                {
                    "candidate_id": "FORMULA_CAND_SRC_TCM_DATA_FOR_ML_000001",
                    "name": "source_formula_000001",
                    "normalized_name": "source_formula_000001",
                    "components": [
                        {"herb_name": "黄芪", "normalized_name": "黄芪", "dosage": None, "unit": None},
                        {"herb_name": "白术", "normalized_name": "白术", "dosage": None, "unit": None},
                    ],
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "source_record_id": "row_10",
                    "source_file": "TCM.csv",
                    "license": "MIT",
                    "evidence_level": "open_dataset_candidate",
                    "confidence": 0.7,
                    "fields_present": ["components"],
                    "fields_missing": ["functions", "cautions", "usage", "source_formula_name"],
                    "visibility_suggestion": ["doctor", "student"],
                    "notes": "Extracted as formula/prescription composition candidate. Not a clinical recommendation.",
                }
            ]
        },
    )
    _write_json(processed_dir / "prescription_candidates.json", {"items": []})
    _write_json(
        processed_dir / "learning_case_candidates.json",
        {
            "items": [
                {
                    "case_candidate_id": "CASE_CAND_SRC_TCM_DATA_FOR_ML_000001",
                    "title": "开源处方组成学习候选案例 000001",
                    "deidentified": True,
                    "source_id": "SRC_TCM_DATA_FOR_ML",
                    "source_record_id": "line_1",
                    "source_file": "textTCM",
                    "license": "MIT",
                    "case_type": "prescription_composition_learning_candidate",
                    "patient_profile": {"age_range": None, "gender": None},
                    "prescription": {"items": [{"herb_name": "黄芪", "dosage": None}], "usage": None},
                    "expected_risks": [],
                    "knowledge_points": ["处方组成识别"],
                    "review_status": "manual_review_required",
                    "visibility": "student_learning",
                    "notes": "Candidate only. Requires manual review before entering official learning case library.",
                }
            ]
        },
    )
    _write_json(processed_dir / "pending_items.json", {"items": []})
    _write_json(processed_dir / "rejected_items.json", {"items": []})


def _herb_candidate(index: int, name: str, record_id: str) -> dict[str, Any]:
    return {
        "candidate_id": f"HERB_CAND_SRC_TCM_DATA_FOR_ML_{index:06d}",
        "name": name,
        "normalized_name": name,
        "original_name": name,
        "aliases": [],
        "source_id": "SRC_TCM_DATA_FOR_ML",
        "source_record_id": record_id,
        "source_file": "IndexName.csv",
        "license": "MIT",
        "evidence_level": "open_dataset_candidate",
        "confidence": 0.8,
        "fields_present": ["name"],
        "fields_missing": ["functions", "cautions", "dose_range"],
        "visibility_suggestion": ["doctor", "student"],
        "notes": "Extracted as herb name candidate only. No function/caution inferred.",
    }


def _processed_data(processed_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "herb_candidates": _items(processed_dir / "herb_candidates.json"),
        "formula_candidates": _items(processed_dir / "formula_candidates.json"),
        "learning_case_candidates": _items(processed_dir / "learning_case_candidates.json"),
    }


def _items(path: Path) -> list[dict[str, Any]]:
    return _read_json(path).get("items", [])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
