from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.content.validators.content_validator import ContentValidator, DEFAULT_CONTENT_DATA_DIR
from app.db.models import ContentImportRun, KnowledgeEntry, KnowledgeSource, LearningCase, Quiz, RuleDefinition, RuleSet, utcnow
from app.knowledge.entity_quality import is_valid_tcm_herb_label
from app.rules.rule_models import StructuredRule
from app.rules.rule_source_policy import normalize_rule_for_source_policy


class ContentImportError(ValueError):
    pass


class ContentImportService:
    def __init__(self, db: Session, content_data_dir: Path | str = DEFAULT_CONTENT_DATA_DIR) -> None:
        self.db = db
        self.content_data_dir = Path(content_data_dir)

    def import_content(self) -> dict[str, Any]:
        validation = ContentValidator(self.content_data_dir).validate()
        if not validation.valid:
            raise ContentImportError(json.dumps(validation.to_dict(), ensure_ascii=False))

        run = ContentImportRun(run_id=f"content_import_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}", status="running")
        self.db.add(run)
        self.db.flush()
        report: dict[str, Any] = {"files": {}, "upserts": {}}
        try:
            sources = _items(self.content_data_dir / "sources.json")
            herbs = _items(self.content_data_dir / "herbs.json")
            formulas = _items(self.content_data_dir / "formulas.json")
            contraindications = _items(self.content_data_dir / "contraindications.json")
            diet_precautions = _items(self.content_data_dir / "diet_precautions.json")
            public_education = _items(self.content_data_dir / "public_education.json")
            review_rules = _items(self.content_data_dir / "review_rules.json")
            learning_cases = _items(self.content_data_dir / "learning_cases.json")
            quizzes = _items(self.content_data_dir / "quizzes.json")
            source_catalog = _source_catalog_map(self.content_data_dir / "source_catalog.json")
            source_map = self._upsert_sources(sources)
            herb_count = self._upsert_knowledge_entries("herb", herbs, source_map, source_catalog)
            formula_count = self._upsert_knowledge_entries("formula", formulas, source_map, source_catalog)
            contraindication_count = self._upsert_knowledge_entries("contraindication", contraindications, source_map, source_catalog)
            diet_count = self._upsert_knowledge_entries("diet_precaution", diet_precautions, source_map, source_catalog)
            education_count = self._upsert_knowledge_entries("public_education", public_education, source_map, source_catalog)
            rule_count = self._upsert_review_rules(review_rules)
            case_count = self._upsert_learning_cases(learning_cases, source_catalog)
            quiz_count = self._upsert_quizzes(quizzes, source_catalog)
            total = len(sources) + herb_count + formula_count + contraindication_count + diet_count + education_count + rule_count + case_count + quiz_count
            report["files"] = {
                "sources": len(sources),
                "herbs": len(herbs),
                "formulas": len(formulas),
                "contraindications": len(contraindications),
                "diet_precautions": len(diet_precautions),
                "public_education": len(public_education),
                "review_rules": len(review_rules),
                "learning_cases": len(learning_cases),
                "quizzes": len(quizzes),
                "source_catalog": len(source_catalog),
            }
            report["upserts"] = {
                "knowledge_sources": len(sources),
                "knowledge_entries": herb_count + formula_count + contraindication_count + diet_count + education_count,
                "rule_definitions": rule_count,
                "learning_cases": case_count,
                "quizzes": quiz_count,
            }
            run.status = "completed"
            run.total_items = total
            run.success_count = total
            run.failed_count = 0
            run.report_json = report
            run.ended_at = utcnow()
            self.db.flush()
            return {
                "run_id": run.run_id,
                "status": run.status,
                "total_items": run.total_items,
                "success_count": run.success_count,
                "failed_count": run.failed_count,
                "report": report,
            }
        except Exception as exc:
            run.status = "failed"
            run.failed_count = 1
            run.report_json = {"error": str(exc), **report}
            run.ended_at = utcnow()
            self.db.flush()
            raise

    def _upsert_sources(self, sources: list[dict[str, Any]]) -> dict[str, KnowledgeSource]:
        source_map: dict[str, KnowledgeSource] = {}
        for raw in sources:
            source_id = raw["source_id"]
            row = self.db.scalar(select(KnowledgeSource).where(KnowledgeSource.source_id == source_id))
            if not row:
                row = KnowledgeSource(source_id=source_id, source_title=raw.get("source_title", source_id), source_type=raw.get("source_type", "open_source_dataset"))
            row.source_title = raw.get("source_title", source_id)
            row.source_type = raw.get("source_type", "open_source_dataset")
            row.source_version = str(raw.get("source_version", ""))
            row.source_year = str(raw.get("source_year", ""))
            row.evidence_level = raw.get("evidence_level", "open_dataset_extracted")
            row.license_note = raw.get("license_note", "")
            self.db.add(row)
            source_map[source_id] = row
        self.db.flush()
        return source_map

    def _upsert_knowledge_entries(
        self,
        entry_type: str,
        items: list[dict[str, Any]],
        source_map: dict[str, KnowledgeSource],
        source_catalog: dict[str, dict[str, Any]],
    ) -> int:
        count = 0
        for raw in items:
            if entry_type == "herb" and not is_valid_tcm_herb_label(str(raw.get("display_name") or raw.get("name") or "")):
                continue
            source_ids = raw.get("source_ids", [])
            source_id = source_ids[0] if source_ids else raw.get("source_id", "")
            source = source_map.get(source_id)
            if not source:
                raise ContentImportError(f"Unknown source_id {source_id} for {raw.get('entry_id')}")
            entry_id = raw["entry_id"]
            title = _knowledge_title(entry_type, raw)
            row = self.db.scalar(select(KnowledgeEntry).where(KnowledgeEntry.source_id == source_id, KnowledgeEntry.entry_id == entry_id))
            if not row:
                row = KnowledgeEntry(entry_id=entry_id, source_id=source_id, entry_type=entry_type, title=title)
            row.entry_type = entry_type
            row.title = title
            row.content = _knowledge_content(entry_type, raw)
            row.keywords_json = raw.get("keywords", [])
            row.source_title = source.source_title
            row.source_type = source.source_type
            row.metadata_json = _metadata_with_source_catalog(raw, source, source_catalog)
            row.updated_at = utcnow()
            self.db.add(row)
            count += 1
        self.db.flush()
        return count

    def _upsert_review_rules(self, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        rule_set = self.db.scalar(select(RuleSet).where(RuleSet.name == "phase9_content_review_rules"))
        if not rule_set:
            rule_set = RuleSet(name="phase9_content_review_rules")
        rule_set.version = "0.9.0"
        rule_set.source = "content/data/review_rules.json"
        rule_set.loaded_at = utcnow()
        self.db.add(rule_set)
        self.db.flush()

        count = 0
        for raw in items:
            rule, _issues = normalize_rule_for_source_policy(StructuredRule.model_validate(_structured_rule_payload(raw)))
            row = self.db.scalar(select(RuleDefinition).where(RuleDefinition.rule_id == rule.rule_id))
            if not row:
                row = RuleDefinition(rule_id=rule.rule_id, rule_set_id=rule_set.id)
            row.rule_set_id = rule_set.id
            row.rule_type = rule.rule_type
            row.rule_category = rule.rule_category
            row.name = rule.name
            row.enabled = rule.enabled
            row.risk_level = rule.risk_level
            row.source_id = rule.source_id
            row.source_record_id = rule.source_record_id
            row.review_status = rule.review_status
            row.applies_to_roles_json = rule.applies_to_roles
            row.condition_json = rule.condition
            row.message = rule.message
            row.evidence_query = rule.evidence_query
            row.metadata_json = rule.metadata
            row.requires_doctor_confirmation = rule.requires_doctor_confirmation
            row.updated_at = utcnow()
            self.db.add(row)
            count += 1
        self.db.flush()
        return count

    def _upsert_learning_cases(self, items: list[dict[str, Any]], source_catalog: dict[str, dict[str, Any]]) -> int:
        count = 0
        existing_rows = self.db.scalars(select(LearningCase)).all()
        existing_by_case_id = {row.anonymized_case.get("case_id"): row for row in existing_rows if isinstance(row.anonymized_case, dict)}
        incoming_tcmeval_ids = {str(item.get("case_id")) for item in items if item.get("source_id") == "SRC_TCMEVAL_SDT"}
        for row in existing_rows:
            payload = row.anonymized_case if isinstance(row.anonymized_case, dict) else {}
            if payload.get("source_id") == "SRC_TCMEVAL_SDT" and payload.get("case_id") not in incoming_tcmeval_ids:
                # TCMEval test/validation rows have no published labels.  When
                # the import policy excludes them, hide stale rows rather than
                # deleting them: a student may already have an answer record
                # referencing the row, and destroying that history violates
                # foreign-key integrity and learning-record retention.
                retired = dict(payload)
                retired["visibility"] = "retired_unlabelled_source"
                retired["review_status"] = "retired_unlabelled_benchmark_row"
                retired["retired_reason"] = "公开源未提供病机/证候答案，不能作为可自动判分的学生练习"
                row.anonymized_case = retired
                self.db.add(row)
        for raw in items:
            case_id = raw["case_id"]
            row = existing_by_case_id.get(case_id)
            if not row:
                row = LearningCase(source_prescription_id=None, title=raw.get("title", case_id), anonymized_case={})
            row.title = raw.get("title", case_id)
            row.anonymized_case = {
                "case_id": case_id,
                "case_type": raw.get("case_type", ""),
                "visibility": raw.get("visibility", "student_learning"),
                "deidentified": raw.get("deidentified", True),
                "patient_profile": raw.get("patient_profile", {}),
                "symptoms": raw.get("symptoms") or raw.get("symptom_summary") or raw.get("chief_complaint") or [],
                "prescription": raw.get("prescription", {}),
                "source_id": raw.get("source_id", ""),
                "source_record_id": raw.get("source_record_id", ""),
                "review_status": raw.get("review_status", ""),
                "knowledge_points": raw.get("knowledge_points", []),
                "training_payload": raw.get("training_payload", {}),
                "notes": raw.get("notes", ""),
                "source_catalog": source_catalog.get(raw.get("source_id", ""), {}),
            }
            row.standard_analysis = raw.get("standard_analysis", "")
            row.risk_points = raw.get("expected_risks", [])
            row.evidence_chain = [{"source_id": raw.get("source_id", ""), "source_record_id": raw.get("source_record_id", "")}]
            self.db.add(row)
            count += 1
        self.db.flush()
        return count

    def _upsert_quizzes(self, items: list[dict[str, Any]], source_catalog: dict[str, dict[str, Any]]) -> int:
        count = 0
        for raw in items:
            quiz_id = raw["quiz_id"]
            row = self.db.scalar(select(Quiz).where(Quiz.quiz_id == quiz_id))
            if not row:
                row = Quiz(quiz_id=quiz_id, case_id=raw.get("case_id", ""), question_type=raw.get("question_type", "single_choice"), question=raw.get("question", ""))
            row.case_id = raw.get("case_id", "")
            row.question_type = raw.get("question_type", "single_choice")
            row.question = raw.get("question", "")
            row.options_json = raw.get("options", [])
            row.answer = raw.get("answer", "")
            row.explanation = raw.get("explanation", "")
            row.knowledge_points_json = raw.get("knowledge_points", [])
            row.source_id = raw.get("source_id", "")
            metadata = dict(raw)
            metadata["source_catalog"] = source_catalog.get(row.source_id, {})
            row.metadata_json = metadata
            row.updated_at = utcnow()
            self.db.add(row)
            count += 1
        self.db.flush()
        return count


def content_db_counts(db: Session) -> dict[str, int]:
    return {
        "sources": db.scalar(select(func.count(KnowledgeSource.id))) or 0,
        "knowledge_entries": db.scalar(select(func.count(KnowledgeEntry.id))) or 0,
        "rule_definitions": db.scalar(select(func.count(RuleDefinition.id))) or 0,
        "learning_cases": db.scalar(select(func.count(LearningCase.id))) or 0,
        "quizzes": db.scalar(select(func.count(Quiz.id))) or 0,
        "content_import_runs": db.scalar(select(func.count(ContentImportRun.id))) or 0,
    }


def _knowledge_content(entry_type: str, raw: dict[str, Any]) -> str:
    if entry_type == "herb":
        return f"{raw.get('name', '')}：开源数据集中提取的药名候选。未推断功效、禁忌或剂量。"
    if entry_type == "formula":
        components = "、".join(component.get("herb_name", "") for component in raw.get("components", []))
        return f"{raw.get('name', '')}：开源数据集中提取的处方组成候选，组成：{components}。不作为临床推荐。"
    if entry_type == "contraindication":
        return raw.get("message_for_doctor") or raw.get("message") or raw.get("title", "")
    if entry_type == "diet_precaution":
        return raw.get("message", "")
    if entry_type == "public_education":
        return raw.get("safe_answer", "")
    return _knowledge_title(entry_type, raw)


def _knowledge_title(entry_type: str, raw: dict[str, Any]) -> str:
    if entry_type in {"herb", "formula"}:
        return raw.get("name", raw.get("entry_id", ""))
    if entry_type == "contraindication":
        return raw.get("title", raw.get("entry_id", ""))
    if entry_type == "diet_precaution":
        return raw.get("title") or raw.get("related_entity") or raw.get("entry_id", "")
    if entry_type == "public_education":
        return raw.get("topic", raw.get("entry_id", ""))
    return raw.get("name") or raw.get("title") or raw.get("entry_id", "")


def _structured_rule_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": raw["rule_id"],
        "rule_type": raw.get("rule_type", "content_review_rule"),
        "rule_category": raw.get("rule_category", "medical_review"),
        "name": raw.get("name", raw["rule_id"]),
        "enabled": raw.get("enabled", True),
        "risk_level": raw.get("risk_level", "yellow"),
        "source_id": raw.get("source_id") or None,
        "source_record_id": raw.get("source_record_id", ""),
        "review_status": raw.get("review_status", "source_bound" if raw.get("source_id") else "manual_review_required"),
        "applies_to_roles": raw.get("applies_to_roles", ["doctor"]),
        "condition": raw.get("condition", {}),
        "message": raw.get("message", ""),
        "evidence_query": raw.get("evidence_query", ""),
        "metadata": {key: value for key, value in raw.items() if key not in {"condition"}},
        "requires_doctor_confirmation": raw.get("requires_doctor_confirmation", True),
    }


def _items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain an items list")
    return items


def _source_catalog_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    sources = data.get("sources", []) if isinstance(data, dict) else []
    if not isinstance(sources, list):
        raise ValueError(f"{path} must contain a sources list")
    return {str(item.get("source_id")): item for item in sources if item.get("source_id")}


def _metadata_with_source_catalog(raw: dict[str, Any], source: KnowledgeSource, source_catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata = dict(raw)
    catalog = dict(source_catalog.get(source.source_id, {}))
    catalog.setdefault("source_id", source.source_id)
    catalog.setdefault("name", source.source_title)
    catalog.setdefault("source_type", source.source_type)
    catalog.setdefault("source_version", source.source_version)
    catalog.setdefault("evidence_level", source.evidence_level)
    catalog.setdefault("license_note", source.license_note)
    metadata["source_catalog"] = catalog
    metadata.setdefault("original_source", catalog.get("name") or source.source_title)
    metadata.setdefault("license", catalog.get("license") or raw.get("license") or "")
    metadata.setdefault("commit_hash", catalog.get("commit_hash") or source.source_version or "")
    metadata.setdefault("processing_script", catalog.get("processing_script") or raw.get("processing_script") or "")
    metadata.setdefault("processing_version", catalog.get("processing_version") or raw.get("processing_version") or "")
    return metadata
