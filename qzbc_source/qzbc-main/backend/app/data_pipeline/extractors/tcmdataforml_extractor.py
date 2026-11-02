from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from app.data_pipeline.extractors.base import BaseSourceExtractor
from app.data_pipeline.normalizers.tcm_entity_normalizer import normalize_entity_name, unique_normalized_names
from app.data_pipeline.normalizers.text_normalizer import normalize_text, split_candidate_text
from app.data_pipeline.scripts.inspect_raw_source import inspect_raw_source
from app.data_pipeline.stable_ids import stable_record_id


PROCESSING_SCRIPT = "extract_tcmdataforml.py"
PROCESSING_VERSION = "0.9.1"


class TCMDataForMLExtractor(BaseSourceExtractor):
    def inspect(self) -> dict[str, Any]:
        return inspect_raw_source(self.source_id, self.raw_dir)

    def extract(self) -> dict[str, Any]:
        result = self.empty_result()
        if not self.source_path.exists():
            result["rejected_items"].append(_rejected(self.source_id, "", "raw_source_missing", f"Missing raw directory: {self.source_path}"))
            result["stats"] = _stats(result)
            return result

        index_files = _find_files(self.source_path, {"indexname.csv", "index_name.csv"})
        tcm_files = _find_files(self.source_path, {"tcm.csv"})
        text_files = _find_text_tcm_files(self.source_path)

        if not index_files:
            result["pending_items"].append(_pending(self.source_id, "", "index_name_file_missing", "IndexName.csv not found", "manual_file_mapping"))
        for file_path in index_files:
            result["herb_candidates"].extend(self._extract_herbs_from_index(file_path, start_index=len(result["herb_candidates"]) + 1))

        if not tcm_files:
            result["pending_items"].append(_pending(self.source_id, "", "tcm_csv_missing", "TCM.csv not found", "manual_file_mapping"))
        for file_path in tcm_files:
            formulas, prescriptions = self._extract_formulas_from_csv(file_path, start_index=len(result["formula_candidates"]) + 1)
            result["formula_candidates"].extend(formulas)
            result["prescription_candidates"].extend(prescriptions)

        for file_path in text_files:
            result["prescription_candidates"].extend(self._extract_prescriptions_from_text(file_path, start_index=len(result["prescription_candidates"]) + 1))

        result["learning_case_candidates"].extend(self._learning_cases_from_prescriptions(result["prescription_candidates"]))
        result["stats"] = _stats(result)
        return result

    def _extract_herbs_from_index(self, file_path: Path, *, start_index: int) -> list[dict[str, Any]]:
        rows = _read_csv_rows(file_path)
        names: list[tuple[str, str]] = []
        for row_number, row in enumerate(rows, start=1):
            values = _entity_name_values(row)
            for value in values:
                for part in split_candidate_text(value):
                    if part.isdigit():
                        continue
                    names.append((part, f"row_{row_number}"))
        unique = unique_normalized_names([name for name, _ in names])
        first_record: dict[str, str] = {}
        for original, record_id in names:
            normalized = normalize_entity_name(original)
            first_record.setdefault(normalized, record_id)
        candidates = []
        for offset, item in enumerate(unique, start=start_index):
            record_id = first_record.get(item["normalized_name"], f"row_{offset}")
            source_file = _relative(file_path, self.source_path)
            candidates.append(
                {
                    "candidate_id": stable_record_id(
                        "HERB",
                        source_id=self.source_id,
                        commit_sha=str(self.download_item.get("commit_hash", "")),
                        source_file=source_file,
                        source_record_id=f"{record_id}#{item['normalized_name']}",
                    ),
                    "name": item["name"],
                    "normalized_name": item["normalized_name"],
                    "original_name": item["original_name"],
                    "aliases": [],
                    "source_id": self.source_id,
                    "source_record_id": record_id,
                    "source_file": source_file,
                    "commit_sha": str(self.download_item.get("commit_hash", "")),
                    "license": self.license_item.get("detected_license") or self.license_item.get("declared_license") or self.source_meta.get("license", ""),
                    "evidence_level": "open_dataset_candidate",
                    "confidence": 0.8,
                    "fields_present": ["name"],
                    "fields_missing": ["functions", "cautions", "dose_range"],
                    "visibility_suggestion": ["doctor", "student"],
                    "notes": "Extracted as herb name candidate only. No function/caution inferred.",
                    "processing_script": PROCESSING_SCRIPT,
                }
            )
        return candidates

    def _extract_formulas_from_csv(self, file_path: Path, *, start_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows = _read_csv_rows(file_path)
        formulas: list[dict[str, Any]] = []
        prescriptions: list[dict[str, Any]] = []
        for row_number, row in enumerate(rows, start=1):
            herbs = _herbs_from_row(row)
            if not herbs:
                continue
            formula_index = start_index + len(formulas)
            name = _formula_name_from_row(row) or f"source_formula_{formula_index:06d}"
            components = [{"herb_name": herb, "normalized_name": normalize_entity_name(herb), "dosage": None, "unit": None} for herb in herbs]
            formulas.append(
                {
                    "candidate_id": stable_record_id(
                        "FORMULA",
                        source_id=self.source_id,
                        commit_sha=str(self.download_item.get("commit_hash", "")),
                        source_file=_relative(file_path, self.source_path),
                        source_record_id=f"row_{row_number}",
                    ),
                    "name": name,
                    "normalized_name": normalize_entity_name(name),
                    "components": components,
                    "source_id": self.source_id,
                    "source_record_id": f"row_{row_number}",
                    "source_file": _relative(file_path, self.source_path),
                    "commit_sha": str(self.download_item.get("commit_hash", "")),
                    "license": self.license_item.get("detected_license") or self.license_item.get("declared_license") or self.source_meta.get("license", ""),
                    "evidence_level": "open_dataset_candidate",
                    "confidence": 0.7,
                    "fields_present": ["components"] + (["source_formula_name"] if _formula_name_from_row(row) else []),
                    "fields_missing": ["functions", "cautions", "usage"] + ([] if _formula_name_from_row(row) else ["source_formula_name"]),
                    "visibility_suggestion": ["doctor", "student"],
                    "notes": "Extracted as formula/prescription composition candidate. Not a clinical recommendation.",
                    "processing_script": PROCESSING_SCRIPT,
                }
            )
            prescriptions.append(_prescription_candidate(self.source_id, file_path, self.source_path, f"row_{row_number}", herbs, self.license_item, self.source_meta, len(prescriptions) + 1, str(self.download_item.get("commit_hash", ""))))
        return formulas, prescriptions

    def _extract_prescriptions_from_text(self, file_path: Path, *, start_index: int) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            herbs = split_candidate_text(line)
            if not herbs:
                continue
            candidates.append(
                _prescription_candidate(
                    self.source_id,
                    file_path,
                    self.source_path,
                    f"line_{line_number}",
                    herbs,
                    self.license_item,
                    self.source_meta,
                    start_index + len(candidates),
                    str(self.download_item.get("commit_hash", "")),
                )
            )
        return candidates

    def _learning_cases_from_prescriptions(self, prescriptions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for index, prescription in enumerate(prescriptions, start=1):
            cases.append(
                {
                    "case_candidate_id": stable_record_id(
                        "CASE",
                        source_id=self.source_id,
                        commit_sha=str(prescription.get("commit_sha", "")),
                        source_file=str(prescription.get("source_file", "")),
                        source_record_id=str(prescription.get("source_record_id", "")),
                    ),
                    "title": f"开源处方组成学习候选案例 {index:06d}",
                    "deidentified": True,
                    "source_id": self.source_id,
                    "source_record_id": prescription["source_record_id"],
                    "source_file": prescription.get("source_file", ""),
                    "commit_sha": prescription.get("commit_sha", ""),
                    "case_type": "prescription_composition_learning_candidate",
                    "patient_profile": {"age_range": None, "gender": None},
                    "prescription": {
                        "items": [{"herb_name": herb, "dosage": None} for herb in prescription.get("herbs", [])],
                        "usage": None,
                    },
                    "expected_risks": [],
                    "knowledge_points": ["处方组成识别"],
                    "review_status": "manual_review_required",
                    "visibility": "student_learning",
                    "license": prescription.get("license", ""),
                    "notes": "Candidate only. Requires manual review before entering official learning case library.",
                    "processing_script": PROCESSING_SCRIPT,
                }
            )
        return cases


def _read_csv_rows(file_path: Path) -> list[dict[str, str]]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(file, dialect=dialect)
        if reader.fieldnames:
            return [{key: normalize_text(value) for key, value in row.items() if key is not None} for row in reader]
        file.seek(0)
        plain_reader = csv.reader(file, dialect=dialect)
        return [{f"column_{index + 1}": normalize_text(value) for index, value in enumerate(row)} for row in plain_reader]


def _find_files(root: Path, names: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name.lower() in names)


def _find_text_tcm_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {"", ".txt", ".text"} and "texttcm" in path.name.lower())


def _formula_name_from_row(row: dict[str, str]) -> str:
    for key, value in row.items():
        if key and any(token in key.lower() for token in ("formula", "prescription", "方", "name")):
            text = normalize_text(value)
            if text:
                return text
    return ""


def _herbs_from_row(row: dict[str, str]) -> list[str]:
    values: list[object] = []
    for key, value in row.items():
        key_lower = str(key).lower()
        if any(token in key_lower for token in ("herb", "medicine", "component", "composition", "药", "组成", "prescription")) or len(row) == 1:
            values.append(value)
    names = unique_normalized_names([part for value in values for part in split_candidate_text(value)])
    return [item["name"] for item in names]


def _entity_name_values(row: dict[str, str]) -> list[str]:
    preferred = []
    for key, value in row.items():
        key_lower = str(key).strip().lower()
        if key_lower in {"name", "herb", "herb_name", "medicine", "药名", "中药名", "名称"}:
            preferred.append(value)
    if preferred:
        return preferred
    return [value for key, value in row.items() if str(key).strip().lower() not in {"index", "id", "序号", "编号"}]


def _prescription_candidate(
    source_id: str,
    file_path: Path,
    source_path: Path,
    record_id: str,
    herbs: list[str],
    license_item: dict[str, Any],
    source_meta: dict[str, Any],
    index: int,
    commit_sha: str,
) -> dict[str, Any]:
    source_file = _relative(file_path, source_path)
    return {
        "candidate_id": stable_record_id(
            "PRESCRIPTION",
            source_id=source_id,
            commit_sha=commit_sha,
            source_file=source_file,
            source_record_id=record_id,
        ),
        "herbs": herbs,
        "normalized_herbs": [normalize_entity_name(herb) for herb in herbs],
        "raw_text": "",
        "source_id": source_id,
        "source_record_id": record_id,
        "source_file": source_file,
        "commit_sha": commit_sha,
        "license": license_item.get("detected_license") or license_item.get("declared_license") or source_meta.get("license", ""),
        "evidence_level": "open_dataset_candidate",
        "usage": None,
        "diagnosis": None,
        "visibility_suggestion": ["student"],
        "notes": "Used as deidentified learning/training candidate only.",
        "processing_script": PROCESSING_SCRIPT,
    }


def _pending(source_id: str, record_id: str, reason: str, raw_summary: str, suggested_action: str) -> dict[str, Any]:
    return {"item_id": "", "source_id": source_id, "source_record_id": record_id, "reason": reason, "raw_summary": raw_summary, "suggested_action": suggested_action}


def _rejected(source_id: str, record_id: str, reason: str, raw_summary: str) -> dict[str, Any]:
    return {"item_id": "", "source_id": source_id, "source_record_id": record_id, "reason": reason, "raw_summary": raw_summary, "rejected_at": ""}


def _stats(result: dict[str, Any]) -> dict[str, int]:
    return {
        "herb_candidates": len(result["herb_candidates"]),
        "formula_candidates": len(result["formula_candidates"]),
        "prescription_candidates": len(result["prescription_candidates"]),
        "learning_case_candidates": len(result["learning_case_candidates"]),
        "pending_items": len(result["pending_items"]),
        "rejected_items": len(result["rejected_items"]),
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
