from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.data_pipeline.extractors.base import BaseSourceExtractor
from app.data_pipeline.stable_ids import stable_record_id


PROCESSING_SCRIPT = "mtcmb_extractor.py"
PROCESSING_VERSION = "1.0.0"

QUARANTINED_DATASETS = {
    "TCM-CHGD",
    "TCM-Diagnosis",
    "TCM-FRD",
    "TCM-MSDD",
    "TCM-PR",
    "TCMeEE",
}
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
EXACT_DATE_RE = re.compile(r"(?:19|20)\d{2}[-/.年](?:0?[1-9]|1[0-2])[-/.月](?:0?[1-9]|[12]\d|3[01])日?")
INTERNAL_ID_RE = re.compile(r"(?:就诊|住院|门诊|病案|医院内部)(?:号|编号|ID)?\s*[:：]?\s*[A-Za-z0-9-]{4,}")
PRECISE_ADDRESS_RE = re.compile(r"(?:住址|地址)\s*[:：]\s*[^，。；\n]{4,}")
IDENTITY_RE = re.compile(r"(?:患者|病人|医生)[：:]?\s*[\u4e00-\u9fff]{2,4}(?:先生|女士|同志)?")
DIRECTIVE_RE = re.compile(r"(?:给你|建议|应当|可以|请|立即|直接|自行).{0,12}(?:开方|服用|加量|减量|停药|换药|调整用药)")
DOSAGE_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:～|-|至)?\s*\d*(?:\.\d+)?\s*(?:mg|g|克|毫克|ml|毫升)(?![A-Za-z])", re.IGNORECASE)
UNSAFE_TERMS = ("自行开方", "直接服用", "保证疗效", "替代医生", "无需就医", "加量", "停药", "换药", "治疗性食疗")


class MTCMBExtractor(BaseSourceExtractor):
    def inspect(self) -> dict[str, Any]:
        files = sorted((self.source_path / "data").glob("*.jsonl"))
        return {"source_id": self.source_id, "status": "ok" if files else "missing", "files": [str(path.relative_to(self.source_path)) for path in files]}

    def extract(self) -> dict[str, Any]:
        result = self.empty_result()
        result["approved_training_items"] = []
        if not self.source_path.exists():
            result["rejected_items"].append(self._rejected("", "", "raw_source_missing", f"Missing raw directory: {self.source_path}"))
            result["stats"] = self._stats(result)
            return result

        files = sorted((self.source_path / "data").glob("*.jsonl"))
        if not files:
            result["rejected_items"].append(self._rejected("", "", "dataset_files_missing", "data/*.jsonl not found"))
        for path in files:
            dataset = path.stem
            for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                record_id = f"{dataset}:{line_number}"
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    result["rejected_items"].append(self._rejected(path, record_id, "invalid_json", str(exc)))
                    continue
                if not isinstance(record, dict):
                    result["rejected_items"].append(self._rejected(path, record_id, "invalid_record", "JSON line is not an object"))
                    continue
                source_record_id = str(record.get("id", line_number))
                location_id = f"{dataset}:{source_record_id}"
                text = json.dumps(record, ensure_ascii=False)
                issues = _screen_record(text)
                if dataset in QUARANTINED_DATASETS:
                    issues.insert(0, "clinical_or_patient_case_dataset")
                if issues:
                    result["rejected_items"].append(self._rejected(path, location_id, ",".join(sorted(set(issues))), _safe_summary(record)))
                    continue
                item = self._training_item(path, dataset, location_id, record)
                result["approved_training_items"].append(item)
                result["learning_case_candidates"].append(item)

        result["stats"] = self._stats(result)
        return result

    def _training_item(self, path: Path, dataset: str, record_id: str, record: dict[str, Any]) -> dict[str, Any]:
        source_file = str(path.relative_to(self.source_path))
        commit_sha = str(self.download_item.get("commit_hash", ""))
        item_id = stable_record_id(
            "CASE",
            source_id=self.source_id,
            commit_sha=commit_sha,
            source_file=source_file,
            source_record_id=record_id,
        )
        question = str(record.get("question") or record.get("text") or "中医教学评测条目")
        return {
            "case_candidate_id": item_id,
            "title": f"MTCMB 教学评测：{dataset} / {record_id}",
            "case_type": "mtcmb_student_evaluation",
            "deidentified": True,
            "source_id": self.source_id,
            "source_record_id": record_id,
            "source_file": source_file,
            "commit_sha": commit_sha,
            "license": self.license_item.get("detected_license") or self.source_meta.get("license", ""),
            "patient_profile": {},
            "prescription": {"items": [], "usage": None},
            "expected_risks": [],
            "knowledge_points": ["中医知识评测", dataset],
            "training_payload": {
                "question": question,
                "options": record.get("options"),
                "answer": record.get("answer") or record.get("points") or record.get("annotations"),
                "original_source": record.get("source", ""),
            },
            "review_status": "approved_training_only",
            "visibility": "student_learning",
            "clinical_evidence_allowed": False,
            "patient_visible": False,
            "doctor_evidence_allowed": False,
            "student_visible": True,
            "evaluation_allowed": True,
            "notes": "Only for deidentified student training and evaluation; never clinical evidence.",
            "processing_script": PROCESSING_SCRIPT,
        }

    def _rejected(self, path: Path | str, record_id: str, reason: str, summary: str) -> dict[str, Any]:
        source_file = str(path.relative_to(self.source_path)) if isinstance(path, Path) else str(path)
        return {
            "item_id": "",
            "source_id": self.source_id,
            "source_record_id": record_id,
            "source_file": source_file,
            "reason": reason,
            "raw_summary": summary[:240],
            "rejected_at": "",
        }

    @staticmethod
    def _stats(result: dict[str, Any]) -> dict[str, int]:
        return {
            "herb_candidates": 0,
            "formula_candidates": 0,
            "prescription_candidates": 0,
            "learning_case_candidates": len(result["learning_case_candidates"]),
            "approved_training_items": len(result["approved_training_items"]),
            "pending_items": len(result["pending_items"]),
            "rejected_items": len(result["rejected_items"]),
        }


def _screen_record(text: str) -> list[str]:
    checks = (
        (PHONE_RE, "phone_number_detected"),
        (ID_CARD_RE, "id_card_detected"),
        (EXACT_DATE_RE, "exact_date_detected"),
        (INTERNAL_ID_RE, "internal_record_id_detected"),
        (PRECISE_ADDRESS_RE, "precise_address_detected"),
        (IDENTITY_RE, "potential_identity_detected"),
        (DIRECTIVE_RE, "dangerous_medical_directive"),
        (DOSAGE_RE, "dosage_output_detected"),
    )
    issues = [code for pattern, code in checks if pattern.search(text)]
    issues.extend(f"unsafe_term:{term}" for term in UNSAFE_TERMS if term in text)
    return issues


def _safe_summary(record: dict[str, Any]) -> str:
    return f"fields={sorted(record.keys())}; source={str(record.get('source', ''))[:120]}"
