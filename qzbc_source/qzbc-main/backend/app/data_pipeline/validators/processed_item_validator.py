from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
PATIENT_UNSAFE_TERMS = ("自行开方", "直接服用", "保证疗效", "替代医生", "无需就医", "加量", "停药", "换药", "治疗性食疗")


@dataclass(frozen=True)
class ProcessedItemIssue:
    code: str
    message: str
    field: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "field": self.field}


class ProcessedItemValidator:
    def validate_candidate(self, item: dict[str, Any], *, candidate_id_field: str = "candidate_id") -> list[ProcessedItemIssue]:
        issues: list[ProcessedItemIssue] = []
        if not item.get("source_id"):
            issues.append(ProcessedItemIssue("MISSING_SOURCE_ID", "Processed item must include source_id.", "source_id"))
        if not item.get("source_record_id"):
            issues.append(ProcessedItemIssue("MISSING_SOURCE_RECORD_ID", "Processed item must include source_record_id.", "source_record_id"))
        if not item.get("license"):
            issues.append(ProcessedItemIssue("MISSING_LICENSE", "Processed item must include license.", "license"))
        if not item.get(candidate_id_field):
            issues.append(ProcessedItemIssue("MISSING_CANDIDATE_ID", "Processed item must include candidate_id.", candidate_id_field))
        if not (item.get("name") or item.get("herbs") or item.get("prescription")):
            issues.append(ProcessedItemIssue("MISSING_CONTENT", "Processed item must include name, herbs, or prescription.", "name"))
        issues.extend(_safety_issues(item))
        return issues

    def is_valid(self, item: dict[str, Any], *, candidate_id_field: str = "candidate_id") -> bool:
        return not self.validate_candidate(item, candidate_id_field=candidate_id_field)


def _safety_issues(item: dict[str, Any]) -> list[ProcessedItemIssue]:
    text = str(item)
    issues: list[ProcessedItemIssue] = []
    if PHONE_PATTERN.search(text):
        issues.append(ProcessedItemIssue("PHONE_NUMBER_DETECTED", "Processed item contains phone-like personal data."))
    if ID_CARD_PATTERN.search(text):
        issues.append(ProcessedItemIssue("ID_CARD_DETECTED", "Processed item contains ID-card-like personal data."))
    for term in PATIENT_UNSAFE_TERMS:
        if term in text:
            issues.append(ProcessedItemIssue("PATIENT_UNSAFE_TERM", f"Processed item contains unsafe patient-facing term: {term}"))
    return issues

