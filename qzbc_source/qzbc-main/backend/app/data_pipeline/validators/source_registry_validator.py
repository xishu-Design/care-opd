from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.core.config import get_settings

import yaml


DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_DIR = DATA_PIPELINE_DIR / "registry"

REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "name",
    "homepage",
    "download_url",
    "repository_url",
    "source_type",
    "license",
    "license_url",
    "allowed_use",
    "data_format",
    "expected_content",
    "status",
    "notes",
    "citation",
}

VALID_STATUSES = {"approved", "candidate", "pending", "rejected", "reference_only"}
VALID_ALLOWED_USE = {"import_allowed", "pending_review", "reference_only", "rejected", "no_import"}
APPROVED_LICENSES = {"mit", "apache-2.0", "apache 2.0", "cc-by", "cc by", "cc-by-4.0", "cc0", "bsd-2-clause", "bsd-3-clause"}
PENDING_LICENSE_MARKERS = {"", "to-be-verified", "to_be_verified", "unknown", "not-found", "not_found", "pending"}

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class SourceRegistryIssue:
    severity: Severity
    code: str
    message: str
    source_id: str = ""
    field: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "source_id": self.source_id,
            "field": self.field,
        }


@dataclass
class SourceRegistryReport:
    valid: bool
    source_count: int
    approved_count: int
    pending_count: int
    rejected_count: int
    download_queue: list[str] = field(default_factory=list)
    issues: list[SourceRegistryIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[SourceRegistryIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[SourceRegistryIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "source_count": self.source_count,
            "approved_count": self.approved_count,
            "pending_count": self.pending_count,
            "rejected_count": self.rejected_count,
            "download_queue": self.download_queue,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class SourceRegistryValidator:
    def __init__(self, registry_dir: Path | str = DEFAULT_REGISTRY_DIR) -> None:
        self.registry_dir = Path(registry_dir)

    def validate(self) -> SourceRegistryReport:
        sources = self.load_sources()
        approved_ids = self.load_approved_ids()
        rejected_ids = self.load_rejected_ids()
        source_map: dict[str, dict[str, Any]] = {}
        issues: list[SourceRegistryIssue] = []

        for source in sources:
            source_id = str(source.get("source_id", ""))
            for field_name in sorted(REQUIRED_SOURCE_FIELDS):
                if field_name not in source:
                    issues.append(_issue("error", "MISSING_FIELD", f"Missing required field: {field_name}", source_id, field_name))
            if not source_id:
                issues.append(_issue("error", "MISSING_SOURCE_ID", "Source is missing source_id", "", "source_id"))
                continue
            if source_id in source_map:
                issues.append(_issue("error", "DUPLICATE_SOURCE_ID", "Duplicate source_id", source_id, "source_id"))
            source_map[source_id] = source
            issues.extend(self._validate_source(source, approved_ids, rejected_ids))

        issues.extend(self._validate_source_lists(source_map, approved_ids, rejected_ids))
        download_queue = [source["source_id"] for source in sources if self.can_download(source, approved_ids, rejected_ids)]
        pending_count = sum(1 for source in sources if self.license_status(source) == "pending_review" or source.get("status") in {"candidate", "pending", "reference_only"})
        rejected_count = sum(1 for source in sources if str(source.get("source_id", "")) in rejected_ids or source.get("status") == "rejected")
        approved_count = len(download_queue)
        return SourceRegistryReport(
            valid=not any(issue.severity == "error" for issue in issues),
            source_count=len(sources),
            approved_count=approved_count,
            pending_count=pending_count,
            rejected_count=rejected_count,
            download_queue=download_queue,
            issues=issues,
        )

    def load_sources(self) -> list[dict[str, Any]]:
        data = _load_yaml(self.registry_dir / "data_sources_registry.yaml")
        sources = data.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError("data_sources_registry.yaml must contain a sources list")
        if not all(isinstance(source, dict) for source in sources):
            raise ValueError("Each source registry item must be a mapping")
        return sources

    def load_approved_ids(self) -> set[str]:
        data = _load_yaml(self.registry_dir / "approved_sources.yaml")
        return _load_id_set(data.get("approved_sources", []), "approved_sources")

    def load_rejected_ids(self) -> set[str]:
        data = _load_yaml(self.registry_dir / "rejected_sources.yaml")
        return _load_id_set(data.get("rejected_sources", []), "rejected_sources")

    def download_queue(self) -> list[dict[str, Any]]:
        approved_ids = self.load_approved_ids()
        rejected_ids = self.load_rejected_ids()
        return [source for source in self.load_sources() if self.can_download(source, approved_ids, rejected_ids)]

    def can_download(self, source: dict[str, Any], approved_ids: set[str] | None = None, rejected_ids: set[str] | None = None) -> bool:
        source_id = str(source.get("source_id", ""))
        approved_ids = approved_ids if approved_ids is not None else self.load_approved_ids()
        rejected_ids = rejected_ids if rejected_ids is not None else self.load_rejected_ids()
        verified_or_internal = self.license_status(source) == "approved" or (
            get_settings().allow_unverified_internal_sources
            and source.get("license_status") == "unverified"
            and source.get("distribution_scope") == "internal_only"
        )
        return (
            source_id in approved_ids
            and source_id not in rejected_ids
            and source.get("status") == "approved"
            and source.get("allowed_use") == "import_allowed"
            and verified_or_internal
            and bool(source.get("download_url"))
        )

    def license_status(self, source: dict[str, Any]) -> str:
        license_name = _normalize_license(str(source.get("license", "")))
        if license_name in APPROVED_LICENSES:
            return "approved"
        if license_name in PENDING_LICENSE_MARKERS or source.get("allowed_use") == "pending_review" or source.get("status") in {"candidate", "pending"}:
            return "pending_review"
        if source.get("allowed_use") in {"rejected", "no_import"} or source.get("status") == "rejected":
            return "rejected"
        return "requires_review"

    def _validate_source(self, source: dict[str, Any], approved_ids: set[str], rejected_ids: set[str]) -> list[SourceRegistryIssue]:
        issues: list[SourceRegistryIssue] = []
        source_id = str(source.get("source_id", ""))
        status = source.get("status")
        allowed_use = source.get("allowed_use")
        if status and status not in VALID_STATUSES:
            issues.append(_issue("error", "INVALID_STATUS", f"Invalid source status: {status}", source_id, "status"))
        if allowed_use and allowed_use not in VALID_ALLOWED_USE:
            issues.append(_issue("error", "INVALID_ALLOWED_USE", f"Invalid allowed_use: {allowed_use}", source_id, "allowed_use"))
        if source.get("expected_content") is not None and not isinstance(source.get("expected_content"), list):
            issues.append(_issue("error", "INVALID_EXPECTED_CONTENT", "expected_content must be a list", source_id, "expected_content"))
        if source_id in approved_ids:
            issues.extend(self._validate_approved_source(source))
        if source_id in rejected_ids and status == "approved":
            issues.append(_issue("error", "REJECTED_SOURCE_APPROVED", "Rejected source cannot be status=approved", source_id, "status"))
        if source_id not in approved_ids and status == "approved":
            issues.append(_issue("warning", "APPROVED_STATUS_NOT_IN_QUEUE", "Approved source is not listed in approved_sources.yaml", source_id, "status"))
        return issues

    def _validate_approved_source(self, source: dict[str, Any]) -> list[SourceRegistryIssue]:
        source_id = str(source.get("source_id", ""))
        issues: list[SourceRegistryIssue] = []
        if source.get("status") != "approved":
            issues.append(_issue("error", "APPROVED_SOURCE_NOT_APPROVED", "Approved source must have status=approved", source_id, "status"))
        if source.get("allowed_use") != "import_allowed":
            issues.append(_issue("error", "APPROVED_SOURCE_NOT_IMPORT_ALLOWED", "Approved source must have allowed_use=import_allowed", source_id, "allowed_use"))
        if self.license_status(source) != "approved" and not (
            get_settings().allow_unverified_internal_sources
            and source.get("license_status") == "unverified"
            and source.get("distribution_scope") == "internal_only"
        ):
            issues.append(_issue("error", "APPROVED_SOURCE_LICENSE_NOT_APPROVED", "Approved source must use an approved license", source_id, "license"))
        if not source.get("download_url"):
            issues.append(_issue("error", "APPROVED_SOURCE_MISSING_DOWNLOAD_URL", "Approved source must have a download_url", source_id, "download_url"))
        return issues

    def _validate_source_lists(self, source_map: dict[str, dict[str, Any]], approved_ids: set[str], rejected_ids: set[str]) -> list[SourceRegistryIssue]:
        issues: list[SourceRegistryIssue] = []
        for source_id in sorted(approved_ids - source_map.keys()):
            issues.append(_issue("error", "APPROVED_SOURCE_NOT_REGISTERED", "approved_sources.yaml references an unknown source", source_id, "approved_sources"))
        for source_id in sorted(rejected_ids - source_map.keys()):
            issues.append(_issue("error", "REJECTED_SOURCE_NOT_REGISTERED", "rejected_sources.yaml references an unknown source", source_id, "rejected_sources"))
        for source_id in sorted(approved_ids & rejected_ids):
            issues.append(_issue("error", "SOURCE_APPROVED_AND_REJECTED", "Source cannot be both approved and rejected", source_id, "approved_sources"))
        return issues


def validate_default_registry() -> SourceRegistryReport:
    return SourceRegistryValidator().validate()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    return data


def _load_id_set(items: Any, field_name: str) -> set[str]:
    if not isinstance(items, list):
        raise ValueError(f"{field_name} must be a list")
    ids: set[str] = set()
    for item in items:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict) and item.get("source_id"):
            ids.add(str(item["source_id"]))
        else:
            raise ValueError(f"{field_name} items must be source_id strings or mappings with source_id")
    return ids


def _normalize_license(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _issue(severity: Severity, code: str, message: str, source_id: str = "", field_name: str = "") -> SourceRegistryIssue:
    return SourceRegistryIssue(severity=severity, code=code, message=message, source_id=source_id, field=field_name)
