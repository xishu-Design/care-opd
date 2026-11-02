from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.data_pipeline.validators.source_registry_validator import DATA_PIPELINE_DIR, SourceRegistryValidator


DEFAULT_RAW_DIR = DATA_PIPELINE_DIR / "raw"
DEFAULT_REPORTS_DIR = DATA_PIPELINE_DIR / "reports"

LICENSE_FILE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.txt")
ALLOWED_LICENSES = {"MIT", "Apache-2.0", "CC-BY", "CC0", "BSD-3-Clause", "BSD-2-Clause"}
PENDING_LICENSES = {"", "TO_BE_VERIFIED", "UNKNOWN", "NO_LICENSE_FOUND", "PENDING"}
REJECTED_LICENSE_MARKERS = {"PROPRIETARY", "NON-COMMERCIAL-ONLY", "RESEARCH-ONLY-WITHOUT-REDISTRIBUTION"}


@dataclass
class LicenseCheckResult:
    source_id: str
    declared_license: str
    detected_license: str
    license_file_found: bool
    license_file_path: str
    can_process: bool
    can_import: bool
    status: str
    notes: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "declared_license": self.declared_license,
            "detected_license": self.detected_license,
            "license_file_found": self.license_file_found,
            "license_file_path": self.license_file_path,
            "can_process": self.can_process,
            "can_import": self.can_import,
            "status": self.status,
            "notes": self.notes,
            "warnings": self.warnings,
        }


class LicenseValidator:
    def __init__(self, raw_dir: Path | str = DEFAULT_RAW_DIR) -> None:
        self.raw_dir = Path(raw_dir)

    def check_source(self, source: dict[str, Any], local_path: Path | str | None = None) -> LicenseCheckResult:
        source_id = str(source.get("source_id", ""))
        declared_license = str(source.get("license", "UNKNOWN") or "UNKNOWN")
        source_path = Path(local_path) if local_path else self.raw_dir / source_id
        license_file = _find_license_file(source_path)
        detected_license = _detect_license(license_file, declared_license)
        warnings: list[str] = []

        declared_norm = _normalize_license(declared_license)
        detected_norm = _normalize_license(detected_license)
        if license_file and declared_norm not in {"", "to-be-verified", "unknown"} and detected_norm != declared_norm:
            warnings.append(f"Declared license {declared_license} differs from detected license {detected_license}.")

        if not license_file:
            return LicenseCheckResult(
                source_id=source_id,
                declared_license=declared_license,
                detected_license="NO_LICENSE_FOUND",
                license_file_found=False,
                license_file_path="",
                can_process=False,
                can_import=False,
                status="pending_review",
                notes="No LICENSE/COPYING file found in downloaded source.",
                warnings=warnings,
            )

        if _is_allowed(detected_license) and _is_allowed(declared_license):
            return LicenseCheckResult(
                source_id=source_id,
                declared_license=declared_license,
                detected_license=detected_license,
                license_file_found=True,
                license_file_path=str(license_file),
                can_process=True,
                can_import=True,
                status="approved",
                notes=f"{detected_license} license detected from repository.",
                warnings=warnings,
            )

        if _is_rejected(detected_license) or _is_rejected(declared_license):
            return LicenseCheckResult(
                source_id=source_id,
                declared_license=declared_license,
                detected_license=detected_license,
                license_file_found=True,
                license_file_path=str(license_file),
                can_process=False,
                can_import=False,
                status="rejected",
                notes="License is not approved for automatic processing.",
                warnings=warnings,
            )

        return LicenseCheckResult(
            source_id=source_id,
            declared_license=declared_license,
            detected_license=detected_license,
            license_file_found=True,
            license_file_path=str(license_file),
            can_process=False,
            can_import=False,
            status="pending_review",
            notes="License requires manual review before processing.",
            warnings=warnings,
        )


def generate_license_report(
    *,
    download_report_path: Path | str = DEFAULT_REPORTS_DIR / "download_report.json",
    output_path: Path | str = DEFAULT_REPORTS_DIR / "license_report.json",
    registry_dir: Path | str | None = None,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    source_id: str | None = None,
) -> dict[str, Any]:
    registry = SourceRegistryValidator(registry_dir) if registry_dir else SourceRegistryValidator()
    sources_by_id = {source["source_id"]: source for source in registry.load_sources()}
    download_report = _load_json(Path(download_report_path))
    started_at = _now()
    items: list[dict[str, Any]] = []
    validator = LicenseValidator(raw_dir)

    for download_item in download_report.get("items", []):
        item_source_id = str(download_item.get("source_id", ""))
        if source_id and item_source_id != source_id:
            continue
        if download_item.get("download_status") not in {"success", "skipped"}:
            items.append(
                {
                    "source_id": item_source_id,
                    "declared_license": sources_by_id.get(item_source_id, {}).get("license", "UNKNOWN"),
                    "detected_license": "NOT_CHECKED",
                    "license_file_found": False,
                    "license_file_path": "",
                    "can_process": False,
                    "can_import": False,
                    "status": "pending_review",
                    "notes": "Download did not succeed; license was not checked.",
                    "warnings": [],
                }
            )
            continue
        source = sources_by_id.get(item_source_id)
        if not source:
            items.append(
                {
                    "source_id": item_source_id,
                    "declared_license": "UNKNOWN",
                    "detected_license": "UNKNOWN_SOURCE",
                    "license_file_found": False,
                    "license_file_path": "",
                    "can_process": False,
                    "can_import": False,
                    "status": "pending_review",
                    "notes": "Downloaded source is not registered.",
                    "warnings": [],
                }
            )
            continue
        result = validator.check_source(source, download_item.get("local_path") or None)
        items.append(result.to_dict())

    report = {
        "run_id": f"license_{_compact_timestamp()}",
        "started_at": started_at,
        "ended_at": _now(),
        "total_sources": len(items),
        "approved_for_processing": sum(1 for item in items if item.get("can_process") is True),
        "pending_review": sum(1 for item in items if item.get("status") == "pending_review"),
        "rejected": sum(1 for item in items if item.get("status") == "rejected"),
        "items": items,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _find_license_file(source_path: Path) -> Path | None:
    for file_name in LICENSE_FILE_NAMES:
        candidate = source_path / file_name
        if candidate.is_file():
            return candidate
    return None


def _detect_license(license_file: Path | None, declared_license: str) -> str:
    if not license_file:
        return "NO_LICENSE_FOUND"
    text = license_file.read_text(encoding="utf-8", errors="ignore")[:8000].lower()
    if "mit license" in text or "permission is hereby granted, free of charge" in text:
        return "MIT"
    if "apache license" in text and "version 2.0" in text:
        return "Apache-2.0"
    if "creative commons attribution" in text or "cc-by" in text or "cc by" in text:
        return "CC-BY"
    if "creative commons zero" in text or "cc0" in text:
        return "CC0"
    if "bsd 3-clause" in text or "redistribution and use in source and binary forms" in text:
        return "BSD-3-Clause"
    declared = declared_license.strip()
    return declared if declared else "UNKNOWN"


def _is_allowed(value: str) -> bool:
    return _normalize_license(value) in {_normalize_license(item) for item in ALLOWED_LICENSES}


def _is_rejected(value: str) -> bool:
    return _normalize_license(value) in {_normalize_license(item) for item in REJECTED_LICENSE_MARKERS}


def _normalize_license(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _compact_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")
