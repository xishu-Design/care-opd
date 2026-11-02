from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.data_pipeline.scripts.process_approved_sources import DEFAULT_PROCESSED_DIR
from app.data_pipeline.validators.license_validator import DEFAULT_REPORTS_DIR
from app.data_pipeline.validators.processed_item_validator import ProcessedItemValidator
from app.data_pipeline.validators.traceability_validator import validate_source_catalog_traceability


def generate_processing_report(
    *,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    output_path: Path | str = DEFAULT_REPORTS_DIR / "processing_report.json",
) -> dict[str, Any]:
    processed_root = Path(processed_dir)
    processed = {
        "herb_candidates": _items(processed_root / "herb_candidates.json"),
        "formula_candidates": _items(processed_root / "formula_candidates.json"),
        "prescription_candidates": _items(processed_root / "prescription_candidates.json"),
        "learning_case_candidates": _items(processed_root / "learning_case_candidates.json"),
    }
    source_catalog = _read_json(processed_root / "source_catalog.json") if (processed_root / "source_catalog.json").exists() else {"sources": []}
    validator = ProcessedItemValidator()
    item_issues: list[dict[str, Any]] = []
    for section, items in processed.items():
        id_field = "case_candidate_id" if section == "learning_case_candidates" else "candidate_id"
        for item in items:
            item_issues.extend(issue.to_dict() | {"section": section} for issue in validator.validate_candidate(item, candidate_id_field=id_field))
    traceability_issues = validate_source_catalog_traceability(
        {
            "herb_candidates": processed["herb_candidates"],
            "formula_candidates": processed["formula_candidates"],
            "learning_case_candidates": processed["learning_case_candidates"],
        },
        source_catalog,
    )
    pending_count = len(_items(processed_root / "pending_items.json"))
    rejected_count = len(_items(processed_root / "rejected_items.json"))
    report = {
        "run_id": f"processing_report_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "processed_dir": str(processed_root),
        "counts": {section: len(items) for section, items in processed.items()},
        "pending_items": pending_count,
        "rejected_items": rejected_count,
        "traceability_issues": traceability_issues,
        "item_issues": item_issues,
        "valid": not traceability_issues and not item_issues,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a GuardTCM Phase 9 processed-data quality report.")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--output", default=str(DEFAULT_REPORTS_DIR / "processing_report.json"))
    args = parser.parse_args(argv)
    report = generate_processing_report(processed_dir=args.processed_dir, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = _read_json(path)
    items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain an items list")
    return items


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
