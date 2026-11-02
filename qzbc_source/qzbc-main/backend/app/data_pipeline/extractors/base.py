from __future__ import annotations

from pathlib import Path
from typing import Any


class BaseSourceExtractor:
    source_id: str

    def __init__(self, raw_dir: Path | str, source_meta: dict[str, Any], license_item: dict[str, Any], download_item: dict[str, Any] | None = None) -> None:
        self.raw_dir = Path(raw_dir)
        self.source_meta = source_meta
        self.license_item = license_item
        self.download_item = download_item or {}
        self.source_id = str(source_meta.get("source_id") or license_item.get("source_id") or "")
        self.source_path = self.raw_dir / self.source_id

    def inspect(self) -> dict[str, Any]:
        raise NotImplementedError

    def extract(self) -> dict[str, Any]:
        raise NotImplementedError

    def empty_result(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "herb_candidates": [],
            "formula_candidates": [],
            "prescription_candidates": [],
            "learning_case_candidates": [],
            "approved_training_items": [],
            "pending_items": [],
            "rejected_items": [],
            "stats": {},
        }
