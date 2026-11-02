"""Load qzbc's reviewed, provenance-bearing TCM assets for seed synthesis.

The post-training corpus must not collapse into generic format examples.  This
loader reuses the knowledge already curated by qzbc while enforcing the
visibility/review labels stored in that repository.  No patient database,
chat history, Graphiti store, or Mem0 store is read.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


ASSET_PATHS: Mapping[str, str] = {
    "herbs": "backend/app/content/data/herbs.json",
    "formulas": "backend/app/content/data/formulas.json",
    "incompatibilities": "backend/app/content/data/herb_incompatibilities.json",
    "diet_precautions": "backend/app/content/data/diet_precautions.json",
    "approved_training_items": (
        "backend/app/data_pipeline/processed/approved_training_items.json"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: root must be an object")
    return value


def _items(value: Mapping[str, Any], key: str = "items") -> list[Mapping[str, Any]]:
    rows = value.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"asset field {key!r} must be an array")
    return [row for row in rows if isinstance(row, Mapping)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _texts(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return tuple()
    return tuple(_text(item) for item in value if _text(item))


def _visible_to(value: Mapping[str, Any], role: str) -> bool:
    visibility = value.get("visibility")
    if isinstance(visibility, str):
        return visibility in {role, f"{role}_learning", f"{role}_internal"}
    if isinstance(visibility, list):
        return role in {str(item) for item in visibility}
    return False


def _primary_source_record(row: Mapping[str, Any]) -> Mapping[str, Any]:
    records = row.get("source_records")
    if not isinstance(records, list):
        return {}
    for record in records:
        if isinstance(record, Mapping):
            return record
    return {}


@dataclass(frozen=True)
class QZBCGroundingAssets:
    herbs: Tuple[Mapping[str, Any], ...]
    formulas: Tuple[Mapping[str, Any], ...]
    incompatibilities: Tuple[Mapping[str, Any], ...]
    diet_precautions: Tuple[Mapping[str, Any], ...]
    approved_training_items: Tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]

    def require_coverage(self) -> None:
        missing = [
            name
            for name in (
                "herbs",
                "formulas",
                "incompatibilities",
                "diet_precautions",
                "approved_training_items",
            )
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"qzbc grounding assets have no usable rows: {missing}")


def _normalize_herb(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    if not _visible_to(row, "doctor") or not _visible_to(row, "student"):
        return None
    name = _text(row.get("display_name")) or _text(row.get("name"))
    source_ids = _texts(row.get("source_ids"))
    functions = _texts(row.get("display_functions")) or _texts(row.get("functions"))
    cautions = _texts(row.get("cautions"))
    properties = _texts(row.get("properties"))
    if not name or not source_ids or not (functions or cautions or properties):
        return None
    source_record = _primary_source_record(row)
    return {
        "entry_id": _text(row.get("entry_id")),
        "name": name,
        "original_name": _text(row.get("original_name")) or _text(row.get("name")),
        "functions": functions,
        "cautions": cautions,
        "properties": properties,
        "source_ids": source_ids,
        "source_record_id": _text(source_record.get("source_record_id")),
        "evidence_level": _text(row.get("evidence_level")),
        "license": _text(source_record.get("license")),
        "license_note": _text(row.get("license_note")),
        "translation_note": _text(row.get("translation_note")),
    }


def _normalize_formula(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    if not _visible_to(row, "doctor") or not _visible_to(row, "student"):
        return None
    name = _text(row.get("display_name")) or _text(row.get("name"))
    source_ids = _texts(row.get("source_ids"))
    components = []
    raw_components = row.get("components")
    if isinstance(raw_components, list):
        for component in raw_components:
            if not isinstance(component, Mapping):
                continue
            herb_name = _text(component.get("herb_name"))
            if herb_name:
                components.append(herb_name)
    functions = _texts(row.get("display_functions")) or _texts(row.get("functions"))
    if not name or not source_ids or len(components) < 2:
        return None
    source_record = _primary_source_record(row)
    return {
        "entry_id": _text(row.get("entry_id")),
        "name": name,
        "original_name": _text(row.get("original_name")) or _text(row.get("name")),
        "components": tuple(components),
        "functions": functions,
        "cautions": _texts(row.get("cautions")),
        "source_ids": source_ids,
        "source_record_id": _text(source_record.get("source_record_id")),
        "evidence_level": _text(row.get("evidence_level")),
        "license": _text(source_record.get("license")),
        "license_note": _text(row.get("license_note")),
        "translation_note": _text(row.get("translation_note")),
    }


def _normalize_incompatibility(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    review_status = _text(row.get("review_status"))
    if review_status not in {
        "manual_transcription_verified_against_pdf",
        "manually_verified",
        "approved",
    }:
        return None
    left = tuple(
        _text(item.get("name"))
        for item in row.get("left", [])
        if isinstance(item, Mapping) and _text(item.get("name"))
    )
    right = tuple(
        _text(item.get("name"))
        for item in row.get("right", [])
        if isinstance(item, Mapping) and _text(item.get("name"))
    )
    source_id = _text(row.get("source_id"))
    if not left or not right or not source_id:
        return None
    return {
        "constraint_id": _text(row.get("constraint_id")),
        "category": _text(row.get("category")),
        "relation_type": _text(row.get("relation_type")),
        "left": left,
        "right": right,
        "risk_level": _text(row.get("risk_level")) or "red",
        "action": _text(row.get("action")),
        "source_id": source_id,
        "source_record_id": _text(row.get("source_record_id")),
        "evidence_level": _text(row.get("evidence_level")),
        "review_status": review_status,
    }


def _normalize_diet(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    source_id = _text(row.get("source_id"))
    message = _text(row.get("message"))
    if not source_id or not message or not _visible_to(row, "patient"):
        return None
    return {
        "entry_id": _text(row.get("entry_id")),
        "title": _text(row.get("title")),
        "related_entities": _texts(row.get("related_entities")),
        "food_items": _texts(row.get("food_items")),
        "advice_type": _text(row.get("advice_type")),
        "message": message,
        "forbidden_as_treatment": bool(row.get("forbidden_as_treatment")),
        "source_id": source_id,
        "source_record_id": _text(row.get("source_record_id")),
    }


def _normalize_training_item(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    if (
        _text(row.get("review_status")) != "approved_training_only"
        or row.get("deidentified") is not True
        or row.get("student_visible") is not True
        or row.get("evaluation_allowed") is not True
        or row.get("clinical_evidence_allowed") is not False
    ):
        return None
    payload = row.get("training_payload")
    if not isinstance(payload, Mapping):
        return None
    question = _text(payload.get("question"))
    answer = _text(payload.get("answer"))
    options_value = payload.get("options")
    if isinstance(options_value, Mapping):
        options = tuple(
            f"{_text(key)}. {_text(value)}"
            for key, value in options_value.items()
            if _text(key) and _text(value)
        )
    else:
        options = _texts(options_value)
    source_id = _text(row.get("source_id"))
    if not question or not answer or not options or not source_id:
        return None
    return {
        "case_id": _text(row.get("case_candidate_id")),
        "title": _text(row.get("title")),
        "question": question,
        "options": options,
        "answer": answer,
        "original_source": _text(payload.get("original_source")),
        "source_id": source_id,
        "source_record_id": _text(row.get("source_record_id")),
        "license": _text(row.get("license")),
        "knowledge_points": _texts(row.get("knowledge_points")),
        "review_status": "approved_training_only",
        "clinical_evidence_allowed": False,
    }


def load_qzbc_grounding_assets(qzbc_root: str | Path) -> QZBCGroundingAssets:
    root = Path(qzbc_root)
    paths = {name: root / relative for name, relative in ASSET_PATHS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing qzbc grounding assets: " + ", ".join(missing))

    herbs = tuple(
        normalized
        for row in _items(_read_json(paths["herbs"]))
        if (normalized := _normalize_herb(row)) is not None
    )
    formulas = tuple(
        normalized
        for row in _items(_read_json(paths["formulas"]))
        if (normalized := _normalize_formula(row)) is not None
    )
    incompatibilities = tuple(
        normalized
        for row in _items(
            _read_json(paths["incompatibilities"]),
            key="constraints",
        )
        if (normalized := _normalize_incompatibility(row)) is not None
    )
    diet_precautions = tuple(
        normalized
        for row in _items(_read_json(paths["diet_precautions"]))
        if (normalized := _normalize_diet(row)) is not None
    )
    approved_training_items = tuple(
        normalized
        for row in _items(_read_json(paths["approved_training_items"]))
        if (normalized := _normalize_training_item(row)) is not None
    )
    counts = {
        "herbs": len(herbs),
        "formulas": len(formulas),
        "incompatibilities": len(incompatibilities),
        "diet_precautions": len(diet_precautions),
        "approved_training_items": len(approved_training_items),
    }
    manifest = {
        "schema_version": "qzbc-grounding-assets-v1",
        "files": {
            ASSET_PATHS[name]: {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "usable_counts": counts,
        "policy": {
            "reads_patient_database": False,
            "reads_chat_history": False,
            "reads_graphiti_or_mem0": False,
            "approved_training_items_are_nonclinical": True,
            "facts_retain_source_ids": True,
        },
    }
    assets = QZBCGroundingAssets(
        herbs=herbs,
        formulas=formulas,
        incompatibilities=incompatibilities,
        diet_precautions=diet_precautions,
        approved_training_items=approved_training_items,
        manifest=manifest,
    )
    assets.require_coverage()
    return assets


def partition_qzbc_grounding_assets(
    assets: QZBCGroundingAssets,
    *,
    seed: int,
    dev_ratio: float,
    test_ratio: float,
) -> tuple[Mapping[str, QZBCGroundingAssets], Mapping[str, Any]]:
    """Create deterministic, exact-asset-disjoint train/dev/test views."""

    if dev_ratio < 0 or test_ratio < 0 or dev_ratio + test_ratio >= 1:
        raise ValueError("invalid grounding asset split ratios")
    fields = (
        "herbs",
        "formulas",
        "incompatibilities",
        "diet_precautions",
        "approved_training_items",
    )
    partitions: Dict[str, Dict[str, Tuple[Mapping[str, Any], ...]]] = {
        split: {} for split in ("train", "dev", "test")
    }
    hashes: Dict[str, Dict[str, set[str]]] = {
        field: {split: set() for split in ("train", "dev", "test")}
        for field in fields
    }
    for field in fields:
        rows = list(getattr(assets, field))
        if len(rows) < 3 and dev_ratio > 0 and test_ratio > 0:
            raise ValueError(f"{field} needs at least three rows for split isolation")
        random.Random(f"qzbc-assets|{seed}|{field}").shuffle(rows)
        dev_count = max(1, round(len(rows) * dev_ratio)) if dev_ratio else 0
        test_count = max(1, round(len(rows) * test_ratio)) if test_ratio else 0
        while dev_count + test_count > len(rows) - 1:
            if dev_count > int(dev_ratio > 0):
                dev_count -= 1
            elif test_count > int(test_ratio > 0):
                test_count -= 1
            else:
                raise AssertionError(f"cannot preserve a train row for {field}")
        by_split = {
            "test": tuple(rows[:test_count]),
            "dev": tuple(rows[test_count : test_count + dev_count]),
            "train": tuple(rows[test_count + dev_count :]),
        }
        for split, split_rows in by_split.items():
            partitions[split][field] = split_rows
            hashes[field][split] = {
                hashlib.sha256(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                for row in split_rows
            }
    for field in fields:
        if (
            hashes[field]["train"] & hashes[field]["dev"]
            or hashes[field]["train"] & hashes[field]["test"]
            or hashes[field]["dev"] & hashes[field]["test"]
        ):
            raise AssertionError(f"grounding asset split overlap for {field}")

    views = {
        split: QZBCGroundingAssets(
            herbs=partitions[split]["herbs"],
            formulas=partitions[split]["formulas"],
            incompatibilities=partitions[split]["incompatibilities"],
            diet_precautions=partitions[split]["diet_precautions"],
            approved_training_items=partitions[split]["approved_training_items"],
            manifest=assets.manifest,
        )
        for split in ("train", "dev", "test")
    }
    active_splits = {"train"}
    if dev_ratio:
        active_splits.add("dev")
    if test_ratio:
        active_splits.add("test")
    for split in active_splits:
        views[split].require_coverage()
    split_manifest = {
        "strategy": "exact_asset_disjoint_deterministic_shuffle",
        "seed": seed,
        "counts": {
            split: {
                field: len(getattr(views[split], field)) for field in fields
            }
            for split in ("train", "dev", "test")
        },
        "overlap_check": "PASS",
    }
    return views, split_manifest
