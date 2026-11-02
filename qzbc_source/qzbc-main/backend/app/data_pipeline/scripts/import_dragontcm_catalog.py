"""Import non-commercial DragonTCM catalog data with explicit provenance.

The source is CC BY-NC 4.0. This script deliberately excludes dosage, disease
claims and treatment recommendations; it only imports catalog fields needed for
the doctor/student herb and formula directories and their composition graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
from pypinyin import lazy_pinyin

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR
from app.data_pipeline.validators.source_registry_validator import DATA_PIPELINE_DIR


SOURCE_ID = "SRC_DRAGON_TCM"
SOURCE_TITLE = "DragonTCM"
SOURCE_URL = "https://huggingface.co/datasets/f-galkin/DragonTCM"
SOURCE_LICENSE = "CC-BY-NC-4.0"
SOURCE_REVISION = "5b535a4"
RAW_DIR = DATA_PIPELINE_DIR / "raw" / SOURCE_ID


def import_dragontcm_catalog(
    *,
    herbs_path: Path | str = RAW_DIR / "herbs.parquet",
    formulas_path: Path | str = RAW_DIR / "formulas.parquet",
    content_dir: Path | str = DEFAULT_CONTENT_DATA_DIR,
) -> dict[str, int]:
    herb_rows = parquet.read_table(herbs_path).to_pylist()
    formula_rows = parquet.read_table(formulas_path).to_pylist()
    root = Path(content_dir)
    # Make repeated imports deterministic: discard the previous Dragon-only
    # rows and remove its provenance from rows that were enriched in place.
    # The current source revision is then applied once below.
    herbs = [_without_dragon_provenance(item) for item in _items(root / "herbs.json") if SOURCE_ID not in item.get("source_ids", []) or len(item.get("source_ids", [])) > 1]
    formulas = [item for item in _items(root / "formulas.json") if SOURCE_ID not in item.get("source_ids", [])]

    source_by_pinyin = {_pinyin_key(row.get("name", "")): row for row in herb_rows if row.get("name")}
    display_by_pinyin = {key: _display_name(row) for key, row in source_by_pinyin.items()}
    existing_by_name = {str(item.get("normalized_name") or item.get("name")): item for item in herbs}

    enriched = 0
    for item in herbs:
        row = source_by_pinyin.get(_pinyin_key("".join(lazy_pinyin(str(item.get("name", ""))))))
        if row is None:
            continue
        _merge_herb(item, row, int(herb_rows.index(row)) + 1)
        enriched += 1

    for index, row in enumerate(herb_rows, start=1):
        name = _display_name(row)
        normalized = str(name).replace(" ", "")
        if normalized in existing_by_name:
            continue
        item = {
            "entry_id": _id("HERB", SOURCE_ID, str(index), row.get("name", "")),
            "name": name,
            "normalized_name": normalized,
            "aliases": _aliases(row),
            "source_ids": [SOURCE_ID],
            "source_records": [_source_record(index)],
            "functions": _actions(row),
            "cautions": _strings(row.get("contraindications")),
            "properties": _properties(row),
            "keywords": _unique([name, str(row.get("name", "")), *_actions(row)]),
            "visibility": ["doctor", "student"],
            "evidence_level": "open_dataset_noncommercial_catalog",
            "license_note": "CC BY-NC 4.0；仅非商业教学/演示。功效为来源原文，不构成诊疗、处方或剂量建议。",
        }
        herbs.append(item)
        existing_by_name[normalized] = item

    formula_items = []
    for index, row in enumerate(formula_rows, start=1):
        name = _display_name(row)
        components = _components(row, display_by_pinyin)
        formula_items.append(
            {
                "entry_id": _id("FORMULA", SOURCE_ID, str(index), row.get("name", "")),
                "name": name,
                "normalized_name": name.replace(" ", ""),
                "components": components,
                "composition_note": "来源未提供组成记录" if not components else "",
                "source_ids": [SOURCE_ID],
                "source_records": [_source_record(index)],
                "functions": _strings(row.get("actions")),
                "cautions": _strings(row.get("contraindications")),
                "usage_note": "CC BY-NC 4.0 非商业教学/演示目录记录；不作为临床处方、用药或剂量建议。",
                "keywords": _unique([name, str(row.get("name", "")), *[item["herb_name"] for item in components]]),
                "visibility": ["doctor", "student"],
                "evidence_level": "open_dataset_noncommercial_catalog",
                "license_note": "CC BY-NC 4.0；仅非商业教学/演示。",
            }
        )

    formula_ids = {str(item.get("entry_id")) for item in formula_items}
    formulas = [item for item in formulas if str(item.get("entry_id")) not in formula_ids] + formula_items
    sources = _items(root / "sources.json")
    sources = [item for item in sources if item.get("source_id") != SOURCE_ID] + [_source()]
    relationships = _relationships(herbs, formulas)

    _write(root / "herbs.json", {"items": sorted(herbs, key=lambda item: str(item.get("name", "")))})
    _write(root / "formulas.json", {"items": sorted(formulas, key=lambda item: str(item.get("name", "")))})
    _write(root / "sources.json", {"items": sorted(sources, key=lambda item: str(item.get("source_id", "")))})
    _write(root / "relationships.json", {"items": relationships})
    return {"herbs": len(herbs), "enriched_existing_herbs": enriched, "formulas": len(formulas), "dragon_formulas": len(formula_items), "relationships": len(relationships)}


def _merge_herb(item: dict[str, Any], row: dict[str, Any], record: int) -> None:
    item["aliases"] = _unique([*item.get("aliases", []), *_aliases(row)])
    item["source_ids"] = _unique([*item.get("source_ids", []), SOURCE_ID])
    item["source_records"] = [*item.get("source_records", []), _source_record(record)]
    if not item.get("functions"):
        item["functions"] = _actions(row)
    if not item.get("cautions"):
        item["cautions"] = _strings(row.get("contraindications"))
    item["properties"] = item.get("properties") or _properties(row)
    item["keywords"] = _unique([*item.get("keywords", []), *_actions(row), str(row.get("name", ""))])
    item["license_note"] = "包含 CC BY-NC 4.0 DragonTCM 来源字段；仅非商业教学/演示。功效为来源原文，不构成诊疗、处方或剂量建议。"


def _without_dragon_provenance(item: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(item)
    cleaned["source_ids"] = [source_id for source_id in item.get("source_ids", []) if source_id != SOURCE_ID]
    cleaned["source_records"] = [
        record for record in item.get("source_records", []) if record.get("source_id") != SOURCE_ID
    ]
    return cleaned


def _relationships(herbs: list[dict[str, Any]], formulas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    herb_by_name = {str(item.get("normalized_name") or item.get("name", "")): item for item in herbs}
    relations: dict[str, dict[str, Any]] = {}
    for herb in herbs:
        for source_id in herb.get("source_ids", []):
            _add_relation(relations, str(herb["entry_id"]), str(source_id), "DERIVED_FROM", herb)
    for formula in formulas:
        for component in formula.get("components", []):
            herb = herb_by_name.get(str(component.get("normalized_name") or component.get("herb_name", "")))
            if herb:
                _add_relation(relations, str(formula["entry_id"]), str(herb["entry_id"]), "CONTAINS", formula)
        for source_id in formula.get("source_ids", []):
            _add_relation(relations, str(formula["entry_id"]), str(source_id), "DERIVED_FROM", formula)
    return [relations[key] for key in sorted(relations)]


def _add_relation(target: dict[str, dict[str, Any]], source: str, destination: str, relation_type: str, item: dict[str, Any]) -> None:
    relation_id = _id("REL", source, destination, relation_type)
    source_ids = _unique([str(value) for value in item.get("source_ids", []) if value])
    target[relation_id] = {
        "relation_id": relation_id,
        "source_entity_id": source,
        "target_entity_id": destination,
        "relation_type": relation_type,
        "source_ids": source_ids,
        "source_record_ids": [str(record.get("source_record_id", "")) for record in item.get("source_records", [])],
        "evidence_level": item.get("evidence_level", "open_dataset_noncommercial_catalog"),
        "visibility": item.get("visibility", ["doctor", "student"]),
    }


def _components(row: dict[str, Any], display_by_pinyin: dict[str, str]) -> list[dict[str, Any]]:
    try:
        composition = json.loads(str(row.get("composition") or "{}"))
    except json.JSONDecodeError:
        return []
    if not isinstance(composition, dict):
        return []
    result = []
    for pinyin_name in composition:
        display = display_by_pinyin.get(_pinyin_key(pinyin_name), str(pinyin_name).title())
        result.append({"herb_name": display, "normalized_name": display.replace(" ", ""), "dosage": None, "unit": None})
    return result


def _display_name(row: dict[str, Any]) -> str:
    chinese = next((value for value in _strings(row.get("synonyms")) if re.search(r"[\u3400-\u9fff]", value)), "")
    return chinese or str(row.get("name", "")).title()


def _aliases(row: dict[str, Any]) -> list[str]:
    return _unique([str(row.get("name", "")), *_strings(row.get("synonyms"))])


def _actions(row: dict[str, Any]) -> list[str]:
    try:
        entries = json.loads(str(row.get("actions") or "[]"))
    except json.JSONDecodeError:
        return []
    return _unique([str(item.get("action", "")) for item in entries if isinstance(item, dict) and item.get("action")])


def _properties(row: dict[str, Any]) -> list[str]:
    try:
        properties = json.loads(str(row.get("properties") or "{}"))
    except json.JSONDecodeError:
        return []
    values = []
    for section in ("taste", "temperature", "meridians"):
        value = properties.get(section, {}) if isinstance(properties, dict) else {}
        values.extend(value.get("primary", []) if isinstance(value, dict) else [])
    return _unique([str(value) for value in values if value])


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed if item] if isinstance(parsed, list) else []


def _source() -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_title": SOURCE_TITLE,
        "source_type": "huggingface_dataset",
        "source_version": SOURCE_REVISION,
        "source_year": "2024",
        "evidence_level": "open_dataset_noncommercial_catalog",
        "license": SOURCE_LICENSE,
        "license_note": "CC BY-NC 4.0；仅非商业教学/演示。目录与图谱数据不构成临床证据、患者建议、处方或剂量建议。",
        "repository_url": SOURCE_URL,
        "processing_script": "import_dragontcm_catalog.py",
        "processing_version": "1.0.0",
    }


def _source_record(index: int) -> dict[str, str]:
    return {"source_id": SOURCE_ID, "source_record_id": f"row_{index}", "source_file": "parquet", "commit_sha": SOURCE_REVISION, "license": SOURCE_LICENSE}


def _pinyin_key(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def _id(prefix: str, *parts: str) -> str:
    # Alphabet-only stable IDs cannot be mistaken for a phone or ID-card number
    # by the de-identification validator.
    digest = hashlib.sha256(chr(31).join(parts).encode()).hexdigest()[:20]
    return f"{prefix}_{''.join(chr(ord('A') + int(char, 16)) for char in digest)}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _items(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("items", [])


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import CC BY-NC DragonTCM catalog data for non-commercial use.")
    parser.add_argument("--herbs", default=str(RAW_DIR / "herbs.parquet"))
    parser.add_argument("--formulas", default=str(RAW_DIR / "formulas.parquet"))
    parser.add_argument("--content-dir", default=str(DEFAULT_CONTENT_DATA_DIR))
    args = parser.parse_args()
    print(json.dumps(import_dragontcm_catalog(herbs_path=args.herbs, formulas_path=args.formulas, content_dir=args.content_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
