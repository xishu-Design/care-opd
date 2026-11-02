"""Project source-backed formula indications onto component herbs as indirect relations.

The generated fields never claim that a herb directly treats a syndrome.
They preserve the auditable path: syndrome/symptom -> formula -> component herb.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR


FORMULA_SOURCE_ID = "SRC_TCM_MKG"
MAX_RELATED_FORMULAS = 40


def enrich_herb_formula_relations(content_dir: Path | str = DEFAULT_CONTENT_DATA_DIR) -> dict[str, int]:
    content = Path(content_dir)
    herbs = _items(content / "herbs.json")
    formulas = _items(content / "formulas.json")
    relations_by_herb: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for formula in formulas:
        source_ids = [str(value) for value in formula.get("source_ids", []) if value]
        if FORMULA_SOURCE_ID not in source_ids:
            continue
        syndromes = _strings(formula.get("syndromes"))
        symptoms = _strings(formula.get("symptoms"))
        functions = _strings(formula.get("display_functions")) or _strings(formula.get("functions"))
        if not syndromes and not symptoms:
            continue
        relation = {
            "formula_name": str(formula.get("display_name") or formula.get("name") or "").strip(),
            "syndromes": syndromes,
            "symptoms": symptoms,
            "functions": functions,
            "source_ids": source_ids,
            "source_records": formula.get("source_records") if isinstance(formula.get("source_records"), list) else [],
            "relation_type": "FORMULA_COMPONENT_INDIRECT_ASSOCIATION",
        }
        if not relation["formula_name"]:
            continue
        for component in formula.get("components") or []:
            if not isinstance(component, dict):
                continue
            herb_name = str(component.get("normalized_name") or component.get("herb_name") or "").strip()
            if herb_name:
                relations_by_herb[herb_name].append(relation)

    enriched_count = 0
    relation_count = 0
    syndrome_count = 0
    symptom_count = 0
    for herb in herbs:
        herb_name = str(herb.get("display_name") or herb.get("normalized_name") or herb.get("name") or "").strip()
        relations = _dedupe_relations(relations_by_herb.get(herb_name, []))[:MAX_RELATED_FORMULAS]
        for key in (
            "indirect_syndromes",
            "indirect_symptoms",
            "indirect_functions",
            "related_formulas",
            "relation_source_ids",
            "formula_relation_note",
        ):
            herb.pop(key, None)
        if not relations:
            continue

        indirect_syndromes = _unique(value for relation in relations for value in relation["syndromes"])
        indirect_symptoms = _unique(value for relation in relations for value in relation["symptoms"])
        indirect_functions = _unique(value for relation in relations for value in relation["functions"])
        relation_source_ids = _unique(value for relation in relations for value in relation["source_ids"])
        herb["indirect_syndromes"] = indirect_syndromes
        herb["indirect_symptoms"] = indirect_symptoms
        herb["indirect_functions"] = indirect_functions
        herb["related_formulas"] = relations
        herb["relation_source_ids"] = relation_source_ids
        herb["formula_relation_note"] = "经来源方剂的“证候/疾病术语→方剂→组成药材”路径间接关联，不表示该药材可单独治疗对应证候。"
        herb["keywords"] = _unique([
            *_strings(herb.get("keywords")),
            *indirect_syndromes,
            *indirect_symptoms,
            *indirect_functions,
            *(relation["formula_name"] for relation in relations),
        ])
        enriched_count += 1
        relation_count += len(relations)
        syndrome_count += len(indirect_syndromes)
        symptom_count += len(indirect_symptoms)

    _write(content / "herbs.json", {"items": herbs})
    return {
        "total_herbs": len(herbs),
        "herbs_with_formula_relations": enriched_count,
        "formula_component_relation_paths": relation_count,
        "indirect_syndrome_terms": syndrome_count,
        "indirect_symptom_terms": symptom_count,
    }


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for relation in relations:
        key = (
            relation["formula_name"],
            tuple(relation["syndromes"]),
            tuple(relation["symptoms"]),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return sorted(result, key=lambda item: item["formula_name"])


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _items(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("items", [])


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-dir", default=str(DEFAULT_CONTENT_DATA_DIR))
    args = parser.parse_args()
    print(json.dumps(enrich_herb_formula_relations(args.content_dir), ensure_ascii=False))
