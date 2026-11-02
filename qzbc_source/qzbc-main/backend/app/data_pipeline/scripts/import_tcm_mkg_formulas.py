"""Import structurally complete Chinese patent-medicine candidate records from TCM-MKG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR

SOURCE_ID = "SRC_TCM_MKG"
SOURCE_URL = "https://zenodo.org/records/15395588"
LICENSE = "CC-BY-NC-4.0"


def import_tcm_mkg(*, root: Path | str, content_dir: Path | str = DEFAULT_CONTENT_DATA_DIR) -> dict[str, int]:
    root = Path(root)
    medicines = _rows(root / "D2_Chinese_patent_medicine.tsv")
    terms = _rows(root / "D3_CPM_TCMT.tsv")
    components = _rows(root / "D4_CPM_CHP.tsv")
    herbs = {row["CHP_ID"]: row["Chinese_herbal_pieces"] for row in _rows(root / "D6_Chinese_herbal_pieces.tsv")}
    term_map: dict[str, list[dict]] = defaultdict(list)
    component_map: dict[str, list[dict]] = defaultdict(list)
    for row in terms:
        term_map[row["CPM_ID"]].append(row)
    for row in components:
        if herbs.get(row["CHP_ID"]):
            component_map[row["CPM_ID"]].append(row)

    content = Path(content_dir)
    formulas = [item for item in _items(content / "formulas.json") if SOURCE_ID not in item.get("source_ids", [])]
    imported = []
    for medicine in medicines:
        cpm_id = medicine["CPM_ID"]
        linked_terms = term_map[cpm_id]
        functions = _unique(row["Chinese_term"] for row in linked_terms if row["Chinese_group"] in {"治法", "治则"})
        syndromes = _unique(row["Chinese_term"] for row in linked_terms if row["Chinese_group"] == "传统医学证候")
        symptoms = _unique(row["Chinese_term"] for row in linked_terms if row["Chinese_group"] == "传统医学疾病")
        linked_components = component_map[cpm_id]
        if not functions or not (syndromes or symptoms) or not linked_components:
            continue
        name = medicine["Chinese_patent_medicine"].strip()
        imported.append({
            "entry_id": _id("FORMULA", SOURCE_ID, cpm_id),
            "name": name,
            "normalized_name": name,
            "components": [{"herb_name": herbs[row["CHP_ID"]], "normalized_name": herbs[row["CHP_ID"]], "dosage": None, "unit": None} for row in linked_components],
            "functions": functions,
            "symptoms": symptoms,
            "syndromes": syndromes,
            "source_ids": [SOURCE_ID],
            "source_records": [{"source_id": SOURCE_ID, "source_record_id": cpm_id, "source_file": "D2/D3/D4/D6.tsv", "commit_sha": "zenodo-15395588-v2", "license": LICENSE}],
            "keywords": _unique([name, *functions, *symptoms, *syndromes]),
            "visibility": ["doctor", "student"],
            "evidence_level": "open_dataset_noncommercial_structured_candidate",
            "review_status": "pending_doctor_clinical_review",
            "usage_note": "仅用于医生端辅助检索候选；须经辨证、审方与医生确认，不含剂量建议。",
            "license_note": "CC BY-NC 4.0；仅非商业使用。已完成字段与溯源完整性校验，尚待医生临床复核。",
        })
    formulas.extend(imported)
    sources = [item for item in _items(content / "sources.json") if item.get("source_id") != SOURCE_ID]
    sources.append({"source_id": SOURCE_ID, "source_title": "TCM-MKG", "source_type": "zenodo_dataset", "source_version": "v2", "source_year": "2025", "evidence_level": "open_dataset_noncommercial_structured_candidate", "license": LICENSE, "license_note": "CC BY-NC 4.0；仅非商业。用于医生候选检索，须临床复核。", "repository_url": SOURCE_URL, "processing_script": "import_tcm_mkg_formulas.py", "processing_version": "1.0.0"})
    _write(content / "formulas.json", {"items": sorted(formulas, key=lambda item: item["name"])})
    _write(content / "sources.json", {"items": sorted(sources, key=lambda item: item["source_id"])})
    return {"imported_complete_formula_candidates": len(imported), "total_formulas": len(formulas)}


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("items", [])


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _unique(values) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]
    return f"{prefix}_{''.join(chr(ord('A') + int(char, 16)) for char in digest)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--content-dir", default=str(DEFAULT_CONTENT_DATA_DIR))
    args = parser.parse_args()
    print(json.dumps(import_tcm_mkg(root=args.root, content_dir=args.content_dir), ensure_ascii=False))
