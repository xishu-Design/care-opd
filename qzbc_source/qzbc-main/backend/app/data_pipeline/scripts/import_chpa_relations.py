from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR


SOURCE_ID = "SRC_DRYAD_TCM_NETWORK"
SOURCE_TITLE = "中医异构网络数据集（CHPA 草药关系）"
SOURCE_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwx9"
SOURCE_VERSION = "10.5061/dryad.wh70rxwx9"
PROCESSING_VERSION = "1.0.0"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "raw" / SOURCE_ID
DEFAULT_TRANSLATIONS = Path(__file__).resolve().parents[1] / "processed" / "chpa_term_translations_zh.json"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "chpa_import_report.json"
HAN_RE = re.compile(r"[\u3400-\u9fff]")
SMHB_ID_RE = re.compile(r"(\d+)$")
TERM_TRANSLATION_OVERRIDES = {
    "efficacy": {
        "Gyozui": "利水",
        "Liyan": "利咽",
    },
    "symptom": {
        "Dexol": "药物名称（非症状）",
        "Olfactory Uncinate Fit": "嗅觉性钩回发作",
        "Sarcoptic Red Mange": "疥螨性红疹",
        "Vocal Tic": "发声性抽动",
    },
}


def import_chpa_relations(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    content_dir: Path | str = DEFAULT_CONTENT_DATA_DIR,
    translations_path: Path | str = DEFAULT_TRANSLATIONS,
    report_path: Path | str = DEFAULT_REPORT,
) -> dict[str, Any]:
    raw_root = Path(raw_dir)
    content_root = Path(content_dir)
    translations_file = Path(translations_path)
    data_root = raw_root / "Supplementary files" / "DATA"

    herbs_document = _read_json(content_root / "herbs.json", {"items": []})
    herbs = herbs_document.get("items", [])
    if not isinstance(herbs, list):
        raise ValueError("herbs.json items must be a list")
    for herb in herbs:
        previous_terms = {
            *herb.get("chpa_efficacies", []),
            *herb.get("chpa_symptoms", []),
        }
        if previous_terms and isinstance(herb.get("keywords"), list):
            herb["keywords"] = [value for value in herb["keywords"] if value not in previous_terms]
        for key in (
            "chpa_efficacies",
            "chpa_symptoms",
            "chpa_raw_efficacies",
            "chpa_raw_symptoms",
            "chpa_relation_note",
            "chpa_translation_review_status",
            "chpa_relation_evidence_level",
        ):
            herb.pop(key, None)

    translations = _read_json(translations_file, {})
    efficacy_translations = _translation_map(translations.get("efficacy"))
    symptom_translations = _translation_map(translations.get("symptom"))

    symmap_rows = _xlsx_dict_rows(raw_root / "SymMap v2.0, SMHB file.xlsx")
    symmap_by_id = {
        str(int(str(row["Herb_id"]).strip())): row
        for row in symmap_rows
        if str(row.get("Herb_id", "")).strip().isdigit() and str(row.get("Chinese_name", "")).strip()
    }
    herb_by_name = _herb_name_index(herbs)
    smhb_to_herb: dict[str, dict[str, Any]] = {}
    added_herbs = 0
    for smhb_id, row in symmap_by_id.items():
        chinese_name = str(row.get("Chinese_name", "")).strip()
        herb = herb_by_name.get(chinese_name)
        if herb is None:
            herb = _new_symmap_herb(smhb_id, row)
            herbs.append(herb)
            herb_by_name[chinese_name] = herb
            added_herbs += 1
        _append_unique(herb, "aliases", _symmap_aliases(row))
        smhb_to_herb[smhb_id] = herb

    efficacy_id_map, efficacy_mapping_report = _efficacy_herb_map(
        efficacy_path=data_root / "herb efficiacy.csv",
        herb_target_path=data_root / "Herb target.csv",
        herbs=herbs,
        symmap_rows=symmap_rows,
    )

    efficacy_pairs = _efficacy_pairs(data_root / "herb efficiacy.csv")
    imported_efficacy_pairs = 0
    untranslated_efficacy_pairs = 0
    for raw_herb_id, raw_term in efficacy_pairs:
        herb = efficacy_id_map.get(raw_herb_id)
        translated = efficacy_translations.get(raw_term, "")
        if herb is None or not _is_importable_translation(translated):
            if herb is not None:
                untranslated_efficacy_pairs += 1
            continue
        _append_unique(herb, "chpa_efficacies", [translated])
        _append_unique(herb, "chpa_raw_efficacies", [raw_term])
        _mark_chpa_herb(herb)
        imported_efficacy_pairs += 1

    symptom_rows, symptom_raw_stats = _significant_symptom_rows(data_root / "herb-symptom.csv")
    imported_symptom_pairs = 0
    untranslated_symptom_pairs = 0
    unmapped_symptom_herb_pairs = 0
    for row in symptom_rows:
        smhb_id = _normalize_smhb_id(row["Herb_id"])
        herb = smhb_to_herb.get(smhb_id)
        if herb is None:
            unmapped_symptom_herb_pairs += 1
            continue
        raw_term = row["symptom"]
        translated = symptom_translations.get(raw_term, "")
        if not _is_importable_translation(translated):
            untranslated_symptom_pairs += 1
            continue
        _append_unique(herb, "chpa_symptoms", [translated])
        _append_unique(herb, "chpa_raw_symptoms", [raw_term])
        _mark_chpa_herb(herb)
        imported_symptom_pairs += 1

    for herb in herbs:
        if herb.get("chpa_efficacies") or herb.get("chpa_symptoms"):
            herb["chpa_relation_note"] = (
                "来自 CHPA/Dryad 的草药—功效或草药—症状统计关联；"
                "英文术语经本地模型辅助转为中文，尚需中医药专业人员逐条复核，不能直接作为临床处方依据。"
            )
            herb["chpa_translation_review_status"] = "machine_translated_pending_medical_review"
            herb["chpa_relation_evidence_level"] = "statistical_association"
            _append_unique(
                herb,
                "source_records",
                [
                    {
                        "source_id": SOURCE_ID,
                        "source_record_id": str(herb.get("entry_id", "")),
                        "source_file": "herb efficiacy.csv; herb-symptom.csv",
                    }
                ],
                identity=lambda item: (
                    str(item.get("source_id", "")),
                    str(item.get("source_record_id", "")),
                    str(item.get("source_file", "")),
                ),
            )
            _append_unique(
                herb,
                "keywords",
                [*herb.get("chpa_efficacies", []), *herb.get("chpa_symptoms", [])],
            )

    herbs_document["items"] = sorted(
        herbs,
        key=lambda item: (
            str(item.get("display_name") or item.get("name") or ""),
            str(item.get("entry_id") or ""),
        ),
    )
    _write_json(content_root / "herbs.json", herbs_document)
    _upsert_content_source(content_root)

    report = {
        "source_id": SOURCE_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_files": {
            "efficacy": _file_info(data_root / "herb efficiacy.csv"),
            "symptom": _file_info(data_root / "herb-symptom.csv"),
            "symmap_herb_mapping": _file_info(raw_root / "SymMap v2.0, SMHB file.xlsx"),
        },
        "published_counts": {
            "efficacy_herbs": 829,
            "efficacy_terms": 373,
            "efficacy_links": 3830,
            "symptom_herbs": 465,
            "symptom_terms": 1027,
            "symptom_links": 16739,
        },
        "released_file_counts": {
            "efficacy_unique_pairs": len(efficacy_pairs),
            "efficacy_unique_herb_identifiers": len({pair[0] for pair in efficacy_pairs}),
            "efficacy_unique_terms": len({pair[1] for pair in efficacy_pairs}),
            **symptom_raw_stats,
        },
        "imported_counts": {
            "new_catalog_herbs": added_herbs,
            "mapped_symmap_herbs": len(smhb_to_herb),
            "efficacy_pairs": imported_efficacy_pairs,
            "symptom_pairs_p_lt_0_05": imported_symptom_pairs,
            "herbs_with_chpa_efficacy": sum(bool(item.get("chpa_efficacies")) for item in herbs),
            "herbs_with_chpa_symptoms": sum(bool(item.get("chpa_symptoms")) for item in herbs),
            "catalog_unique_translated_efficacy_relations": sum(len(item.get("chpa_efficacies") or []) for item in herbs),
            "catalog_unique_translated_symptom_relations": sum(len(item.get("chpa_symptoms") or []) for item in herbs),
        },
        "not_imported": {
            "efficacy_pairs_missing_chinese_translation": untranslated_efficacy_pairs,
            "symptom_pairs_missing_chinese_translation": untranslated_symptom_pairs,
            "symptom_pairs_missing_herb_mapping": unmapped_symptom_herb_pairs,
            **efficacy_mapping_report,
        },
        "quality_notes": [
            "公开补充 CSV 的实际条数与论文/数据仓库摘要中的统计数字不一致；系统记录两组数字但只导入实际文件内容。",
            "草药—症状关系仅导入具有数值 P_value 且 P<0.05 的去重关系。",
            "CHPA 关系属于统计关联，中文术语为本地模型辅助翻译，默认标记为待医学复核。",
        ],
    }
    _write_json(Path(report_path), report)
    return report


def translate_chpa_terms(
    *,
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    output_path: Path | str = DEFAULT_TRANSLATIONS,
    ollama_url: str = "http://127.0.0.1:11434",
    model: str = "qwen2.5:7b",
    batch_size: int = 24,
) -> dict[str, Any]:
    raw_root = Path(raw_dir)
    data_root = raw_root / "Supplementary files" / "DATA"
    efficacy_terms = sorted({term for _, term in _efficacy_pairs(data_root / "herb efficiacy.csv")})
    symptom_rows, _ = _significant_symptom_rows(data_root / "herb-symptom.csv")
    symptom_terms = sorted({row["symptom"] for row in symptom_rows})
    output = _read_json(Path(output_path), {})
    output.setdefault("efficacy", {})
    output.setdefault("symptom", {})
    output["efficacy"].update(TERM_TRANSLATION_OVERRIDES["efficacy"])
    output["symptom"].update(TERM_TRANSLATION_OVERRIDES["symptom"])
    output["metadata"] = {
        "model": model,
        "ollama_url": ollama_url,
        "generated_at": datetime.now(UTC).isoformat(),
        "review_status": "machine_translated_pending_medical_review",
    }
    for kind, terms, instruction in (
        ("efficacy", efficacy_terms, "中药或中医功效术语，使用规范、简洁的中医药中文表达"),
        ("symptom", symptom_terms, "现代医学症状或体征术语，使用规范、简洁的简体中文医学表达"),
    ):
        pending = [term for term in terms if not _is_chinese_term(str(output[kind].get(term, "")))]
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            translated = _ollama_translate_batch(
                batch,
                instruction=instruction,
                ollama_url=ollama_url,
                model=model,
            )
            output[kind].update(translated)
            _write_json(Path(output_path), output)
            print(f"{kind}: {min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)
    output["metadata"]["efficacy_terms"] = len(efficacy_terms)
    output["metadata"]["symptom_terms"] = len(symptom_terms)
    output["efficacy"].update(TERM_TRANSLATION_OVERRIDES["efficacy"])
    output["symptom"].update(TERM_TRANSLATION_OVERRIDES["symptom"])
    _write_json(Path(output_path), output)
    return output


def _ollama_translate_batch(
    terms: list[str],
    *,
    instruction: str,
    ollama_url: str,
    model: str,
) -> dict[str, str]:
    indexed = [{"id": index, "term": term} for index, term in enumerate(terms)]
    prompt = (
        f"把下列英文术语翻译成{instruction}。"
        "不要扩写、不要诊断、不要加入用药建议。"
        "必须为每个 id 返回一个结果，只输出 JSON 对象，格式为"
        '{"items":[{"id":0,"zh":"中文"}]}。'
        f"\n术语：{json.dumps(indexed, ensure_ascii=False)}"
    )
    request = urllib.request.Request(
        ollama_url.rstrip("/") + "/api/generate",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": 4096},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        envelope = json.loads(response.read().decode())
    payload = json.loads(envelope["response"])
    result: dict[str, str] = {}
    for item in payload.get("items", []):
        raw_id = item.get("id")
        if not str(raw_id).isdigit():
            continue
        index = int(raw_id)
        zh = str(item.get("zh", "")).strip()
        if 0 <= index < len(terms) and _is_chinese_term(zh):
            result[terms[index]] = zh
    if len(result) != len(terms):
        missing = [term for term in terms if term not in result]
        if len(terms) == 1:
            raise ValueError(f"Ollama did not return a Chinese translation for {missing[0]}")
        midpoint = len(terms) // 2
        return {
            **_ollama_translate_batch(
                terms[:midpoint],
                instruction=instruction,
                ollama_url=ollama_url,
                model=model,
            ),
            **_ollama_translate_batch(
                terms[midpoint:],
                instruction=instruction,
                ollama_url=ollama_url,
                model=model,
            ),
        }
    return result


def _efficacy_herb_map(
    *,
    efficacy_path: Path,
    herb_target_path: Path,
    herbs: list[dict[str, Any]],
    symmap_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    try:
        from pypinyin import lazy_pinyin
    except ImportError as exc:  # pragma: no cover - environment dependency guard
        raise RuntimeError("pypinyin is required for CHPA herb-name normalization") from exc

    name_to_herb = _herb_name_index(herbs)
    pinyin_to_herbs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for herb in herbs:
        names = [
            herb.get("display_name"),
            herb.get("name"),
            herb.get("normalized_name"),
            *(herb.get("aliases") or []),
        ]
        for value in names:
            text = str(value or "").strip()
            if not text:
                continue
            key = _latin_key("".join(lazy_pinyin(text))) if HAN_RE.search(text) else _latin_key(text)
            if key and herb not in pinyin_to_herbs[key]:
                pinyin_to_herbs[key].append(herb)
    for row in symmap_rows:
        herb = name_to_herb.get(str(row.get("Chinese_name", "")).strip())
        if herb is None:
            continue
        for value in (row.get("Pinyin_name"), row.get("English_name"), row.get("Latin_name")):
            key = _latin_key(value)
            if key and herb not in pinyin_to_herbs[key]:
                pinyin_to_herbs[key].append(herb)

    h_to_pinyin: dict[str, set[str]] = defaultdict(set)
    with herb_target_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            herb_id = str(row.get("Herb ID", "")).strip()
            pinyin = str(row.get("Chinese Pin Yin", "")).strip()
            if herb_id and pinyin:
                h_to_pinyin[herb_id].add(pinyin)

    raw_ids = sorted({herb_id for herb_id, _ in _efficacy_pairs(efficacy_path)})
    result: dict[str, dict[str, Any]] = {}
    ambiguous = 0
    missing = 0
    for raw_id in raw_ids:
        candidates: list[dict[str, Any]] = []
        for value in {raw_id, *h_to_pinyin.get(raw_id, set())}:
            for herb in pinyin_to_herbs.get(_latin_key(value), []):
                if herb not in candidates:
                    candidates.append(herb)
        if len(candidates) == 1:
            result[raw_id] = candidates[0]
        elif candidates:
            ambiguous += 1
        else:
            missing += 1
    return result, {
        "efficacy_herb_identifiers_mapped": len(result),
        "efficacy_herb_identifiers_ambiguous": ambiguous,
        "efficacy_herb_identifiers_unmapped": missing,
    }


def _efficacy_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            herb_id = str(row.get("herb ", "")).strip()
            efficacy = str(row.get("efficacy", "")).strip()
            if herb_id and efficacy:
                pairs.add((herb_id, efficacy))
    return sorted(pairs)


def _significant_symptom_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    raw_rows = 0
    raw_pairs: set[tuple[str, str]] = set()
    significant_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    all_herbs: set[str] = set()
    all_symptoms: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            raw_rows += 1
            symptom = str(raw.get("symptom ", "")).strip()
            herb_id = str(raw.get("Herb_id", "")).strip()
            if not symptom or not herb_id:
                continue
            all_herbs.add(herb_id)
            all_symptoms.add(symptom)
            pair = (symptom, herb_id)
            raw_pairs.add(pair)
            try:
                p_value = float(str(raw.get("P_value", "")).strip())
            except ValueError:
                continue
            if p_value >= 0.05:
                continue
            current = significant_by_pair.get(pair)
            if current is None or p_value < float(current["p_value"]):
                significant_by_pair[pair] = {
                    "symptom": symptom,
                    "Herb_id": herb_id,
                    "p_value": str(p_value),
                }
    rows = sorted(significant_by_pair.values(), key=lambda row: (row["Herb_id"], row["symptom"]))
    return rows, {
        "symptom_raw_rows": raw_rows,
        "symptom_unique_pairs": len(raw_pairs),
        "symptom_unique_herb_identifiers": len(all_herbs),
        "symptom_unique_terms": len(all_symptoms),
        "symptom_p_lt_0_05_unique_pairs": len(rows),
        "symptom_p_lt_0_05_unique_herbs": len({row["Herb_id"] for row in rows}),
        "symptom_p_lt_0_05_unique_terms": len({row["symptom"] for row in rows}),
    }


def _xlsx_dict_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
                for item in root.findall("a:si", namespace)
            ]
        workbook = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in workbook.findall(".//a:row", namespace):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", namespace):
                column = _xlsx_column_index(str(cell.get("r", "A1")))
                value_node = cell.find("a:v", namespace)
                value = "" if value_node is None else str(value_node.text or "")
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                values[column] = value
            if values:
                rows.append([values.get(index, "") for index in range(max(values) + 1)])
    if not rows:
        return []
    headers = rows[0]
    return [
        {header: row[index] if index < len(row) else "" for index, header in enumerate(headers) if header}
        for row in rows[1:]
    ]


def _xlsx_column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    result = 0
    for letter in letters.group(0) if letters else "A":
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def _new_symmap_herb(smhb_id: str, row: dict[str, str]) -> dict[str, Any]:
    chinese_name = str(row.get("Chinese_name", "")).strip()
    digest = hashlib.sha256(f"symmap:{smhb_id}:{chinese_name}".encode()).hexdigest()[:20].upper()
    properties = [
        value.strip()
        for value in re.split(r"[,，;；、]", str(row.get("Properties_Chinese", "")))
        if value.strip()
    ]
    return {
        "entry_id": f"HERB_CHPA_{digest}",
        "name": chinese_name,
        "display_name": chinese_name,
        "normalized_name": chinese_name,
        "original_name": chinese_name,
        "aliases": _symmap_aliases(row),
        "properties": properties,
        "functions": [],
        "cautions": [],
        "keywords": [chinese_name, *_symmap_aliases(row)],
        "source_ids": [SOURCE_ID],
        "source_records": [
            {
                "source_id": SOURCE_ID,
                "source_record_id": f"SMHB{int(smhb_id):05d}",
                "source_file": "SymMap v2.0, SMHB file.xlsx",
            }
        ],
        "visibility": ["doctor", "student"],
        "evidence_level": "open_dataset_statistical_association",
        "license_note": "CHPA/Dryad 关系数据为 CC0；中文药名映射参考 SymMap v2 下载文件。",
        "translation_note": "药材中文名来自 SymMap v2 草药标识映射。",
    }


def _symmap_aliases(row: dict[str, str]) -> list[str]:
    aliases = []
    for key in ("Pinyin_name", "Alias"):
        for value in re.split(r"[;；|]", str(row.get(key, ""))):
            value = value.strip()
            if value and value not in aliases:
                aliases.append(value)
    return aliases


def _herb_name_index(herbs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for herb in herbs:
        for value in (
            herb.get("display_name"),
            herb.get("name"),
            herb.get("normalized_name"),
            *(herb.get("aliases") or []),
        ):
            text = str(value or "").strip()
            if text and HAN_RE.search(text):
                result.setdefault(text, herb)
    return result


def _mark_chpa_herb(herb: dict[str, Any]) -> None:
    _append_unique(herb, "source_ids", [SOURCE_ID])
    _append_unique(herb, "relation_source_ids", [SOURCE_ID])
    visibility = herb.get("visibility")
    if not isinstance(visibility, list) or not visibility:
        herb["visibility"] = ["doctor", "student"]


def _upsert_content_source(content_root: Path) -> None:
    sources_path = content_root / "sources.json"
    sources = _read_json(sources_path, {"items": []})
    items = sources.setdefault("items", [])
    source_item = {
        "source_id": SOURCE_ID,
        "source_title": SOURCE_TITLE,
        "source_type": "dryad_dataset",
        "source_version": SOURCE_VERSION,
        "source_year": "2024",
        "evidence_level": "open_dataset_statistical_association",
        "license": "CC0",
        "license_note": "Dryad 发布的 CHPA 关系文件为 CC0；统计关联不等同于临床证据。",
        "repository_url": SOURCE_URL,
        "processing_script": "import_chpa_relations.py",
        "processing_version": PROCESSING_VERSION,
    }
    _replace_by_id(items, source_item, "source_id")
    _write_json(sources_path, sources)

    catalog_path = content_root / "source_catalog.json"
    catalog = _read_json(catalog_path, {"sources": []})
    catalog_items = catalog.setdefault("sources", [])
    catalog_item = {
        "source_id": SOURCE_ID,
        "name": SOURCE_TITLE,
        "repository_url": SOURCE_URL,
        "license": "CC0",
        "commit_hash": SOURCE_VERSION,
        "downloaded_at": "2026-07-16",
        "processed_at": datetime.now(UTC).isoformat(),
        "processing_script": "import_chpa_relations.py",
        "processing_version": PROCESSING_VERSION,
        "can_process": True,
        "clinical_evidence_allowed": False,
        "patient_visible": False,
        "doctor_evidence_allowed": True,
        "student_visible": True,
        "evaluation_allowed": False,
        "notes": "草药功效/症状统计关联；中文术语为模型辅助翻译并待医学复核，不可作为独立临床证据。",
    }
    _replace_by_id(catalog_items, catalog_item, "source_id")
    _write_json(catalog_path, catalog)


def _replace_by_id(items: list[dict[str, Any]], value: dict[str, Any], key: str) -> None:
    for index, item in enumerate(items):
        if item.get(key) == value[key]:
            items[index] = value
            return
    items.append(value)


def _append_unique(
    item: dict[str, Any],
    key: str,
    values: Iterable[Any],
    *,
    identity=lambda value: value,
) -> None:
    current = item.get(key)
    if not isinstance(current, list):
        current = []
        item[key] = current
    seen = {identity(value) for value in current}
    for value in values:
        marker = identity(value)
        if value not in (None, "") and marker not in seen:
            current.append(value)
            seen.add(marker)


def _translation_map(value: Any) -> dict[str, str]:
    return {str(key): str(translated).strip() for key, translated in value.items()} if isinstance(value, dict) else {}


def _normalize_smhb_id(value: str) -> str:
    match = SMHB_ID_RE.search(str(value).strip())
    return str(int(match.group(1))) if match else ""


def _latin_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _is_chinese_term(value: str) -> bool:
    return bool(value.strip() and HAN_RE.search(value))


def _is_importable_translation(value: str) -> bool:
    return _is_chinese_term(value) and "非症状" not in value and not re.search(r"[A-Za-z]{3,}", value)


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate and import CHPA herb-efficacy and herb-symptom relations.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--content-dir", type=Path, default=DEFAULT_CONTENT_DATA_DIR)
    parser.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.translate:
        translate_chpa_terms(
            raw_dir=args.raw_dir,
            output_path=args.translations,
            ollama_url=args.ollama_url,
            model=args.model,
            batch_size=args.batch_size,
        )
    report = import_chpa_relations(
        raw_dir=args.raw_dir,
        content_dir=args.content_dir,
        translations_path=args.translations,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
