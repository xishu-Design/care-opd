from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR


SOURCE_ID = "SRC_DB4403_T142_2021"
SOURCE_TITLE = "DB4403/T 142—2021 中药饮片处方审核规范"
SOURCE_URL = "https://amr.sz.gov.cn/attachment/0/755/755971/8581454.pdf"
SOURCE_SHA256 = "454ec4a03d1d2e506db7e50288485790ceb0e6be6be9913a31b0ac96e8bf8c72"
SOURCE_RECORD_ID = "DB4403/T 142-2021:6.3.1.4+Appendix-C:Table-C.1"
PROCESSING_VERSION = "1.0.0"
DEFAULT_RAW_PDF = Path(__file__).resolve().parents[1] / "raw" / SOURCE_ID / "DB4403_T142-2021.pdf"
DEFAULT_OUTPUT = DEFAULT_CONTENT_DATA_DIR / "herb_incompatibilities.json"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "tcm_incompatibility_import_report.json"


def _term(name: str, *, source_term: str | None = None, aliases: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "source_term": source_term or name,
        "aliases": aliases or [],
    }


def _constraint(
    constraint_id: str,
    category: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    relation_type = "TRADITIONAL_INCOMPATIBILITY_18" if category == "十八反" else "TRADITIONAL_INCOMPATIBILITY_19"
    return {
        "constraint_id": constraint_id,
        "category": category,
        "relation_type": relation_type,
        "left": left,
        "right": right,
        "risk_level": "red",
        "action": "pause_and_require_doctor_or_pharmacist_review",
        "source_id": SOURCE_ID,
        "source_record_id": SOURCE_RECORD_ID,
        "evidence_level": "public_local_standard_factual_relationship",
        "review_status": "manual_transcription_verified_against_pdf",
    }


CONSTRAINTS = [
    _constraint(
        "DB4403_C_18_01",
        "十八反",
        [_term("甘草")],
        [_term("甘遂"), _term("大戟"), _term("芫花"), _term("海藻")],
    ),
    _constraint(
        "DB4403_C_18_02",
        "十八反",
        [_term("乌头"), _term("川乌"), _term("草乌"), _term("附子")],
        [
            _term("贝母"),
            _term("川贝母"),
            _term("浙贝母"),
            _term("平贝母"),
            _term("伊贝母"),
            _term("湖北贝母"),
            _term("瓜蒌"),
            _term("瓜蒌子"),
            _term("瓜蒌皮"),
            _term("天花粉"),
            _term("半夏"),
            _term("白蔹"),
            _term("白及"),
        ],
    ),
    _constraint(
        "DB4403_C_18_03",
        "十八反",
        [_term("藜芦")],
        [
            _term("人参"),
            _term("丹参"),
            _term("沙参"),
            _term("玄参"),
            _term("苦参"),
            _term("细辛"),
            _term("芍药"),
            _term("白芍"),
            _term("赤芍"),
        ],
    ),
    _constraint("DB4403_C_19_01", "十九畏", [_term("硫黄", aliases=["硫磺"])], [_term("朴硝")]),
    _constraint("DB4403_C_19_02", "十九畏", [_term("水银")], [_term("砒霜")]),
    _constraint("DB4403_C_19_03", "十九畏", [_term("狼毒")], [_term("密陀僧")]),
    _constraint("DB4403_C_19_04", "十九畏", [_term("巴豆")], [_term("牵牛子", source_term="牵牛", aliases=["牵牛"])]),
    _constraint("DB4403_C_19_05", "十九畏", [_term("丁香")], [_term("郁金")]),
    _constraint("DB4403_C_19_06", "十九畏", [_term("川乌"), _term("草乌")], [_term("犀角")]),
    _constraint("DB4403_C_19_07", "十九畏", [_term("牙硝")], [_term("三棱")]),
    _constraint("DB4403_C_19_08", "十九畏", [_term("官桂")], [_term("赤石脂", source_term="石脂", aliases=["石脂"])]),
    _constraint("DB4403_C_19_09", "十九畏", [_term("人参")], [_term("五灵脂")]),
]


def import_tcm_incompatibilities(
    *,
    source_pdf: Path | str = DEFAULT_RAW_PDF,
    output_path: Path | str = DEFAULT_OUTPUT,
    report_path: Path | str = DEFAULT_REPORT,
) -> dict[str, Any]:
    source_path = Path(source_pdf)
    actual_hash = _sha256(source_path)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"Unexpected source PDF hash: {actual_hash}")
    source_text = _pdf_text(source_path)
    required_markers = [
        "6.3.1.4",
        "处方用药不应有配伍禁忌",
        "附录C",
        "十八反",
        "甘草反甘遂",
        "乌头（川乌、草乌、附子）",
        "藜芦反人参",
        "十九畏",
        "人参畏五灵脂",
    ]
    missing_markers = [marker for marker in required_markers if marker not in source_text]
    if missing_markers:
        raise ValueError(f"Source PDF verification failed; missing markers: {missing_markers}")

    payload = {
        "schema_version": "1.0.0",
        "source": {
            "source_id": SOURCE_ID,
            "title": SOURCE_TITLE,
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "source_record_id": SOURCE_RECORD_ID,
            "source_section": "附录 C（资料性）表 C.1",
            "license_status": "publicly_accessible_standard_no_open_data_license",
            "usage_scope": "factual_relationship_extraction_only",
        },
        "policy": {
            "scope": "十八反、十九畏传统配伍禁忌预警",
            "matching": "exact_term_or_explicit_alias_only",
            "clinical_action": "触发红色预警并暂停绿色结论，必须由中药师或医生人工复核",
            "limitation": "这是来源标准的传统配伍禁忌关系，不代表系统自动证明绝对临床禁用；炮制、剂量和个体情况需人工判断。",
        },
        "constraints": CONSTRAINTS,
    }
    _write_json(Path(output_path), payload)

    expanded_pair_count = sum(len(item["left"]) * len(item["right"]) for item in CONSTRAINTS)
    report = {
        "source_id": SOURCE_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_file": {
            "path": str(source_path.resolve()),
            "sha256": actual_hash,
            "size_bytes": source_path.stat().st_size,
            "url": SOURCE_URL,
        },
        "output_file": str(Path(output_path).resolve()),
        "processing_version": PROCESSING_VERSION,
        "counts": {
            "constraint_groups": len(CONSTRAINTS),
            "eighteen_antagonisms_groups": sum(item["category"] == "十八反" for item in CONSTRAINTS),
            "nineteen_incompatibilities_groups": sum(item["category"] == "十九畏" for item in CONSTRAINTS),
            "expanded_graph_pairs": expanded_pair_count,
        },
        "quality_notes": [
            "逐项人工转录后，脚本再次核对 PDF 哈希和关键原文标记。",
            "Dryad/CHPA herbherb.csv 是共享属性关联而非禁配关系，未导入本规则集。",
            "未获得开放数据许可证；仅保存不受表达形式影响的事实关系和来源定位，不再分发原标准正文。",
            "规则用于医生端审方预警，不能用于患者端自动用药建议。",
        ],
    }
    _write_json(Path(report_path), report)
    return report


def download_reference_pdf(destination: Path | str = DEFAULT_RAW_PDF) -> Path:
    """Explicitly download the public reference PDF and verify its fixed hash."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".download")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "GuardTCM-source-verifier/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        temporary_path.write_bytes(response.read())
    actual_hash = _sha256(temporary_path)
    if actual_hash != SOURCE_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded source PDF hash mismatch: {actual_hash}")
    temporary_path.replace(destination_path)
    return destination_path


def _pdf_text(path: Path) -> str:
    completed = subprocess.run(
        ["pdftotext", "-f", "8", "-l", "13", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return "".join(completed.stdout.split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import source-bound TCM herb incompatibility rules")
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_RAW_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--download-reference",
        action="store_true",
        help="Explicitly download the government-hosted reference PDF before verification",
    )
    args = parser.parse_args()
    if args.download_reference:
        download_reference_pdf(args.source_pdf)
    print(json.dumps(import_tcm_incompatibilities(source_pdf=args.source_pdf, output_path=args.output, report_path=args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
