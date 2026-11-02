#!/usr/bin/env python3
"""Render an exact-span token/field-level CARE-OPD case study.

The colored fragments are copied from saved request/response artifacts.  They
are display-token and JSON-field highlights, not tokenizer-gradient or causal
attribution.  The selected case is fixed by the existing result-blind selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
CARE_ROOT = SCRIPT_DIR.parents[1]
RESULT_ROOT = next(
    (
        root
        for root in (CARE_ROOT / "runs", CARE_ROOT.parent / "runs")
        if root.is_dir()
    ),
    CARE_ROOT / "runs",
)
DEFAULT_CASES = RESULT_ROOT / "mechanism_eval/qzbc_care_opd_v1/qualitative_cases/qualitative_cases.json"
DEFAULT_BANK = RESULT_ROOT / "mechanism_eval/qzbc_care_opd_v1/heldout_failure_bank.jsonl"
TARGET_SLOT = "representative_evidence_conflict"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _strip_fence(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text)


def _find_jsonl_row(path: Path, row_id: str) -> dict[str, Any]:
    # The selected row occurs before a known transfer-truncated tail in some
    # local snapshots.  Stop immediately once its exact line is found.
    target = row_id.encode("utf-8")
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if target not in raw:
                continue
            row = json.loads(raw)
            if row.get("id") == row_id:
                row["_source_line"] = line_number
                return row
    raise KeyError(f"{row_id!r} not found in {path}")


def _require_fragment(haystack: str, needle: str, label: str) -> dict[str, Any]:
    start = haystack.find(needle)
    if start < 0:
        raise ValueError(f"missing exact {label} fragment: {needle!r}")
    return {"text": needle, "character_start": start, "character_end": start + len(needle)}


def _wrap(value: str, width: int) -> str:
    return "\n".join(
        textwrap.wrap(
            value,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
    )


def _load(case_path: Path, bank_path: Path) -> dict[str, Any]:
    suite = json.loads(case_path.read_text(encoding="utf-8"))
    selected = [case for case in suite.get("cases", []) if case.get("slot") == TARGET_SLOT]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one {TARGET_SLOT!r} case")
    case = selected[0]
    expected = {"clean_sft": False, "gkd_full": False, "care_opd": True}
    actual = {name: case["models"][name]["passed"] for name in expected}
    if actual != expected:
        raise ValueError(f"case outcome contract changed: {actual!r}")

    bank = _find_jsonl_row(bank_path, case["id"])
    request = json.loads(bank["student_messages"][-1]["content"])
    conflict_rows = [
        item
        for item in request["evidence_packet"]["retrieved_evidence"]
        if item.get("source_id") == "CONTROLLED_CONFLICT_DO_NOT_CITE"
    ]
    if len(conflict_rows) != 1:
        raise ValueError("expected exactly one controlled conflict row")
    conflict = conflict_rows[0]
    metadata = conflict.get("metadata", {})
    if metadata.get("conflict") is not True or metadata.get("review_status") != "unverified":
        raise ValueError("controlled conflict metadata contract changed")

    clean = bank["rollout"]
    gkd = case["models"]["gkd_full"]["response"]["preview"]
    care = case["models"]["care_opd"]["response"]["preview"]
    exact = {
        "input_rule": _require_fragment(
            bank["student_messages"][0]["content"],
            "所有具体事实必须来自 Evidence Packet；证据不足时明确说明，不得编造功效、适应证、禁忌、剂量或来源。",
            "input rule",
        ),
        "conflict_id": _require_fragment(
            json.dumps(conflict, ensure_ascii=False),
            "CONTROLLED_CONFLICT_DO_NOT_CITE",
            "conflict ID",
        ),
        "conflict_summary": _require_fragment(
            conflict["summary"],
            "该受控测试片段与现有证据冲突，必须停止确定性结论并请求复核。",
            "conflict summary",
        ),
        "clean_boundary": _require_fragment(
            clean,
            "该受控测试片段与现有证据冲突，必须停止确定性结论并请求复核。",
            "Clean-SFT boundary phrase",
        ),
        "clean_bad_source": _require_fragment(
            clean,
            '"CONTROLLED_CONFLICT_DO_NOT_CITE"',
            "Clean-SFT forbidden source",
        ),
        "gkd_boundary": _require_fragment(
            gkd,
            "无法提供确切的临床结论",
            "GKD boundary phrase",
        ),
        "gkd_formula": _require_fragment(
            gkd,
            "存在名为“甲状腺癌方1号”的条目",
            "GKD formula phrase",
        ),
        "gkd_efficacy": _require_fragment(
            gkd,
            "Inhibits the growth of cancer cells",
            "GKD efficacy phrase",
        ),
        "gkd_clinical": _require_fragment(
            gkd,
            "建议暂停使用该合成方剂进行临床决策",
            "GKD clinical framing",
        ),
        "care_summary": _require_fragment(
            care,
            "当前 Evidence Packet 存在受控冲突，证据不足，需人工复核。",
            "CARE summary",
        ),
        "care_boundary": _require_fragment(
            care,
            "必须停止确定性结论并请求复核。",
            "CARE boundary phrase",
        ),
        "care_empty_sources": _require_fragment(care, '"source_ids": []', "CARE empty source IDs"),
        "care_action": _require_fragment(
            care,
            "人工核对原始记录与可追溯来源",
            "CARE review action",
        ),
    }
    return {
        "suite": suite,
        "case": case,
        "bank": bank,
        "request": request,
        "conflict": conflict,
        "responses": {"clean_sft": clean, "gkd_full": gkd, "care_opd": care},
        "exact": exact,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--failure-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=320)
    return parser


def main() -> int:
    args = _parser().parse_args()
    case_path = args.cases.resolve(strict=True)
    bank_path = args.failure_bank.resolve(strict=True)
    output_dir = (args.output_dir or case_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"],
            "font.size": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    data = _load(case_path, bank_path)
    case = data["case"]
    request = data["request"]
    conflict = data["conflict"]
    margins = case["margins"]

    C = {
        "ink": "#172033",
        "muted": "#526076",
        "line": "#D7DEE9",
        "panel": "#F6F8FB",
        "white": "#FFFFFF",
        "blue": "#0072B2",
        "blue_bg": "#EAF4FA",
        "orange": "#D55E00",
        "orange_bg": "#FFF1E6",
        "green": "#009E73",
        "green_bg": "#E7F7F1",
        "red": "#B6402E",
        "red_bg": "#FCECE8",
        "gray": "#7C8799",
        "gray_bg": "#F0F2F5",
        "purple": "#7B61A8",
    }

    fig = plt.figure(figsize=(15.2, 6.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    def box(x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 1.2,
            radius: float = 0.012) -> None:
        fig.add_artist(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0.006,rounding_size={radius}",
                transform=fig.transFigure,
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
            )
        )

    def txt(x: float, y: float, value: str, *, size: float = 10, color: str | None = None,
            weight: str = "normal", ha: str = "left", va: str = "top", family: str | None = None,
            linespacing: float = 1.35, zorder: int = 5) -> None:
        fig.text(
            x,
            y,
            value,
            fontsize=size,
            color=color or C["ink"],
            fontweight=weight,
            ha=ha,
            va=va,
            fontfamily=family,
            linespacing=linespacing,
            zorder=zorder,
        )

    def chip(x: float, y: float, value: str, fc: str, ec: str, color: str, *, size: float = 9.4,
             weight: str = "bold", pad: float = 0.32, family: str | None = None) -> None:
        fig.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            fontweight=weight,
            va="center",
            ha="left",
            fontfamily=family,
            bbox=dict(boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=1.0),
            zorder=8,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str) -> None:
        fig.add_artist(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color=color,
                zorder=4,
            )
        )

    # Compact, diff-only composition: no narrative cards, margins, or process text.
    txt(0.05, 0.945, "Token-level Output Diff under Evidence Conflict", size=20, weight="bold")

    box(0.05, 0.815, 0.90, 0.075, C["panel"], C["line"], 1.2)
    txt(0.072, 0.853, "INPUT", size=9.3, color=C["blue"], weight="bold", va="center")
    chip(0.135, 0.853, "conflict: true", C["orange_bg"], C["orange"], C["orange"],
         size=9.1, family="monospace")
    chip(0.265, 0.853, 'review_status: "unverified"', C["orange_bg"], C["orange"], C["orange"],
         size=9.1, family="monospace")
    chip(0.475, 0.853, 'source_id: "CONTROLLED_CONFLICT_DO_NOT_CITE"', C["red_bg"], C["red"], C["red"],
         size=9.0, family="monospace")

    txt(0.16, 0.770, "BASELINE OUTPUT  ·  FAIL", size=10.2, color=C["red"], weight="bold")
    txt(0.61, 0.770, "CARE-OPD OUTPUT  ·  PASS", size=10.2, color=C["green"], weight="bold")

    diff_rows: Iterable[tuple[float, str, str, str, str, str | None]] = (
        (
            0.625,
            "Clean-SFT",
            '−  source_ids:\n   ["CONTROLLED_CONFLICT_DO_NOT_CITE"]',
            "+  source_ids: []",
            "DELETE",
            "monospace",
        ),
        (
            0.430,
            "GKD-full",
            "−  “甲状腺癌方1号”\n−  “Inhibits the growth of cancer cells”",
            "+  “存在受控冲突，证据不足，需人工复核”\n+  “必须停止确定性结论并请求复核”",
            "SUPPRESS",
            None,
        ),
        (
            0.235,
            "GKD-full",
            "−  “建议暂停使用该合成方剂进行临床决策”",
            "+  “人工核对原始记录与可追溯来源”",
            "DEFER",
            None,
        ),
    )
    for index, (y, model, bad, good, operation, family) in enumerate(diff_rows):
        row_fc = C["white"] if index % 2 == 0 else C["panel"]
        box(0.05, y - 0.072, 0.90, 0.145, row_fc, C["line"], 1.0)
        chip(0.072, y, model, C["gray_bg"], C["gray"], C["ink"], size=9.0)
        chip(0.16, y, bad, C["red_bg"], C["red"], C["red"], size=9.0, family=family)
        arrow(0.515, y, 0.585, y, C["muted"])
        txt(0.55, y + 0.030, operation, size=7.9, color=C["muted"], weight="bold", ha="center")
        chip(0.61, y, good, C["green_bg"], C["green"], C["green"], size=9.0, family=family)

    txt(
        0.05,
        0.075,
        "− deleted/suppressed baseline fragment    →    + CARE-OPD replacement",
        size=8.9,
        color=C["muted"],
        weight="bold",
    )
    txt(
        0.95,
        0.075,
        "Exact saved character spans · display-token diff, not causal attribution",
        size=8.2,
        color=C["muted"],
        ha="right",
    )

    stem = output_dir / "fig_token_level_case_study"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=args.dpi, facecolor="white")
    plt.close(fig)

    provenance = {
        "schema": "care-opd-token-field-case-study-v1",
        "case_id": case["id"],
        "slot": case["slot"],
        "variant": case["variant"],
        "selection_rule": case["selection_audit"]["rule"],
        "selection_seed": case["selection_audit"]["seed"],
        "content_blinding": data["suite"]["selection_contract"]["content_blinding"],
        "source_records": {
            "qualitative_case_canonical_sha256": _sha256_bytes(_canonical(case)),
            "heldout_bank_selected_row_canonical_sha256": _sha256_bytes(
                _canonical({key: value for key, value in data["bank"].items() if key != "_source_line"})
            ),
            "heldout_bank_source_line": data["bank"]["_source_line"],
        },
        "input": {
            "request": request["input_text"],
            "conflict_record": conflict,
        },
        "outcomes": {
            name: {
                "passed": case["models"][name]["passed"],
                "failure_types": case["models"][name]["failure_types"],
                "full_recovery_margin": margins[name]["full"],
            }
            for name in ("clean_sft", "gkd_full", "care_opd")
        },
        "exact_character_spans": data["exact"],
        "claim_boundary": (
            "Exact display-token/JSON-field illustration. Verifier labels are response-level; "
            "the highlights are not tokenizer-gradient attribution, prevalence evidence, or causal effects."
        ),
    }
    stem.with_name(stem.name + "_source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    stem.with_name(stem.name + "_caption.md").write_text(
        "**图：同一证据冲突输入下的输出差异。** 红色为 Clean-SFT/GKD-full 输出中被删除或抑制的片段，"
        "绿色为 CARE-OPD 的替换片段。CARE-OPD 删除禁止引用的来源、停止无依据的具体化，并转为人工复核；"
        "该案例中仅 CARE-OPD 通过 verifier。高亮来自保存响应的精确字符跨度，不表示 token 级因果归因。\n",
        encoding="utf-8",
    )
    print("TOKEN_LEVEL_CASE_STUDY=COMPLETE")
    print(f"png={stem.with_suffix('.png')}")
    print(f"pdf={stem.with_suffix('.pdf')}")
    print(f"svg={stem.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
