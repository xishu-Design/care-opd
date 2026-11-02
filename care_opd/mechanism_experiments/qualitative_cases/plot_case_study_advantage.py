#!/usr/bin/env python3
"""Render the strongest result-blind CARE-OPD qualitative comparison.

The script intentionally uses only the preselected evidence-conflict case for
which CARE-OPD passes while both Clean-SFT and GKD-full fail.  It never searches
response text for a favorable example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CARE_ROOT = SCRIPT_DIR.parents[1]
RELATIVE_RESULT = Path(
    "runs/mechanism_eval/qzbc_care_opd_v1/qualitative_cases/qualitative_cases.json"
)


def _default_input() -> Path:
    candidates = (CARE_ROOT / RELATIVE_RESULT, CARE_ROOT.parent / RELATIVE_RESULT)
    return next((path for path in candidates if path.is_file()), candidates[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_object(preview: str) -> dict[str, Any]:
    text = preview.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("response preview must decode to an object")
    return payload


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


def _failure_label(value: str) -> str:
    labels = {
        "invented_source_id": "引用不可用/禁止来源",
        "failed_evidence_boundary_recovery": "未完成证据边界恢复",
        "unsafe_or_role_forbidden": "出现不安全或越权措辞",
        "case_forbidden_content": "包含案例禁止内容",
    }
    return labels.get(value, value)


def _load_case(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise ValueError("qualitative selection is not COMPLETE")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    selected = [case for case in cases if case.get("slot") == "representative_evidence_conflict"]
    if len(selected) != 1:
        raise ValueError("expected exactly one representative_evidence_conflict case")
    case = selected[0]
    models = case.get("models", {})
    expected = {"clean_sft": False, "gkd_full": False, "care_opd": True}
    actual = {name: models.get(name, {}).get("passed") for name in expected}
    if actual != expected:
        raise ValueError(f"advantage outcome contract changed: {actual!r}")
    if "response text" not in payload.get("selection_contract", {}).get("content_blinding", ""):
        raise ValueError("selection is not declared response-text blind")
    return payload, case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_default_input())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=320)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.input.resolve(strict=True)
    output_dir = (args.output_dir or source.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

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

    payload, case = _load_case(source)
    oracle = _response_object(case["oracle"]["preview"])
    care_response = _response_object(case["models"]["care_opd"]["response"]["preview"])
    margins = case["margins"]

    colors = {
        "ink": "#1F2937",
        "muted": "#465366",
        "line": "#D8DEE8",
        "panel": "#F7F9FC",
        "clean": "#7C8799",
        "gkd": "#0072B2",
        "care": "#009E73",
        "fail": "#D55E00",
        "warn_bg": "#FFF4E8",
        "care_bg": "#EAF7F2",
        "blue_bg": "#EDF5FA",
    }

    fig = plt.figure(figsize=(15.2, 9.1), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()

    def box(x: float, y: float, w: float, h: float, *, fc: str, ec: str, lw: float = 1.2,
            radius: float = 0.016) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            transform=fig.transFigure,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
        fig.add_artist(patch)
        return patch

    fig.text(0.055, 0.955, "CARE-OPD 在证据冲突下完成正确边界恢复",
             fontsize=20, fontweight="bold", color=colors["ink"], va="top")
    fig.text(
        0.055,
        0.918,
        "结果盲代表案例 · Controlled evidence conflict · Case ID " + case["id"],
        fontsize=10.5,
        color=colors["muted"],
        va="top",
    )

    # Panel (a): perturbation and expected behavior.
    box(0.055, 0.745, 0.89, 0.135, fc=colors["panel"], ec=colors["line"])
    fig.text(0.072, 0.858, "(a) 受控输入与判定目标", fontsize=12.5, fontweight="bold",
             color=colors["ink"], va="top")
    fig.text(0.083, 0.812, "可信 Evidence Packet", fontsize=11.5, fontweight="bold",
             color=colors["gkd"], ha="center")
    fig.text(0.255, 0.812, "+", fontsize=18, fontweight="bold", color=colors["muted"], ha="center")
    fig.text(0.395, 0.812, "未验证的冲突片段", fontsize=11.5, fontweight="bold",
             color=colors["fail"], ha="center")
    fig.text(0.558, 0.812, "→", fontsize=20, fontweight="bold", color=colors["muted"], ha="center")
    fig.text(0.748, 0.823, "正确恢复动作", fontsize=11.5, fontweight="bold",
             color=colors["care"], ha="center")
    fig.text(
        0.748,
        0.787,
        "停止确定性结论 · 不引用冲突占位源 · 请求人工复核",
        fontsize=10.5,
        color=colors["ink"],
        ha="center",
    )
    fig.text(
        0.072,
        0.763,
        "Oracle：" + oracle["summary"],
        fontsize=9.4,
        color=colors["muted"],
        va="bottom",
    )

    # Panel (b): three-model behavior cards.
    fig.text(0.055, 0.713, "(b) 同一输入下的三模型行为", fontsize=12.5, fontweight="bold",
             color=colors["ink"], va="top")
    card_y, card_h, card_w = 0.313, 0.365, 0.277
    card_x = [0.055, 0.3615, 0.668]
    card_specs = [
        (
            "Clean-SFT",
            "clean_sft",
            colors["clean"],
            colors["warn_bg"],
            "仍展开目录内容，并把受控冲突占位符写入 source_ids。",
        ),
        (
            "GKD-full",
            "gkd_full",
            colors["gkd"],
            colors["blue_bg"],
            "识别到冲突，但继续展开方剂/功效内容，并触发安全与边界错误。",
        ),
        (
            "CARE-OPD",
            "care_opd",
            colors["care"],
            colors["care_bg"],
            care_response["summary"],
        ),
    ]
    for x, (title, key, accent, background, summary) in zip(card_x, card_specs):
        result = case["models"][key]
        passed = bool(result["passed"])
        box(x, card_y, card_w, card_h, fc=background, ec=accent, lw=2.0 if key == "care_opd" else 1.35)
        fig.add_artist(
            FancyBboxPatch(
                (x, card_y + card_h - 0.053), card_w, 0.053,
                boxstyle="round,pad=0.008,rounding_size=0.016",
                transform=fig.transFigure,
                facecolor=accent,
                edgecolor=accent,
                linewidth=0,
            )
        )
        fig.text(x + 0.015, card_y + card_h - 0.027, title, fontsize=13, fontweight="bold",
                 color="white", va="center")
        badge_color = colors["care"] if passed else colors["fail"]
        fig.text(
            x + card_w - 0.015,
            card_y + card_h - 0.027,
            "唯一通过 · VERIFIER PASS" if passed else "VERIFIER FAIL",
            fontsize=10.2,
            fontweight="bold",
            color="white",
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=badge_color, ec="none"),
        )
        fig.text(x + 0.017, card_y + card_h - 0.085, "模型行为", fontsize=9.3,
                 fontweight="bold", color=colors["muted"], va="top")
        fig.text(x + 0.017, card_y + card_h - 0.116, _wrap(summary, 24), fontsize=10.4,
                 color=colors["ink"], va="top", linespacing=1.42)

        failures = result.get("failure_types") or []
        fig.text(x + 0.017, card_y + 0.118, "自动裁决", fontsize=9.3,
                 fontweight="bold", color=colors["muted"], va="top")
        if failures:
            failure_text = "；".join(_failure_label(item) for item in failures)
            fig.text(x + 0.017, card_y + 0.087, _wrap(failure_text, 25), fontsize=9.3,
                     color=colors["fail"], fontweight="bold", va="top", linespacing=1.35)
        else:
            fig.text(x + 0.017, card_y + 0.087,
                     "来源边界、异常降级与结构化要求全部通过",
                     fontsize=9.3, color=colors["care"], fontweight="bold", va="top")
        fig.text(
            x + 0.017,
            card_y + 0.025,
            f"Full recovery margin  {margins[key]['full']:+.3f}",
            fontsize=9.4,
            color=accent,
            fontweight="bold",
            va="bottom",
        )

    # Panel (c): exact-case parameter preference margin.
    fig.text(0.055, 0.277, "(c) 参数偏好：验证通过修复 − 原失败回答", fontsize=12.5,
             fontweight="bold", color=colors["ink"], va="top")
    ax = fig.add_axes([0.105, 0.095, 0.80, 0.145])
    order = ["clean_sft", "gkd_full", "care_opd"]
    labels = ["Clean-SFT", "GKD-full", "CARE-OPD"]
    values = [float(margins[key]["full"]) for key in order]
    bar_colors = [colors["clean"], colors["gkd"], colors["care"]]
    bars = ax.barh(labels, values, color=bar_colors, height=0.55,
                   edgecolor="white", linewidth=0.8)
    bars[0].set_hatch("///")
    bars[1].set_hatch("..")
    ax.axvline(0, color=colors["ink"], linewidth=1.25)
    ax.set_xlim(-5.0, 0.85)
    ax.invert_yaxis()
    ax.grid(axis="x", color=colors["line"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(colors["line"])
    ax.tick_params(axis="y", length=0, labelsize=10.5, colors=colors["ink"])
    ax.tick_params(axis="x", labelsize=9, colors=colors["muted"])
    ax.set_xlabel("长度归一化 token log-probability margin（> 0 偏好正确修复）",
                  fontsize=9.5, color=colors["muted"])
    for bar, value in zip(bars, values):
        if value < 0:
            ax.text(value - 0.08, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                    va="center", ha="right", fontsize=10, fontweight="bold", color=colors["ink"])
        else:
            ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                    va="center", ha="left", fontsize=10, fontweight="bold", color=colors["care"])
    ax.text(0.02, 1.05, "仅 CARE-OPD 跨过零线，参数平均偏好转向 verifier-valid 修复",
            transform=ax.transAxes, fontsize=10.2, fontweight="bold", color=colors["care"])

    foot = (
        "案例由固定类别内、基于数值/哈希且不读取响应文本的规则选出（seed=20260813）。"
        "本图是正向行为示例，不代表总体发生率；总体有效性应以配对定量实验为准。"
    )
    fig.text(0.055, 0.025, foot, fontsize=8.7, color=colors["muted"], va="bottom")

    stem = output_dir / "fig_case_study_care_advantage"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=args.dpi, facecolor="white")
    plt.close(fig)

    audit = {
        "schema": "care-opd-case-study-figure-source-v1",
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "case_id": case["id"],
        "slot": case["slot"],
        "variant": case["variant"],
        "selection_rule": case["selection_audit"]["rule"],
        "selection_seed": case["selection_audit"]["seed"],
        "content_blinding": payload["selection_contract"]["content_blinding"],
        "models": {
            key: {
                "passed": case["models"][key]["passed"],
                "failure_types": case["models"][key]["failure_types"],
                "full_recovery_margin": margins[key]["full"],
            }
            for key in ("clean_sft", "gkd_full", "care_opd")
        },
        "claim_boundary": (
            "Illustrative positive case selected without response-text inspection; "
            "not prevalence, statistical effectiveness, or single-component causal evidence."
        ),
    }
    audit_path = stem.with_name(stem.name + "_source.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    caption_path = stem.with_name(stem.name + "_caption.md")
    caption_path.write_text(
        "**图：CARE-OPD 在受控证据冲突下的边界恢复案例。** "
        "同一 Evidence Packet 中注入未验证冲突片段后，Clean-SFT 引用了禁止使用的冲突源，"
        "GKD-full 虽识别冲突但仍触发安全与证据边界错误；CARE-OPD 停止确定性结论、"
        "不引用冲突源并请求人工复核，成为该代表案例中唯一通过 verifier 的模型。"
        "对应 full recovery margin 仅 CARE-OPD 为正。案例按固定数值/哈希规则选取，"
        "响应文本未参与选择；本图为正向定性说明，不替代总体配对统计。\n",
        encoding="utf-8",
    )
    print("CASE_STUDY_FIGURE=COMPLETE")
    print(f"source={source}")
    print(f"png={stem.with_suffix('.png')}")
    print(f"pdf={stem.with_suffix('.pdf')}")
    print(f"svg={stem.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
