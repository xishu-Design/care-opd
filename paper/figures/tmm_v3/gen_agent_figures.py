#!/usr/bin/env python3
"""Publication-sized Agent figures from archived evidence, without reevaluation.

Grouped-bar and annotated-matrix grammars follow the CCFA recipe contracts,
ported to Matplotlib for type-42 PDF and editable SVG output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
PAPER = OUT.parent.parent
REPO = PAPER.parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_agent"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
import numpy as np

STAT = REPO / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/02_完整统计"
BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"
SKY_BLUE = "#56B4E9"
GRAY, LIGHT_GRAY, INK = "#666666", "#B8B8B8", "#111111"
RETRIEVAL_COLORS = {"BM25": BLUE, "Hybrid": SKY_BLUE,
                    "Naive LightRAG": ORANGE, "Guard Retrieval": GREEN}


def read_csv(name: str) -> list[dict]:
    with (STAT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def provenance(paths: list[Path]) -> list[dict]:
    return [{"path": p.relative_to(REPO).as_posix(),
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]


def style() -> None:
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/times.ttf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "hatch.linewidth": 0.5,
    })


def percent_axis(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Score (%)", labelpad=2)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=2.5, pad=2)
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)


def save(fig, stem: str, data: dict, source_paths: list[Path]) -> None:
    # Validate the compact layout without resizing or clipping text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
                clipped.append(item.get_text())
    assert not clipped, (stem, clipped)
    # Exact physical size is preserved; no bbox_inches='tight' rescaling.
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=300)
    payload = {
        "schema": "tmm-agent-figure-source-v3", "sources": provenance(source_paths),
        "figure": stem, "size_inches": [float(v) for v in fig.get_size_inches()],
        "font": "Times New Roman", "font_size_pt": 8.5, "tick_size_pt": 8,
        "minimum_text_size_pt": 7.5, "data": data,
        "color_encoding": {
            "fig_agent_evidence": {"retriever": RETRIEVAL_COLORS,
                                   "configuration": {"Role-wide": BLUE, "MinimalTool": GREEN},
                                   "metric_shape": {"Top-1": "filled circle", "Recall@5": "open diamond"}},
            "fig_memory": {"judge_accuracy": BLUE, "recall_at_5": GREEN,
                           "metric_shape": {"judge_accuracy": "filled circle",
                                            "recall_at_5": "filled diamond"},
                           "connection": "none", "uncertainty_marks": "none"},
            "fig_memory_categories": {"colormap": "Blues", "range": [0, 100],
                                      "text": "black or white selected by relative luminance"},
        }[stem],
        "statistics": "Archived observations only; no new evaluation or uncertainty estimation.",
    }
    (OUT / f"{stem}.source.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)


def agent_evidence() -> None:
    retrieval = read_csv("RETRIEVAL_BENCHMARK_FULL.csv")
    execution = {row["endpoint"]: row for row in read_csv("AGENT_TOOL_CONTRACT_FULL.csv")}
    fig = plt.figure(figsize=(7.16, 2.0))
    ax = fig.add_axes([0.145, 0.235, 0.315, 0.650])
    bx = fig.add_axes([0.690, 0.235, 0.280, 0.650])
    fig.text(0.025, 0.975, "(a)", va="top", fontsize=8.5)
    fig.text(0.545, 0.975, "(b)", va="top", fontsize=8.5)
    positions = [8, 7, 6, 5, 2.7, 1.7, 0.7, -0.3]
    for row, y in zip(retrieval, positions):
        color = RETRIEVAL_COLORS[row["method"]]
        ax.plot(float(row["top_1"]) * 100, y, "o", color=color,
                markeredgecolor=color, markersize=3.8, clip_on=False)
        ax.plot(float(row["recall_at_5"]) * 100, y, "D", markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=0.85, markersize=3.5, clip_on=False)
    ax.set_yticks(positions, [row["method"] for row in retrieval])
    for tick, row in zip(ax.get_yticklabels(), retrieval):
        tick.set_color(RETRIEVAL_COLORS[row["method"]])
    ax.set_ylim(-0.95, 9.25)
    ax.text(0, 8.95, "General (n=1,773)", fontsize=8, va="center")
    ax.text(0, 3.65, "Relation hidden (n=432)", fontsize=8, va="center")
    ax.axhline(4.2, color="#AAAAAA", linewidth=0.45)
    percent_axis(ax)
    handles = [Line2D([], [], marker="o", color=GRAY, linestyle="none", markersize=3.8, label="Top-1"),
               Line2D([], [], marker="D", color=GRAY, markerfacecolor="white", linestyle="none",
                      markersize=3.5, label="Recall@5")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.19, 0.975),
               ncol=2, handletextpad=0.4, columnspacing=1.2, borderaxespad=0, fontsize=8)

    metrics = ["tool_micro_precision", "tool_micro_recall", "tool_exact", "evidence_source_coverage"]
    labels = ["Tool precision", "Tool recall", "Tool exact", "Source coverage\n(41 source slots)"]
    y = np.arange(4)[::-1]
    for key, color, offset, name in [("legacy_role_wide", BLUE, 0.20, "Role-wide"),
                                    ("minimal_tool", GREEN, -0.20, "MinimalTool")]:
        vals = [float(execution[m][key]) * 100 for m in metrics]
        bx.barh(y + offset, vals, height=0.29, color=color, edgecolor="white", linewidth=0.35)
        for yi, value in zip(y + offset, vals):
            # High values are labeled inside their own bars to avoid clipping at 100%.
            if value > 75:
                bx.text(value - 1.5, yi, f"{value:.2f}", va="center", ha="right",
                        fontsize=7.5, color="white" if key == "legacy_role_wide" else INK)
            else:
                bx.text(value + 1.4, yi, f"{value:.2f}", va="center", ha="left", fontsize=7.5)
    bx.set_yticks(y, labels)
    bx.set_ylim(-0.62, 3.67)
    percent_axis(bx)
    fig.legend(handles=[Patch(facecolor=BLUE, label="Role-wide"),
                        Patch(facecolor=GREEN, label="MinimalTool")],
               loc="upper left", bbox_to_anchor=(0.660, 0.975), borderaxespad=0,
               ncol=2, handlelength=1.05, handletextpad=0.4, columnspacing=1.0, fontsize=8)
    data = {
        "retrieval": retrieval, "execution_endpoints": metrics,
        "execution": [{"endpoint": m,
                       "role_wide_percent": float(execution[m]["legacy_role_wide"]) * 100,
                       "minimal_tool_percent": float(execution[m]["minimal_tool"]) * 100}
                      for m in metrics],
        "execution_requests": 120, "eligible_source_slots": 41,
        "source_coverage_counts": {"role_wide": 26, "minimal_tool": 22},
        "claim_boundary": "Independent retrieval and fixed-generator tool-exposure experiments; not a CARE training effect.",
    }
    save(fig, "fig_agent_evidence", data, [STAT / "RETRIEVAL_BENCHMARK_FULL.csv", STAT / "AGENT_TOOL_CONTRACT_FULL.csv"])


def memory() -> None:
    rows = [r for r in read_csv("MEMORY_BENCHMARK_FULL.csv") if "controlled" in r["track"]]
    fig = plt.figure(figsize=(3.49, 1.90))
    ax = fig.add_axes([0.210, 0.220, 0.755, 0.640])
    y = np.arange(4)[::-1]
    labels = ["Sliding\nwindow", "Summary +\nrecent", "Mem0", "Mem0 +\nGraphiti"]
    # The two marker lanes belong to the same configuration, not intervals.
    # Vertical dodging also separates the nearby low-score value labels.
    for key, color, offset, marker in [("judge_accuracy", BLUE, 0.21, "o"),
                                      ("recall_at_5", GREEN, -0.21, "D")]:
        values = [100 * float(r[key]) for r in rows]
        ax.plot(values, y + offset, linestyle="none", marker=marker,
                markersize=3.8, markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=0.5, zorder=3)
        for yi, value in zip(y + offset, values):
            text = f"{value:.1f}" if key == "judge_accuracy" else f"{value:.2f}"
            inside = value >= 85
            ax.annotate(text, (value, yi), xytext=(-4 if inside else 4, 0),
                        textcoords="offset points", va="center",
                        ha="right" if inside else "left", fontsize=7.5, color=INK,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.10})
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.65, 3.65)
    percent_axis(ax)
    fig.legend(handles=[Line2D([], [], linestyle="none", marker="o", color=BLUE,
                               markersize=3.8, label="Judge accuracy (non-official)"),
                        Line2D([], [], linestyle="none", marker="D", color=GREEN,
                               markersize=3.8, label="Evidence Recall@5")],
               loc="upper left", bbox_to_anchor=(0.210, 0.990), borderaxespad=0,
               handlelength=1.2, handletextpad=0.5, labelspacing=0.15, fontsize=8)
    save(fig, "fig_memory", {"protocol": "controlled", "overall_n": 500,
         "judge": "project non-official", "rows": rows,
         "claim_boundary": "System-memory configurations with a fixed generator; no real-component pooling or CARE attribution."},
         [STAT / "MEMORY_BENCHMARK_FULL.csv"])


def memory_categories() -> None:
    source = OUT.parent / "fig_memory_categories.data.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.16, 2.60))
    ax = fig.add_axes([0.255, 0.145, 0.625, 0.805])
    bottoms = [5.25, 4.25, 3.25, 2.25, 1, 0]
    cmap = plt.get_cmap("Blues")
    norm = colors.Normalize(0, 100)
    for i, row in enumerate(payload["rows"]):
        for j, value in enumerate(row["values_percent"]):
            cell_color = cmap(norm(value))
            rgb_linear = [(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
                          for v in cell_color[:3]]
            luminance = sum(v * w for v, w in zip(rgb_linear, (0.2126, 0.7152, 0.0722)))
            ax.add_patch(Rectangle((j, bottoms[i]), 1, 1, facecolor=cell_color,
                                   edgecolor="white", linewidth=0.7))
            ax.text(j + 0.5, bottoms[i] + 0.5, str(value), va="center", ha="center",
                    fontsize=8.5, color="white" if luminance < 0.179 else INK)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6.25)
    ax.set_xticks(np.arange(6) + 0.5, payload["columns"])
    ax.set_yticks([b + 0.5 for b in bottoms], [r["configuration"] for r in payload["rows"]])
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(2.125, color=GRAY, linewidth=0.45)
    fig.text(0.027, 0.6924, "Controlled", rotation=90, va="center", ha="center", fontsize=8.5)
    fig.text(0.027, 0.2738, "Real isolated", rotation=90, va="center", ha="center", fontsize=8.5)
    cax = fig.add_axes([0.902, 0.145, 0.016, 0.805])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, 50, 100])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, width=0.6, pad=3)
    bar.set_label("Judge accuracy (%)", labelpad=3)
    save(fig, "fig_memory_categories", {
        "columns": payload["columns"], "rows": payload["rows"],
        "overall_n_per_protocol": 500, "category_sample_sizes": None, "category_intervals": None,
        "judge": "project non-official", "display_precision": "source-figure integer percentages",
        "claim_boundary": "Protocols separated; no inferred subgroup precision or significance."}, [source])


if __name__ == "__main__":
    style()
    agent_evidence()
    memory()
    memory_categories()
    print("Wrote Agent evidence, controlled memory, and supplementary memory categories in", OUT)
