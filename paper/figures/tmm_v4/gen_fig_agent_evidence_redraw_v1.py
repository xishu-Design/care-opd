"""Redraw independent agent evaluations without changing archived results."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
STAT = ROOT / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/02_完整统计"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_agent"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

from figure_style import apply_style

STEM = "fig_agent_evidence_redraw_v1"
BLUE, APRICOT = "#6C94C7", "#E6B18A"
BLUE_LINE, APRICOT_LINE = "#396B9E", "#AC713E"
INK, GRID, AXIS = "#20314B", "#E7ECF2", "#8492A2"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    paths = [STAT / "RETRIEVAL_BENCHMARK_FULL.csv", STAT / "AGENT_TOOL_CONTRACT_FULL.csv"]
    retrieval, execution_rows = [read_rows(p) for p in paths]
    execution = {r["endpoint"]: r for r in execution_rows}
    methods = ["BM25", "Hybrid", "Naive LightRAG", "Guard Retrieval"]
    tracks = ["general_semantic", "relation_hidden"]
    lookup = {(r["track"], r["method"]): r for r in retrieval}
    values = {track: {metric: [100 * float(lookup[track, m][metric]) for m in methods]
                      for metric in ("top_1", "recall_at_5")} for track in tracks}
    metric_keys = ["tool_micro_precision", "tool_micro_recall", "tool_exact", "evidence_source_coverage"]
    exec_values = {c: [100 * float(execution[k][c]) for k in metric_keys]
                   for c in ("legacy_role_wide", "minimal_tool")}
    previous = json.loads((OUT / "fig_agent_evidence.source.json").read_text(encoding="utf-8"))
    assert previous["data"]["retrieval"] == retrieval
    for metric, row in zip(metric_keys, previous["data"]["execution"]):
        assert row["endpoint"] == metric
        assert abs(row["role_wide_percent"] - 100 * float(execution[metric]["legacy_role_wide"])) < 1e-10
        assert abs(row["minimal_tool_percent"] - 100 * float(execution[metric]["minimal_tool"])) < 1e-10

    apply_style()
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": AXIS, "axes.grid": False,
                         "svg.hashsalt": STEM, "savefig.bbox": None,
                         "hatch.linewidth": .35})
    fig = plt.figure(figsize=(7.16, 2.34), dpi=450)
    a = fig.add_axes([.062, .160, .426, .622])
    b = fig.add_axes([.562, .160, .426, .622])
    headings = [fig.text(.062, .957, "(a)  Retrieval quality", weight="bold", fontsize=8.6),
                fig.text(.562, .957, "(b)  Tool exposure", weight="bold", fontsize=8.6)]
    x = np.arange(4)
    labels = []
    for track, offset, face, edge, marker, style in [
            (tracks[0], -.19, BLUE, BLUE_LINE, "o", "-"),
            (tracks[1], .19, APRICOT, APRICOT_LINE, "D", (0, (3.5, 2.3)))]:
        top1, recall = values[track]["top_1"], values[track]["recall_at_5"]
        a.bar(x + offset, top1, width=.33, color=face,
              edgecolor=edge if track == tracks[1] else "none",
              hatch="/" if track == tracks[1] else None,
              linewidth=.3, zorder=3)
        a.plot(x + offset, recall, color=edge, linestyle=style, lw=1.2,
               marker=marker, ms=3.7, markerfacecolor="white", markeredgewidth=1.0, zorder=5)
        for xp, y in zip(x + offset, top1):
            labels.append(a.text(xp, y - 6.8, f"{y:.2f}", ha="center", va="center",
                                 color=INK, fontsize=7.1, zorder=6))
    a.set_xticks(x, ["BM25", "Hybrid", "Naive\nLightRAG", "Guard\nRetrieval"])
    a.set_ylabel("Score (%)", labelpad=3)
    legend_a = a.legend(handles=[
        Patch(facecolor=BLUE, label="General Top-1"),
        Line2D([], [], color=BLUE_LINE, marker="o", markerfacecolor="white", ms=3.4, lw=1.2,
               label="General Recall@5"),
        Patch(facecolor=APRICOT, edgecolor=APRICOT_LINE, hatch="/", linewidth=.3, label="Relation Top-1"),
        Line2D([], [], color=APRICOT_LINE, marker="D", markerfacecolor="white", ms=3.3, lw=1.2,
               ls=(0, (3.5, 2.3)), label="Relation Recall@5")],
        loc="upper center", bbox_to_anchor=(.50, 1.258), ncol=2,
        borderaxespad=0, handlelength=1.6, columnspacing=1.05,
        handletextpad=.45, labelspacing=.25, fontsize=7.3)

    for key, offset, face, edge in [("legacy_role_wide", -.19, APRICOT, APRICOT_LINE),
                                    ("minimal_tool", .19, BLUE, BLUE_LINE)]:
        yvals = exec_values[key]
        b.bar(x + offset, yvals, width=.33, color=face,
              edgecolor=edge if key == "legacy_role_wide" else "none",
              hatch="/" if key == "legacy_role_wide" else None, linewidth=.3, zorder=3)
        for xp, y in zip(x + offset, yvals):
            labels.append(b.text(xp, y - 6.8, f"{y:.2f}", ha="center", va="center",
                                 color=INK, fontsize=7.1, zorder=4))
    b.set_xticks(x, ["Precision", "Recall", "Tool exact", "Source\ncoverage"])
    legend_b = b.legend(handles=[Patch(facecolor=APRICOT, edgecolor=APRICOT_LINE, hatch="/",
                                      linewidth=.3, label="Role-wide"),
                                Patch(facecolor=BLUE, label="MinimalTool")],
                         loc="upper center", bbox_to_anchor=(.5, 1.190), ncol=2,
                         borderaxespad=0, handlelength=1.4, handletextpad=.5,
                         columnspacing=1.5, fontsize=7.7)
    for ax in (a, b):
        ax.set(xlim=(-.55, 3.55), ylim=(0, 110), yticks=[0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=5, labelsize=7.7)
        ax.tick_params(axis="y", length=2.3, pad=3, labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(.6)
        ax.spines["bottom"].set_linewidth(.6)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():
            continue
        box = t.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            clipped.append(t.get_text())
    assert not clipped, clipped
    boxes = [t.get_window_extent(renderer) for t in labels]
    assert not any(p.overlaps(q) for i, p in enumerate(boxes) for q in boxes[i+1:]), "Numeric label overlap"
    for heading, legend in zip(headings, [legend_a, legend_b]):
        assert not heading.get_window_extent(renderer).overlaps(legend.get_window_extent(renderer)), "Heading/legend overlap"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)
    payload = {
        "figure": STEM, "dimensions_inches": [7.16, 2.34],
        "sources": [{"path": p.relative_to(ROOT).as_posix(),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "retrieval": values, "retrieval_method_order": methods,
        "execution": exec_values, "execution_metric_order": metric_keys,
        "n_general": 1773, "n_relation": 432, "n_execution_requests": 120,
        "source_coverage_eligible_slots": 41,
        "encoding": {"a_bars": "Top-1", "a_lines": "Recall@5", "a_colors": "query track",
                     "b_bars": "endpoint percentages", "b_colors": "tool-exposure policy"},
        "qa": {"all_24_values_equal_original_figure": True, "clipped_text": [],
               "numeric_label_overlap": False, "heading_legend_overlap": False},
        "boundaries": ["Independent retrieval and fixed-generator tool-exposure evaluations.",
                       "Not a CARE-OPD post-training comparison.",
                       "No category interpolation, smoothing, new evaluation, or invented uncertainty.",
                       "Source coverage decrease retained, with the 41-slot denominator in the caption."]
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: all 24 plotted values match original archived sources; aligned panels; no clipping or numeric-label overlap.")


if __name__ == "__main__":
    main()
