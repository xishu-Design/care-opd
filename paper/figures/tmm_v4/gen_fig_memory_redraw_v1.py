"""Compact controlled-memory bar/line plot from archived aggregate results."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
SOURCE = ROOT / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/02_完整统计/MEMORY_BENCHMARK_FULL.csv"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_agent"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as PlotPath
from matplotlib.text import Text
import numpy as np

from figure_style import apply_style

STEM = "fig_memory_redraw_v1"
BLUE, GREEN = "#6C94C7", "#287D6A"
INK, GRID, AXIS = "#20314B", "#E7ECF2", "#8492A2"


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if "controlled" in r["track"]]
    previous = json.loads((OUT / "fig_memory.source.json").read_text(encoding="utf-8"))
    assert previous["data"]["rows"] == rows
    accuracy = [100 * float(r["judge_accuracy"]) for r in rows]
    recall = [100 * float(r["recall_at_5"]) for r in rows]
    assert [f"{v:.1f}" for v in accuracy] == ["10.0", "18.4", "47.6", "49.4"]
    assert [f"{v:.2f}" for v in recall] == ["7.48", "13.74", "91.99", "93.00"]

    apply_style()
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": AXIS, "axes.grid": False,
                         "savefig.bbox": None, "svg.hashsalt": STEM})
    fig = plt.figure(figsize=(3.49, 2.05), dpi=450)
    ax = fig.add_axes([.143, .202, .842, .655])
    x = np.arange(4)
    bar_x, line_x = x - .16, x + .30
    bars = ax.bar(bar_x, accuracy, width=.43, color=BLUE, edgecolor="none", zorder=3)
    ax.plot(line_x, recall, color=GREEN, linestyle=(0, (3.3, 2.1)), linewidth=1.35,
            marker="D", markersize=4.0, markerfacecolor="white", markeredgewidth=1.2, zorder=5)
    value_labels = []
    for xp, value in zip(bar_x, accuracy):
        value_labels.append(ax.text(xp, value + 4.0 if value < 25 else value - 11.0, f"{value:.1f}", ha="center",
                                     va="center", color=INK, fontsize=7.6, zorder=6))
    recall_labels = []
    for i, (xp, value) in enumerate(zip(line_x, recall)):
        offset = 7.0 if i == 0 else (-7.0 if i == 1 else 6.0)
        t = ax.text(xp, value + offset, f"{value:.2f}", ha="center",
                    va="center", color=GREEN, fontsize=7.5, zorder=6)
        value_labels.append(t)
        recall_labels.append(t)
    ax.set(xlim=(-.66, 3.65), ylim=(0, 110), yticks=[0, 25, 50, 75, 100], ylabel="Score (%)")
    ax.set_xticks(x, ["Sliding\nwindow", "Summary\n+ recent", "Mem0", "Mem0 +\nGraphiti"])
    ax.yaxis.labelpad = 3
    ax.tick_params(axis="x", length=0, pad=5, labelsize=7.7)
    ax.tick_params(axis="y", length=2.1, pad=3, labelsize=7.5)
    ax.grid(axis="y", color=GRID, linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.6)
    legend = fig.legend(handles=[Patch(facecolor=BLUE, label="Judge accuracy"),
                                Line2D([], [], color=GREEN, linestyle=(0, (3.3, 2.1)), lw=1.35,
                                       marker="D", markersize=3.8, markerfacecolor="white",
                                       markeredgewidth=1.1, label="Evidence R@5")],
                        loc="upper center", bbox_to_anchor=(.55, .994), ncol=2,
                        borderaxespad=0, handlelength=1.6, columnspacing=1.05,
                        handletextpad=.45, fontsize=7.7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            box = t.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
                clipped.append(t.get_text())
    assert not clipped, clipped
    boxes = [t.get_window_extent(renderer) for t in value_labels]
    assert not any(p.overlaps(q) for i, p in enumerate(boxes) for q in boxes[i+1:]), "Numeric label overlap"
    assert not any(p.overlaps(legend.get_window_extent(renderer)) for p in boxes), "Values overlap legend"
    assert not any(t.get_window_extent(renderer).overlaps(bar.get_window_extent(renderer))
                   for t in recall_labels for bar in bars), "Recall labels overlap bars"
    line_path = PlotPath(ax.transData.transform(np.column_stack([line_x, recall])))
    assert not any(line_path.intersects_bbox(box, filled=False) for box in boxes), "Line intersects a value label"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)
    payload = {
        "figure": STEM, "dimensions_inches": [3.49, 2.05],
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(),
                   "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "data": rows, "judge_accuracy_percent": accuracy, "evidence_recall_at_5_percent": recall,
        "n_questions": 500, "judge": "project-specific, non-official",
        "style": {"font": "Arial", "minimum_text_points": 7.5, "png_dpi": 450,
                  "accuracy_encoding": "blue bars", "recall_encoding": "green dashed line, hollow diamonds"},
        "qa": {"all_eight_values_match_prior_figure": True, "clipped_text": [],
               "numeric_label_overlap": False, "recall_labels_overlap_bars": False,
               "line_intersects_value_label": False},
        "boundaries": ["Only controlled rows are used; isolated real-component rows are excluded.",
                       "A fixed-generator memory comparison, not a CARE-OPD post-training effect.",
                       "Categorical connectors are comparison aids, not a time series.",
                       "No invented uncertainty or smoothing."]
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: all eight values unchanged; no clipped/overlapping labels; PDF, SVG, PNG exported.")


if __name__ == "__main__":
    main()
