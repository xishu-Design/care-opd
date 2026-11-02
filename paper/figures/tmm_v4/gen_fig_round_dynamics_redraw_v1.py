"""Compact stacked counts with same-axis new-failure diamonds.

Uses the existing CCFA-derived bar composition workflow and paper style.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
STEM = "fig_round_dynamics_redraw_v1"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_agent"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from figure_style import apply_style

BLUE, APRICOT, APRICOT_EDGE = "#6C94C7", "#E0B38D", "#A6744A"
GREEN, INK, GRID, AXIS = "#227D68", "#20314B", "#E7ECF2", "#8492A2"


def load_data():
    archived = json.loads((OUT / "fig_round_dynamics.data.json").read_text(encoding="utf-8"))
    for source in archived["source_files"]:
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    csv_path = ROOT / next(s["path"] for s in archived["source_files"] if s["path"].endswith(".csv"))
    json_path = ROOT / next(s["path"] for s in archived["source_files"] if s["path"].endswith(".json"))
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    detail = json.loads(json_path.read_text(encoding="utf-8-sig"))
    result = []
    for row in rows:
        if row["row_type"] != "round":
            continue
        record = {key: int(row[key]) for key in ["round", "failed", "semantic_failures", "format_failures", "rollouts", "passed"]}
        assert record["semantic_failures"] + record["format_failures"] == record["failed"]
        assert record["failed"] + record["passed"] == record["rollouts"] == 12000
        raw = next(r["recomputed"] for r in detail["rounds"] if int(r["label"]) == record["round"])
        assert all(record[key] == raw[key] for key in record if key != "round")
        transition = next((r for r in rows if r["row_type"] == "transition" and r["failure_class"] == "all" and int(r["to"]) == record["round"]), None)
        record["new_failures_vs_previous"] = int(transition["new"]) if transition else None
        record["persistent_failures_vs_previous"] = int(transition["persistent"]) if transition else None
        if transition:
            assert record["new_failures_vs_previous"] + record["persistent_failures_vs_previous"] == record["failed"]
        result.append(record)
    assert result == archived["data"]
    assert [r["failed"] for r in result] == [3350, 2996, 2731]
    assert [r["semantic_failures"] for r in result] == [3256, 2668, 2677]
    assert [r["format_failures"] for r in result] == [94, 328, 54]
    assert [r["new_failures_vs_previous"] for r in result] == [None, 2737, 1490]
    return result, archived


def main():
    rows, archived = load_data()
    apply_style()
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": AXIS, "axes.grid": False,
                         "savefig.bbox": None, "svg.hashsalt": STEM})
    fig = plt.figure(figsize=(3.49, 1.90), dpi=450)
    ax = fig.add_axes([.137, .145, .845, .825])
    xs = [1, 2, 3]
    semantic = [r["semantic_failures"] for r in rows]
    formats = [r["format_failures"] for r in rows]
    width = .65
    ax.bar(xs, semantic, width=width, color=BLUE, edgecolor="white", linewidth=.45, zorder=3)
    ax.bar(xs, formats, width=width, bottom=semantic, color=APRICOT,
           edgecolor=APRICOT_EDGE, hatch="///", linewidth=.45, zorder=3)
    value_labels, new_labels, total_labels = [], [], []
    markers = []
    for x, row in zip(xs, rows):
        sem, form, total = row["semantic_failures"], row["format_failures"], row["failed"]
        value_labels.append(ax.text(x, sem * .30, f"{sem:,}", ha="center", va="center", fontsize=7.8, color="#172638", zorder=5))
        t = ax.text(x, total + 105, f"{total:,}", ha="center", va="bottom", fontsize=7.9, fontweight="bold", zorder=5)
        total_labels.append(t)
        value_labels.append(t)
        cap_y = sem + form / 2
        ax.plot([x + width / 2, x + .395], [cap_y, cap_y], color=APRICOT_EDGE, lw=.55, zorder=5)
        value_labels.append(ax.text(x + .425, cap_y, f"{form:,}", ha="left", va="center", fontsize=7.5, zorder=5))
        new = row["new_failures_vs_previous"]
        if new is not None:
            marker, = ax.plot([x], [new], marker="D", linestyle="none", markersize=5.6,
                              markerfacecolor=GREEN, markeredgecolor="white", markeredgewidth=.8, zorder=6)
            markers.append(marker)
            offset = -13.5 if x == 2 else 10.0
            t = ax.annotate(f"{new:,}", (x, new), xytext=(0, offset), textcoords="offset points",
                            ha="center", va="center", color=GREEN, fontsize=7.6,
                            bbox={"boxstyle": "round,pad=0.15,rounding_size=0.10", "facecolor": "white", "edgecolor": "none"}, zorder=7)
            new_labels.append(t)
            value_labels.append(t)
    ax.set(xlim=(.48, 3.71), ylim=(0, 4050), ylabel="Failed rollouts",
           yticks=[0, 1000, 2000, 3000, 4000])
    ax.set_yticklabels(["0", "1k", "2k", "3k", "4k"])
    ax.set_xticks(xs, ["Round 1", "Round 2", "Round 3"])
    ax.yaxis.labelpad = 3.0
    ax.tick_params(axis="x", length=0, pad=4, labelsize=7.8)
    ax.tick_params(axis="y", length=2.0, pad=3, labelsize=7.5)
    ax.grid(axis="y", color=GRID, linewidth=.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.6)
    legend = ax.legend(handles=[Patch(facecolor=BLUE, label="Semantic"),
                               Patch(facecolor=APRICOT, edgecolor=APRICOT_EDGE, hatch="///", label="Format"),
                               Line2D([], [], marker="D", color=GREEN, markersize=4.3, linestyle="none", label="New")],
                       loc="upper center", bbox_to_anchor=(.51, 1.003), ncol=3,
                       borderaxespad=0, handlelength=1.15, handletextpad=.45,
                       columnspacing=.85, fontsize=7.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
                clipped.append(text.get_text())
    assert not clipped, clipped
    boxes = [text.get_window_extent(renderer) for text in value_labels]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:]), "Numeric labels overlap"
    legend_box = legend.get_window_extent(renderer)
    assert not any(box.overlaps(legend_box) for box in boxes), "Legend overlaps values"
    assert not any(marker.get_window_extent(renderer).overlaps(box) for marker in markers for box in boxes), "Marker overlaps numeric label"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)
    payload = {
        "figure": STEM, "dimensions_inches": [3.49, 1.90],
        "source_files": archived["source_files"], "data": rows,
        "definitions": archived["definitions"],
        "encoding": {"stacked_bars": {"semantic": BLUE, "format": APRICOT, "format_hatch": "///"},
                     "diamond": "new ALL-CLASS failures vs immediately previous round; same count axis; not a third stack segment",
                     "total": "bold numbers above bar", "semantic_count": "inside blue segment", "format_count": "short right-hand callout"},
        "style": {"font": "Arial", "minimum_font_pt": 7.5, "png_dpi": 450, "axes_bounds": [.137, .145, .845, .825]},
        "qa": {"source_hashes_verified": True, "all_11_displayed_counts_unchanged": True,
               "clipped_text": [], "numeric_label_overlaps": False, "marker_label_overlaps": False,
               "legend_value_overlaps": False, "old_axes_area_fraction": .775*.64, "new_axes_area_fraction": .845*.825},
        "caption": "Failure composition across three 12,000-rollout collections. Diamonds mark failures absent from the preceding collection.",
        "claim_limits": ["No new-by-class partition is inferred from all-class transition counts.",
                         "Descriptive collection diagnostics, not causal evidence of online resampling or a round-count ablation.",
                         "No invented uncertainty, smoothing, or interpolation."]
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS: all 11 counts unchanged, no clipped or colliding labels; PDF/SVG/PNG exported.")


if __name__ == "__main__":
    main()
