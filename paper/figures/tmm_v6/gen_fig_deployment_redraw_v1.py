"""Compact single-column latency bars, preserving archived measurements.

Adapts the CCFA grouped-bar grammar to Matplotlib and the existing paper style.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
STEM = "fig_deployment_redraw_v1"
SOURCE = ROOT / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/02_完整统计/DEPLOYMENT_AUDIT_FULL.csv"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_deployment"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

sys.path.insert(0, str(OUT.parent / "tmm_v4"))
from figure_style import apply_style

METHODS = ["clean_sft", "gkd_full", "care_opd"]
LABELS = ["Clean-SFT", "GKD-Full", "CARE-OPD"]
BLUE, APRICOT, APRICOT_EDGE = "#6C94C7", "#E0B38D", "#A6744A"
INK, GRID, AXIS = "#20314B", "#E7ECF2", "#8492A2"


def main():
    archived = json.loads((OUT / "fig_deployment.data.json").read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert source_hash == archived["source"]["sha256"]
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [next(row for row in source_rows if row["model"] == method) for method in METHODS]
    median = np.array([float(row["p50_s"]) for row in rows])
    p95 = np.array([float(row["p95_s"]) for row in rows])
    for row, old, med, tail in zip(rows, archived["rows"], median, p95):
        assert row["model"] == old["model"]
        assert med == old["median_s"] and tail == old["p95_s"]
        assert int(row["n"]) == old["n"] == 200
        assert 0 <= med <= tail <= 40
    assert [f"{v:.2f}" for v in median] == ["13.80", "20.94", "12.65"]
    assert [f"{v:.2f}" for v in p95] == ["19.93", "34.43", "18.35"]

    apply_style()
    plt.rcParams.update({
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": AXIS, "axes.grid": False,
        "savefig.bbox": None, "svg.hashsalt": STEM,
    })
    fig = plt.figure(figsize=(3.49, 1.90), dpi=450)
    ax = fig.add_axes([.140, .140, .840, .830])
    x = np.arange(3)
    bars_median = ax.bar(x - .168, median, .285, color=BLUE,
                         edgecolor=BLUE, linewidth=.45, label="Median", zorder=3)
    bars_p95 = ax.bar(x + .168, p95, .285, color=APRICOT,
                     edgecolor=APRICOT_EDGE, linewidth=.45, hatch="//", label="P95", zorder=3)
    labels = []
    all_bars = [*bars_median, *bars_p95]
    for bar in all_bars:
        value = bar.get_height()
        labels.append(ax.text(bar.get_x() + bar.get_width()/2, value + .85,
                              f"{value:.2f}", ha="center", va="bottom", fontsize=7.6,
                              color=INK, zorder=5))
    ax.set(xlim=(-.55, 2.55), ylim=(0, 40), yticks=[0, 10, 20, 30, 40],
           ylabel="Model-call latency (s)")
    ax.set_xticks(x, LABELS)
    ax.yaxis.labelpad = 3.0
    ax.tick_params(axis="x", length=0, pad=4, labelsize=7.8)
    ax.tick_params(axis="y", length=2.0, pad=3, labelsize=7.5)
    ax.get_xticklabels()[-1].set_color("#0072B2")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.grid(axis="y", color=GRID, linewidth=.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.6)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(.012, .996),
                       borderaxespad=0, ncol=1, fontsize=7.7,
                       handlelength=1.30, handletextpad=.5, labelspacing=.32)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
                clipped.append(text.get_text())
    assert not clipped, clipped
    boxes = [text.get_window_extent(renderer) for text in labels]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:]), "Label overlap"
    legend_box = legend.get_window_extent(renderer)
    assert not any(box.overlaps(legend_box) for box in boxes), "Legend overlaps values"
    assert not any(bar.get_window_extent(renderer).overlaps(legend_box) for bar in all_bars), "Legend overlaps bars"
    assert not any(text_box.overlaps(bar.get_window_extent(renderer)) for text_box in boxes for bar in all_bars), "Values overlap bars"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)
    payload = {
        "figure": STEM, "dimensions_inches": [3.49, 1.90],
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": source_hash},
        "rows": rows, "median_s": median.tolist(), "p95_s": p95.tolist(),
        "display_decimals": 2, "sample_size_per_checkpoint": 200,
        "style": {"font": "Arial", "minimum_font_pt": 7.5, "png_dpi": 450,
                  "median": "blue bars", "p95": "apricot bars with diagonal hatching",
                  "axis_bounds": [.140, .140, .840, .830], "axis_range_s": [0, 40]},
        "qa": {"source_hash_matches_original": True, "all_six_values_unchanged": True,
               "clipped_text": [], "overlapping_numeric_labels": False,
               "legend_intersects_bars_or_values": False, "labels_intersect_bars": False,
               "original_axes_area_fraction": .761 * .65, "new_axes_area_fraction": .84 * .83},
        "caption": "Median and P95 generator-call latency on 200 requests per checkpoint on one A800 GPU. Retrieval, tools, memory, and confirmation are excluded.",
        "boundaries": ["P95 is an empirical latency quantile, not a confidence limit.",
                       "These are model-call timings, not end-to-end agent latency.",
                       "Training-method labels identify the evaluated checkpoints.",
                       "Output-length differences prevent interpreting latency as decoder throughput."]
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS: all six values preserved; no clipping/collisions; exported PDF, SVG, PNG.")


if __name__ == "__main__":
    main()
