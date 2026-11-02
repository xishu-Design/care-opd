"""Single-column grouped counts and Tool Exact comparison, from archived data."""
from __future__ import annotations

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from figure_style import apply_style
from gen_fig_tool_contract_v3 import OUT, ROOT, STAT, read, BASE, BLUE, INK, GRID, AXIS

STEM = "fig_tool_contract_v5"
EXACT = "#287D6A"
HATCH = "#AF794D"


def main():
    paths = [STAT / name for name in ("AGENT_TOOL_CONTRACT_FULL.csv", "AGENT_PAIRED_STATISTICS.csv", "AGENT_SECONDARY_ENDPOINTS.csv")]
    full, paired, secondary = [read(path) for path in paths]
    columns = ("legacy_role_wide", "minimal_tool")
    total = [float(full["mean_tool_calls"][c]) for c in columns]
    over = [float(full["mean_over_call"][c]) for c in columns]
    row = paired["tool_exact"]
    exact = [100 * int(row[f"{p}_numerator"]) / int(row[f"{p}_denominator"])
             for p in ("baseline", "candidate")]
    assert total == [4.475, .95] and over == [3.525, 0]
    assert [f"{v:.2f}" for v in exact] == ["21.67", "98.33"]

    apply_style()
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": AXIS, "axes.grid": False,
                         "savefig.bbox": None, "svg.hashsalt": STEM,
                         "hatch.linewidth": .42})
    fig = plt.figure(figsize=(3.49, 2.10), dpi=450)
    ax = fig.add_axes([.146, .133, .704, .724])
    rate = ax.twinx()
    labels = []
    bars = []
    for i in range(2):
        for offset, value, color, hatch in [(-.25, total[i], BLUE, None),
                                            (.25, over[i], BASE, "/")]:
            container = ax.bar(i + offset, value, width=.19, color=color,
                               edgecolor=HATCH if hatch else "none",
                               linewidth=.4 if hatch else 0, hatch=hatch, zorder=3)
            bars.append(container[0])
            if not value:
                ax.plot(i + offset, 0, marker="_", markersize=7.5,
                        markeredgewidth=1.2, color=HATCH, clip_on=False, zorder=5)
            labels.append(ax.text(i + offset, value + .105, f"{value:.3f}" if value else "0",
                                  fontsize=8.0, ha="center", va="bottom", color=INK, zorder=6))

    ax.set(xlim=(-.54, 1.54), ylim=(0, 5.25), xticks=[0, 1], yticks=[0, 1, 2, 3, 4, 5],
           ylabel="Calls per request")
    ax.set_xticklabels(["Role-wide", "MinimalTool"], fontsize=8.3)
    ax.yaxis.labelpad = 3
    ax.grid(axis="y", color=GRID, linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=2.1, pad=2.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    rate.set(ylim=(0, 105), yticks=[0, 25, 50, 75, 100], ylabel="Tool Exact (%)")
    rate.plot([0, 1], exact, color=EXACT, linewidth=1.45, linestyle=(0, (3.2, 2.1)),
              marker="o", markersize=5.0, markerfacecolor="white",
              markeredgewidth=1.35, zorder=8, clip_on=False)
    for label_x, y, label_y in [(0, exact[0], 12.5), (1.13, exact[1], 86.0)]:
        labels.append(rate.text(label_x, label_y, f"{y:.2f}", ha="center", va="center",
                                 color=EXACT, fontsize=8.0, weight="bold", zorder=9))
    rate.tick_params(axis="y", colors=EXACT, length=2.1, pad=2.6)
    rate.yaxis.label.set_color(EXACT)
    rate.yaxis.labelpad = 3
    rate.spines["right"].set_color(EXACT)
    rate.spines["right"].set_linewidth(.6)
    for side in ("top", "left", "bottom"):
        rate.spines[side].set_visible(False)

    legend = fig.legend(handles=[Patch(facecolor=BLUE, label="Total calls"),
                                Patch(facecolor=BASE, edgecolor=HATCH, linewidth=.4,
                                      hatch="/", label="Overcalls"),
                                Line2D([], [], color=EXACT, ls=(0, (3.2, 2.1)),
                                       marker="o", markersize=4.4, markerfacecolor="white",
                                       markeredgewidth=1.1, lw=1.25, label="Tool Exact")],
                        loc="upper center", bbox_to_anchor=(.498, .997), ncol=3,
                        borderaxespad=0, handlelength=1.5, handleheight=.8,
                        handletextpad=.45, columnspacing=.9, fontsize=7.7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
                clipped.append(text.get_text())
    assert not clipped, clipped
    boxes = [label.get_window_extent(renderer) for label in labels]
    assert not any(box0.overlaps(box1) for i, box0 in enumerate(boxes) for box1 in boxes[i+1:]), "Overlapping numeric labels"
    assert not any(box.overlaps(legend.get_window_extent(renderer)) for box in boxes), "Legend overlaps values"
    # Percentage labels must sit in the open central gap, not cut count bars.
    assert not any(label.get_window_extent(renderer).overlaps(bar.get_window_extent(renderer))
                   for label in labels[4:] for bar in bars), "Percentage text overlaps a bar"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)

    payload = {
        "figure": STEM, "n_requests": 120, "fixed_generator": True,
        "source_files": [{"path": p.relative_to(ROOT).as_posix(),
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "total_calls_per_request": total, "overcalls_per_request": over,
        "tool_exact_percent": exact,
        "caption_pair_delta": {k: row[k] for k in ("delta", "ci95_low", "ci95_high")},
        "caption_source_coverage": secondary["evidence_source_coverage"],
        "style": {"dimensions_inches": [3.49, 2.10], "font": "Arial", "png_dpi": 450,
                  "calls_left_axis_limits": [0, 5.25], "percent_right_axis_limits": [0, 105]},
        "qa": {"all_values_source_checked": True, "clipped_labels": [],
               "numeric_label_overlap": False, "percent_labels_overlap_bars": False},
        "boundaries": ["Two vertical scales have different units; mark heights must not be compared across scales.",
                       "The dashed connector compares two categories, not continuous training progression.",
                       "No independent uncertainty interval is inferred from the paired difference interval.",
                       "Source-coverage decrease is retained in the caption."]
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: six values source-checked, annotations clear of bars and legend, no clipping; PDF/SVG/PNG exported.")


if __name__ == "__main__":
    main()
