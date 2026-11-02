"""One compact single-column figure: count bars with outcome readouts."""
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
from gen_fig_tool_contract_v3 import OUT, ROOT, STAT, read, BASE, BLUE, BASE_INK, BLUE_INK, INK, GRID, AXIS

STEM = "fig_tool_contract_v4"


def main():
    paths = [STAT / name for name in ("AGENT_TOOL_CONTRACT_FULL.csv",
                                     "AGENT_PAIRED_STATISTICS.csv",
                                     "AGENT_SECONDARY_ENDPOINTS.csv")]
    full, paired, secondary = [read(p) for p in paths]
    keys = ["mean_tool_calls", "mean_over_call"]
    counts = [[float(full[k][c]) for c in ("legacy_role_wide", "minimal_tool")] for k in keys]
    assert counts == [[4.475, .95], [3.525, 0.0]]
    outcomes = []
    for name, key in [("Tool Exact (%)", "tool_exact"),
                      ("Task success (%)", "task_success"),
                      ("Required recall (%)", "tool_micro_recall"),
                      ("Source coverage (%)", "evidence_source_coverage")]:
        if key in paired:
            row = paired[key]
            rates = [100 * int(row[f"{p}_numerator"]) / int(row[f"{p}_denominator"])
                     for p in ("baseline", "candidate")]
        elif key == "evidence_source_coverage":
            row = secondary[key]
            rates = [100 * int(row[c].split("/")[0]) / int(row[c].split("/")[1])
                     for c in ("role_wide_baseline", "minimal_tool")]
        else:
            rates = [100 * float(full[key][c]) for c in ("legacy_role_wide", "minimal_tool")]
        assert all(abs(value - 100 * float(full[key][c])) < .005
                   for value, c in zip(rates, ("legacy_role_wide", "minimal_tool")))
        outcomes.append(dict(label=name, endpoint=key, rates=rates))

    apply_style()
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": AXIS, "axes.grid": False,
                         "savefig.bbox": None, "svg.hashsalt": STEM})
    fig = plt.figure(figsize=(3.49, 2.46), dpi=450)
    ax = fig.add_axes([.150, .442, .825, .448])
    legend = fig.legend(handles=[Patch(facecolor=BASE, label="Role-wide"),
                                Patch(facecolor=BLUE, label="MinimalTool")],
                        loc="upper center", bbox_to_anchor=(.56, .999), ncol=2,
                        borderaxespad=0, handlelength=1.15, handleheight=.8,
                        handletextpad=.5, columnspacing=1.7, fontsize=8.0)
    labels = []
    for j, pair in enumerate(counts):
        for value, dx, color, ink in zip(pair, [-.18, .18], [BASE, BLUE], [BASE_INK, BLUE_INK]):
            ax.bar(j + dx, value, width=.30, facecolor=color, edgecolor="none", zorder=3)
            if value == 0:
                ax.plot(j + dx, 0, marker="_", color=BLUE_INK, markersize=8,
                        markeredgewidth=1.1, clip_on=False, zorder=5)
            labels.append(ax.text(j + dx, value + .105, f"{value:.3f}" if value else "0",
                                  ha="center", va="bottom", fontsize=8.1, color=ink))
    ax.set(xlim=(-.56, 1.56), ylim=(0, 5.05), xticks=[0, 1], yticks=[0, 2, 4],
           ylabel="Mean calls per request")
    ax.set_xticklabels(["Total calls", "Overcalls"], fontsize=8.2)
    ax.yaxis.labelpad = 3
    ax.grid(axis="y", color=GRID, linewidth=.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=2.3, pad=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(.6)

    # One compact aligned readout, not a second plot or heatmap. Colors retain
    # their configuration meanings, including the adverse coverage change.
    fig.add_artist(Line2D([.032, .974], [.345, .345], transform=fig.transFigure,
                          color="#CBD5E1", lw=.65))
    fig.add_artist(Line2D([.514, .514], [.043, .315], transform=fig.transFigure,
                          color=GRID, lw=.65))
    for i, row in enumerate(outcomes):
        center = .266 if i % 2 == 0 else .750
        title_y, value_y = (.283, .211) if i < 2 else (.130, .058)
        labels.append(fig.text(center, title_y, row["label"], ha="center", va="center",
                               fontsize=7.8, color=INK))
        labels.append(fig.text(center - .052, value_y, f"{row['rates'][0]:.2f}",
                               ha="right", va="center", fontsize=8.3, color=BASE_INK))
        labels.append(fig.text(center, value_y, "\N{RIGHTWARDS ARROW}", ha="center", va="center",
                               fontsize=8.3, color=AXIS))
        labels.append(fig.text(center + .052, value_y, f"{row['rates'][1]:.2f}",
                               ha="left", va="center", fontsize=8.3, color=BLUE_INK,
                               weight="bold" if row["endpoint"] == "tool_exact" else "normal"))

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
    assert not any(b0.overlaps(b1) for i, b0 in enumerate(boxes) for b1 in boxes[i+1:]), "Label overlap"
    assert len(fig.axes) == 1
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)

    payload = {
        "figure": STEM, "n_requests": 120, "fixed_generator": True,
        "source_files": [{"path": p.relative_to(ROOT).as_posix(),
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "count_metrics": dict(zip(keys, counts)), "outcomes": outcomes,
        "caption_intervals": {k: {v: paired[k][v] for v in ("delta", "ci95_low", "ci95_high")}
                              for k in ("tool_exact", "task_success")},
        "style": {"width_inches": 3.49, "height_inches": 2.46, "axes_count": 1,
                  "font": "Arial", "minimum_text_points": 7.6, "png_dpi": 450},
        "qa": {"all_12_printed_values_match_sources": True,
               "label_overlap": False, "clipped_text": [], "subplot_labels": False},
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: 12 archived values; one plotting axis; no clipped/overlapping labels; PDF, SVG, PNG exported.")


if __name__ == "__main__":
    main()
