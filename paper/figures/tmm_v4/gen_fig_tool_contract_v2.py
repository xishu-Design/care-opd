"""Source-grounded two-panel tool exposure comparison; no new inference.

Grammar: grouped counts + paired endpoint positions (CCFA slopegraph recipe
adapted to a horizontal common percentage axis). The count metrics are NOT
stacked because no disjoint tool-call partition is assumed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from figure_style import apply_style

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
STAT = ROOT / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/02_完整统计"
STEM = "fig_tool_contract_v2"
BASE = "#E6B18A"
BLUE = "#6C94C7"
BASE_INK = "#9C5B2A"
BLUE_INK = "#2B5C97"
INK = "#20314B"
GRID = "#E7ECF2"
AXIS = "#8492A2"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {r["endpoint"]: r for r in csv.DictReader(handle)}


def main():
    paths = [STAT / "AGENT_TOOL_CONTRACT_FULL.csv", STAT / "AGENT_PAIRED_STATISTICS.csv", STAT / "AGENT_SECONDARY_ENDPOINTS.csv"]
    full, paired, secondary = [read(p) for p in paths]
    counts = [
        ("Total calls", "mean_tool_calls"),
        ("Overcalls", "mean_over_call"),
    ]
    count_values = [[float(full[k][c]) for c in ("legacy_role_wide", "minimal_tool")] for _, k in counts]
    assert count_values == [[4.475, 0.95], [3.525, 0.0]]
    quality = []
    for name, key in [("Tool Exact", "tool_exact"), ("Required recall", "tool_micro_recall"), ("Task success", "task_success"), ("Source coverage", "evidence_source_coverage")]:
        if key in paired:
            row = paired[key]
            values = [100 * int(row[f"{p}_numerator"]) / int(row[f"{p}_denominator"]) for p in ("baseline", "candidate")]
        elif key == "evidence_source_coverage":
            row = secondary[key]
            values = [100 * int(row[c].split("/")[0]) / int(row[c].split("/")[1]) for c in ("role_wide_baseline", "minimal_tool")]
        else:
            values = [100 * float(full[key][c]) for c in ("legacy_role_wide", "minimal_tool")]
        expected = [100 * float(full[key][c]) for c in ("legacy_role_wide", "minimal_tool")]
        assert all(abs(a - b) < .005 for a, b in zip(values, expected))
        quality.append({"name": name, "endpoint": key, "values_percent": values})

    apply_style()
    plt.rcParams.update({
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": AXIS,
        "axes.grid": False, "savefig.bbox": None,
        "svg.hashsalt": STEM, "font.size": 8,
    })
    fig = plt.figure(figsize=(6.99, 2.48), dpi=450)
    a = fig.add_axes([.072, .160, .300, .660])
    b = fig.add_axes([.565, .160, .405, .660])
    fig.text(.035, .875, "(a)  Call burden", fontsize=8.8, weight="bold")
    fig.text(.442, .875, "(b)  Selection and task outcomes", fontsize=8.8, weight="bold")
    fig.legend(handles=[Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=BASE_INK,
                               markeredgewidth=1.15, color=BASE, linestyle="none", markersize=5.5, label="Role-wide"),
                        Line2D([], [], marker="D", markerfacecolor=BLUE, markeredgecolor=BLUE_INK,
                               markeredgewidth=.8, color=BLUE, linestyle="none", markersize=5, label="MinimalTool")],
               loc="upper center", bbox_to_anchor=(.52, 1.01), ncol=2,
               handlelength=1.4, handletextpad=.55, columnspacing=1.6, fontsize=8.0)
    labels = []
    for j, ((name, _), pair) in enumerate(zip(counts, count_values)):
        for value, dx, color, ink in zip(pair, [-.19, .19], [BASE, BLUE], [BASE_INK, BLUE_INK]):
            a.bar(j + dx, value, width=.32, facecolor=color, edgecolor="none", zorder=3)
            if value == 0:
                a.plot(j + dx, 0, marker="_", color=BLUE_INK, markersize=9,
                       markeredgewidth=1.1, clip_on=False, zorder=5)
            labels.append(a.text(j + dx, value + .11, f"{value:.3f}" if value else "0",
                                 ha="center", va="bottom", fontsize=8.2, color=ink))
    a.set(xlim=(-.6, 1.6), ylim=(0, 5.0), xticks=[0, 1], yticks=[0, 1, 2, 3, 4, 5],
          ylabel="Mean calls per request")
    a.set_xticklabels([r[0] for r in counts], fontsize=8)
    a.grid(axis="y", color=GRID, linewidth=.45, zorder=0)
    a.set_axisbelow(True)
    a.tick_params(axis="x", length=0, pad=5)
    a.tick_params(axis="y", length=2.3, pad=3)

    for i, row in enumerate(quality):
        y = 3 - i
        v0, v1 = row["values_percent"]
        # Small vertical offsets keep near-coincident markers identifiable.
        b.plot([v0, v1], [y + .04, y - .04], color="#AEBBC9", lw=1.2, zorder=2)
        b.plot(v0, y + .04, marker="o", markersize=5.2, markerfacecolor="white",
               markeredgecolor=BASE_INK, markeredgewidth=1.15, ls="none", zorder=4)
        b.plot(v1, y - .04, marker="D", markersize=4.6, markerfacecolor=BLUE,
               markeredgecolor=BLUE_INK, markeredgewidth=.8, ls="none", zorder=5)
        labels.append(b.text(v0, y + .21, f"{v0:.2f}", color=BASE_INK, ha="center", va="bottom", fontsize=7.7))
        labels.append(b.text(v1, y - .22, f"{v1:.2f}", color=BLUE_INK, ha="center", va="top", fontsize=7.7))
    b.set(xlim=(0, 107), ylim=(-.61, 3.63), xticks=[0, 25, 50, 75, 100],
          yticks=[3, 2, 1, 0], xlabel="Endpoint value (%)")
    b.set_yticklabels([r["name"] for r in quality], fontsize=8)
    b.get_yticklabels()[0].set_weight("bold")
    b.get_yticklabels()[0].set_color(BLUE_INK)
    b.grid(axis="x", color=GRID, lw=.45)
    b.set_axisbelow(True)
    b.tick_params(axis="y", length=0, pad=6)
    b.tick_params(axis="x", length=2.3, pad=3)
    b.xaxis.labelpad = 4
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_linewidth(.6)
        ax.spines["bottom"].set_linewidth(.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
                clipped.append(artist.get_text())
    assert not clipped, clipped
    boxes = [t.get_window_extent(renderer) for t in labels]
    assert not any(b0.overlaps(b1) for i, b0 in enumerate(boxes) for b1 in boxes[i + 1:]), "Data label overlap"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)
    payload = {
        "figure": STEM, "n_requests": 120, "fixed_generator": True,
        "source_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "count_metrics": {k: v for (_, k), v in zip(counts, count_values)},
        "quality_metrics": quality,
        "uncertainty_for_caption": {k: {v: paired[k][v] for v in ("delta", "ci95_low", "ci95_high", "bootstrap_replicates")} for k in ("tool_exact", "task_success")},
        "style": {"font": "Arial", "dimensions_inches": [6.99, 2.48], "role_wide": BASE, "minimal_tool": BLUE, "svg_text": True, "pdf_fonttype": 42, "png_dpi": 450},
        "boundaries": ["No stacked-call partition is inferred.", "Endpoints have different denominators and are not averaged.", "No uncertainty is invented for recall or source coverage.", "This is a fixed-generator system comparison, not a post-training effect."],
        "qa": {"data_labels_in_frame": True, "data_label_overlap": False, "metrics_match_archived_source": True},
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: 4 count values, 8 endpoint values, no clipped/overlapping labels; exported PDF, SVG, PNG, source manifest.")


if __name__ == "__main__":
    main()
