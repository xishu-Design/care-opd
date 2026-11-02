"""Single-column call-burden and endpoint-change figure from archived results.

Adapted grouped-bar grammar from the CCFA plotting recipes. Panel (b) uses
signed horizontal bars, not paired dots or a heatmap. No new inference runs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text

from figure_style import apply_style

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
STAT = ROOT / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/02_完整统计"
STEM = "fig_tool_contract_v3"
BASE, BLUE = "#E6B18A", "#6C94C7"
BASE_INK, BLUE_INK = "#9C5B2A", "#2B5C97"
INK, GRID, AXIS = "#20314B", "#E7ECF2", "#8492A2"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["endpoint"]: row for row in csv.DictReader(handle)}


def main():
    paths = [STAT / filename for filename in (
        "AGENT_TOOL_CONTRACT_FULL.csv", "AGENT_PAIRED_STATISTICS.csv",
        "AGENT_SECONDARY_ENDPOINTS.csv")]
    full, paired, secondary = [read(path) for path in paths]
    names = ["Total calls", "Overcalls"]
    keys = ["mean_tool_calls", "mean_over_call"]
    counts = [[float(full[key][c]) for c in ("legacy_role_wide", "minimal_tool")]
              for key in keys]
    assert counts == [[4.475, .95], [3.525, 0.0]]

    endpoints = []
    for name, key in [("Tool Exact", "tool_exact"),
                      ("Task success", "task_success"),
                      ("Source coverage", "evidence_source_coverage")]:
        if key in paired:
            row = paired[key]
            values = [100 * int(row[f"{p}_numerator"]) / int(row[f"{p}_denominator"])
                      for p in ("baseline", "candidate")]
            delta = float(row["delta"])
            ci = [float(row["ci95_low"]), float(row["ci95_high"])]
        else:
            row = secondary[key]
            values = [100 * int(row[c].split("/")[0]) / int(row[c].split("/")[1])
                      for c in ("role_wide_baseline", "minimal_tool")]
            delta = round(values[1] - values[0], 2)
            ci = None
        assert abs(delta - (values[1] - values[0])) <= .005
        assert all(abs(value - 100 * float(full[key][col])) < .005
                   for value, col in zip(values, ("legacy_role_wide", "minimal_tool")))
        endpoints.append(dict(name=name, key=key, values_percent=values,
                              delta_pp=delta, ci95_pp=ci))

    apply_style()
    plt.rcParams.update({
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": AXIS, "axes.grid": False,
        "savefig.bbox": None, "svg.hashsalt": STEM, "font.size": 8,
    })
    fig = plt.figure(figsize=(3.49, 3.20), dpi=450)
    a = fig.add_axes([.155, .575, .810, .295])
    b = fig.add_axes([.312, .118, .653, .282])
    headings = [
        fig.text(.018, .950, "(a)  Tool-call burden", fontsize=8.2, weight="bold", va="baseline"),
        fig.text(.018, .457, "(b)  Changes in selection and outcomes", fontsize=8.2, weight="bold", va="baseline"),
    ]
    legend = fig.legend(handles=[Patch(facecolor=BASE, label="Role-wide"),
                                Patch(facecolor=BLUE, label="MinimalTool")],
                        loc="upper right", bbox_to_anchor=(.98, .991), ncol=2,
                        borderaxespad=0, handlelength=1, handleheight=.8,
                        handletextpad=.35, columnspacing=.75, fontsize=7.0)
    labels = []
    for j, pair in enumerate(counts):
        for value, dx, color, ink in zip(pair, [-.18, .18], [BASE, BLUE], [BASE_INK, BLUE_INK]):
            a.bar(j + dx, value, width=.30, facecolor=color, edgecolor="none", zorder=3)
            if value == 0:
                a.plot(j + dx, 0, marker="_", color=BLUE_INK, markersize=8,
                       markeredgewidth=1.15, clip_on=False, zorder=5)
            labels.append(a.text(j + dx, value + .10, f"{value:.3f}" if value else "0",
                                 ha="center", va="bottom", fontsize=8.0, color=ink))
    a.set(xlim=(-.56, 1.56), ylim=(0, 5.05), xticks=[0, 1], yticks=[0, 2, 4],
          ylabel="Mean calls / request")
    a.set_xticklabels(names, fontsize=8)
    a.yaxis.labelpad = 3
    a.grid(axis="y", color=GRID, linewidth=.45, zorder=0)
    a.set_axisbelow(True)
    a.tick_params(axis="x", length=0, pad=4)
    a.tick_params(axis="y", length=2.3, pad=3)

    for y, row in zip([2, 1, 0], endpoints):
        delta, ci = row["delta_pp"], row["ci95_pp"]
        color = BLUE if delta >= 0 else BASE
        b.barh(y, delta, height=.46, color=color, linewidth=0, zorder=3)
        if ci is not None:
            b.errorbar(delta, y, xerr=[[delta-ci[0]], [ci[1]-delta]],
                       fmt="none", ecolor=INK, elinewidth=.9, capsize=2.5,
                       capthick=.85, zorder=4)
        if delta > 20:
            tx, color, ha = delta * .49, INK, "center"
        else:
            tx, color, ha = (10.5 if delta >= 0 else 4), INK, "left"
        text = f"{delta:+.2f}".replace("-", "\N{MINUS SIGN}")
        labels.append(b.text(tx, y, text, ha=ha, va="center", fontsize=8.1,
                             color=color, weight="bold" if y == 2 else "normal", zorder=5))
    b.set(xlim=(-18, 93), ylim=(-.54, 2.54), xticks=[0, 25, 50, 75],
          yticks=[2, 1, 0], xlabel="Change from Role-wide (pp)")
    b.set_yticklabels([row["name"] for row in endpoints], fontsize=7.7)
    b.axvline(0, color=AXIS, lw=.85, zorder=2)
    b.grid(axis="x", color=GRID, lw=.45, zorder=0)
    b.set_axisbelow(True)
    b.tick_params(axis="y", length=0, pad=4)
    b.tick_params(axis="x", length=2.3, pad=3)
    b.xaxis.labelpad = 4
    b.spines["left"].set_visible(False)
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
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
    assert abs(headings[0].get_window_extent(renderer).x0 - headings[1].get_window_extent(renderer).x0) < .1
    assert not headings[0].get_window_extent(renderer).overlaps(legend.get_window_extent(renderer)), "Heading overlaps legend"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)
    payload = {
        "figure": STEM, "n_requests": 120, "fixed_generator": True,
        "source_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "count_metrics": dict(zip(keys, counts)), "endpoints": endpoints,
        "required_recall_percent": [96.49, 98.25],
        "style": {"font": "Arial", "dimensions_inches": [3.49, 3.20],
                  "role_wide": BASE, "minimal_tool": BLUE, "png_dpi": 450},
        "qa": {"clipped_text": [], "data_label_overlap": False, "panel_headings_aligned": True,
               "metrics_match_archived_source": True, "legend_heading_overlap": False},
        "boundaries": ["No disjoint call partition is assumed.",
                       "Only Tool Exact and task success have displayed paired confidence intervals.",
                       "No interval is fabricated for source coverage.",
                       "Endpoint denominators differ and are not averaged.",
                       "This compares tool-exposure policies with a fixed generator, not post-training methods."],
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: source values and paired CIs; aligned headings; no clipped or overlapping labels; PDF/SVG/PNG exported.")


if __name__ == "__main__":
    main()
