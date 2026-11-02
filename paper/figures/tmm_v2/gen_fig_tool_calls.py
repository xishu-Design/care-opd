#!/usr/bin/env python3
"""Mean tool-call counts from the archived fixed-model Agent comparison.

Contract: a single-column grouped bar chart, not a stack; show the observed
zero overcall count explicitly. No confidence intervals are inferred.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from gen_agent_figures import OUT, REPO, STAT, BLUE, GREEN, ORANGE, INK, style
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np


def main() -> None:
    source = STAT / "AGENT_TOOL_CONTRACT_FULL.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        records = {row["endpoint"]: row for row in csv.DictReader(handle)}
    fields = ["mean_tool_calls", "mean_over_call"]
    configs = ["legacy_role_wide", "minimal_tool"]
    data = [{"configuration": config,
             "total_calls_per_request": float(records[fields[0]][config]),
             "overcalls_per_request": float(records[fields[1]][config])}
            for config in configs]
    assert [r["total_calls_per_request"] for r in data] == [4.475, 0.95]
    assert [r["overcalls_per_request"] for r in data] == [3.525, 0.0]
    style()
    fig = plt.figure(figsize=(3.49, 2.25), dpi=300)
    ax = fig.add_axes([0.180, 0.200, 0.785, 0.615])
    for x, row in zip([0, 1], data):
        for offset, key, hatch in [(-0.18, "total_calls_per_request", None),
                                    (0.18, "overcalls_per_request", "///")]:
            value = row[key]
            color = BLUE if key == "total_calls_per_request" else ORANGE
            ax.bar(x + offset, value, width=0.32, color=color,
                   edgecolor=INK if hatch else "white", linewidth=0.4, hatch=hatch, zorder=3)
            label = "0" if value == 0 else f"{value:.3f}"
            ax.text(x + offset, value + 0.10, label, ha="center", va="bottom",
                    fontsize=8, color=INK)
            if value == 0:
                ax.plot(x + offset, 0, marker="_", markersize=8, color=color,
                        markeredgewidth=1.1, linestyle="none", zorder=5, clip_on=False)
    ax.set_xlim(-0.60, 1.60)
    ax.set_ylim(0, 5.05)
    ax.set_xticks([0, 1], ["Role-wide", "MinimalTool"])
    ax.get_xticklabels()[1].set_color(GREEN)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel("Mean calls per request", labelpad=3)
    ax.tick_params(axis="both", length=2.5, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.text(0.035, 0.970, "Fixed model (n=120)", va="top", fontsize=8.5)
    fig.legend(handles=[Patch(facecolor=BLUE, edgecolor="white", label="Total calls"),
                        Patch(facecolor=ORANGE, edgecolor=INK, linewidth=0.4,
                              hatch="///", label="Overcalls")],
               loc="upper right", bbox_to_anchor=(0.970, 0.922), ncol=2,
               borderaxespad=0, handlelength=1.05, handletextpad=0.45,
               columnspacing=1.0, fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
                clipped.append(item.get_text())
    assert not clipped, clipped
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"fig_tool_calls.{extension}", dpi=300)
    plt.close(fig)
    payload = {
        "schema": "tmm-fixed-model-tool-calls-v1", "figure": "fig_tool_calls",
        "size_inches": [3.49, 2.25], "font": "Times New Roman", "min_font_pt": 8,
        "sources": [{"path": source.relative_to(REPO).as_posix(),
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "endpoint_fields": fields, "configuration_fields": configs}],
        "n_requests": 120, "n_scope": "archived AgentBench v4 hidden fixed-model comparison",
        "unit": "mean calls per request", "data": data,
        "visual_encoding": "Grouped original metrics; zero overcalls explicitly labeled at baseline.",
        "color_encoding": {"total_calls_per_request": BLUE,
                           "overcalls_per_request": ORANGE,
                           "overcall_hatch": "///", "minimal_tool_tick_label": GREEN},
        "confidence_intervals": None,
        "claim_boundary": "Fixed-generator tool-exposure comparison; not a CARE-OPD training effect.",
        "checks": {"source_values_match": True, "zero_preserved": True,
                   "no_text_clipped": True, "no_new_inference": True},
    }
    (OUT / "fig_tool_calls.source.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote fig_tool_calls.pdf/png/svg/source.json; 3.49 x 2.25 inches.")


if __name__ == "__main__":
    main()
