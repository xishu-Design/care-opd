#!/usr/bin/env python3
"""Redraw fixed-reference transitions with in-plot method/net headers.

Publication target: one IEEE column. Preserve the original CSV filter, counts,
cohort denominators, order and orange/green semantics. The plotting area replaces
the old outside method and net columns; no rate or uncertainty is invented.
Exports a sibling version so earlier manuscript figures remain unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from gen_fig_fixed_transitions import ROOT, SOURCE_REL, MODELS, plt
from figure_style import apply_style, INK, AXIS, GRID, ORANGE, GREEN, CARE
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter


OUT = Path(__file__).resolve().parent
STEM = "fig_fixed_transitions_compact"
FIGSIZE = (3.49, 2.13)


def signed(value: int) -> str:
    return ("+" if value >= 0 else "\N{MINUS SIGN}") + f"{abs(value):,}"


def load_data():
    source = ROOT / SOURCE_REL
    with source.open(encoding="utf-8-sig", newline="") as stream:
        records = {
            row["model"]: row for row in csv.DictReader(stream)
            if row["record_type"] == "transition"
            and row["state_or_slice"] == "vs_Clean-SFT_all_four_states"
        }
    reference = records["Clean-SFT"]
    assert int(reference["successes"]) == 2756
    assert int(reference["failures"]) == 1144
    data = []
    for model in MODELS:
        row = records[model]
        item = {"model": model, **{key: int(row[key]) for key in (
            "n", "successes", "failures", "reference_failures", "rescued",
            "reference_successes", "regressed", "net_success_change"
        )}}
        assert item["n"] == 3900
        assert item["reference_successes"] == 2756
        assert item["reference_failures"] == 1144
        assert item["rescued"] - item["regressed"] == item["net_success_change"]
        assert item["successes"] - 2756 == item["net_success_change"]
        assert item["successes"] + item["failures"] == 3900
        data.append(item)
    archived = json.loads((OUT / "fig_fixed_transitions.data.json").read_text(encoding="utf-8"))
    assert data == archived["data"], "The redesign must not change source values."
    return source, data


def main():
    source, data = load_data()
    apply_style()
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.grid": False,
        "savefig.bbox": None, "svg.hashsalt": "care-opd-fixed-transitions-compact",
    })
    fig = plt.figure(figsize=FIGSIZE, dpi=400)
    ax = fig.add_axes([0.035, 0.165, 0.945, 0.715])
    ax.set(xlim=(-2250, 1250), ylim=(-0.47, 4.66), yticks=[],
           xticks=[-2000, -1000, 0, 1000], xlabel="Absolute request count")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):,.0f}"))
    ax.tick_params(axis="x", length=2.4, width=0.6, pad=2.2, labelsize=7.6)
    ax.xaxis.labelpad = 3.0
    ax.axvline(0, color=AXIS, lw=0.65, zorder=1)
    semantic_text = []
    for y, row in zip(range(4, -1, -1), data):
        if row["model"] == "CARE-OPD":
            ax.axhspan(-0.32, 0.60, color="#EAF3FB", lw=0, zorder=0)
        for tick in [-2000, -1000, 1000]:
            ax.plot([tick, tick], [y - 0.28, y + 0.22], color=GRID,
                    linewidth=0.5, zorder=1)
        is_care = row["model"] == "CARE-OPD"
        name = ax.text(-2210, y + 0.37, row["model"], ha="left", va="center",
                       fontsize=7.8, color=CARE if is_care else INK,
                       weight="bold" if is_care else "normal", zorder=4)
        net = ax.text(1200, y + 0.37, f"Net {signed(row['net_success_change'])}",
                      ha="right", va="center", fontsize=7.8,
                      color=CARE if is_care else INK,
                      weight="bold" if is_care else "normal", zorder=4)
        semantic_text.extend([name, net])
        regressed, rescued = row["regressed"], row["rescued"]
        ax.barh(y, -regressed, height=0.43, color=ORANGE, edgecolor="white",
                linewidth=0.55, zorder=2)
        ax.barh(y, rescued, height=0.43, color=GREEN, edgecolor="white",
                linewidth=0.55, zorder=2)
        for x, count in [(-regressed / 2, regressed), (rescued / 2, rescued)]:
            semantic_text.append(ax.text(x, y, f"{count:,}", ha="center", va="center",
                                         fontsize=7.4, color=INK, zorder=3))

    legend = fig.legend(
        handles=[Patch(facecolor=ORANGE, label="Regressed (n = 2,756)"),
                 Patch(facecolor=GREEN, label="Rescued (n = 1,144)")],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2,
        frameon=False, handlelength=0.9, handletextpad=0.4,
        columnspacing=0.9, borderaxespad=0.0, fontsize=7.3,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1
                or box.y1 > fig.bbox.height + 1):
            clipped.append(text.get_text())
    assert not clipped, f"Clipped text: {clipped}"
    overlaps = []
    for index, first in enumerate(semantic_text):
        for second in semantic_text[index + 1:]:
            if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                overlaps.append([first.get_text(), second.get_text()])
    assert not overlaps, f"Overlapping data/header labels: {overlaps}"
    assert legend.get_window_extent(renderer).y0 > max(
        text.get_window_extent(renderer).y1 for text in semantic_text)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{STEM}.{extension}", dpi=400, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    payload = {
        "figure": STEM, "dimensions_inches": list(FIGSIZE), "font": "Arial",
        "minimum_font_points": 7.3, "data": data,
        "source_file": {"path": SOURCE_REL,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "visual_contract": {
            "format": "single-column vector paper figure",
            "evidence_layer": "fixed-reference descriptive behavior",
            "question": "How do rescues and regressions balance against the same Clean-SFT reference?",
            "grammar": "diverging count bars, with inline row headers for method and net",
            "left": "regressed absolute count", "right": "rescued absolute count",
            "conditioning_denominators": {"regressed": 2756, "rescued": 1144},
            "net": "rescued minus regressed",
            "unchanged": ["all five methods", "method order", "counts", "denominators", "semantic palette"],
            "limits": ["descriptive counts, not conditional-rate comparison", "no new uncertainty estimates"],
        },
        "layout": {"plot_width_fraction_old": 0.545, "plot_width_fraction_new": 0.945,
                   "separate_side_text_columns": False},
        "caption": "Rescues from 1,144 Clean-SFT failures and regressions from its 2,756 successes. Net counts use this fixed reference.",
        "qa": {"original_data_exact_match": True, "count_identities": "PASS",
               "text_clipping": "PASS", "data_label_overlap": "PASS",
               "legend_clearance": "PASS", "editable_svg_text": True, "png_dpi": 400},
    }
    (OUT / f"{STEM}.data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [f"{STEM}.{ext}" for ext in ["pdf", "svg", "png", "data.json"]],
                      "qa": payload["qa"]}))


if __name__ == "__main__":
    main()
