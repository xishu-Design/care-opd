#!/usr/bin/env python3
"""Method-aligned paired bars with in-plot net-change annotations.

Academic-plotting workflow: zero-based shared count axis, muted apricot/blue,
direct exact counts, and an unboxed net annotation above each pair. No data transformation,
conditional-rate pooling, smoothing, or uncertainty estimation is performed.
"""
from __future__ import annotations

import hashlib
import json

from gen_fig_fixed_transitions_compact import OUT, load_data, signed
from gen_fig_fixed_transitions import plt
from figure_style import apply_style
from matplotlib.patches import Patch
from matplotlib import patheffects
from matplotlib.text import Text
from matplotlib.ticker import StrMethodFormatter


STEM = "fig_fixed_transitions_grouped_v5"
FIGSIZE = (4.15, 1.97)
INK = "#20314B"
MUTED = "#53667C"
AXIS = "#8492A2"
GRID = "#E7ECF2"
REGRESSED = "#E6B18A"
RESCUED = "#6C94C7"
CARE = "#2B5C97"


def main():
    source, data = load_data()
    apply_style()
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "savefig.bbox": None,
        "axes.linewidth": 0.60,
        "svg.hashsalt": "care-opd-fixed-transitions-grouped-v5",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": AXIS,
    })
    fig = plt.figure(figsize=FIGSIZE, dpi=450)
    ax = fig.add_axes([0.125, 0.20, 0.86, 0.695])
    ax.set(xlim=(-0.57, 4.57), ylim=(0, 2600),
           xticks=list(range(5)), yticks=[0, 500, 1000, 1500, 2000],
           ylabel="Request count")
    ax.set_xticklabels(["Frozen", "Recovery\nSFT", "Recovery\nDPO", "GKD-Full", "CARE-OPD"],
                       fontsize=7.5, linespacing=1.12)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=2.3, width=0.6, pad=2.5, labelsize=7.4)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.labelpad = 3.3
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    value_text = []
    net_text = []
    for index, row in enumerate(data):
        for offset, field, color in [(-0.165, "regressed", REGRESSED), (0.165, "rescued", RESCUED)]:
            value = row[field]
            ax.bar(index + offset, value, width=0.30, color=color,
                   edgecolor="none", linewidth=0, zorder=3)
            value_text.append(ax.text(index + offset, value + 45, f"{value:,}",
                                      ha="center", va="bottom", color=INK,
                                      fontsize=7.2, zorder=4))
        is_care = row["model"] == "CARE-OPD"
        # Pair-local annotations use label placement, not a third numerical mark.
        # No boxes, footer, extra data series, or cross-method connecting lines.
        top = max(row["regressed"], row["rescued"])
        label = ax.text(index, top + 380, f"Net {signed(row['net_success_change'])}",
                        ha="center", va="center", fontsize=7.0,
                        color=CARE if is_care else MUTED,
                        weight="bold" if is_care else "normal",
                        path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                        zorder=5)
        value_text.append(label)
        net_text.append(label)
    ax.get_xticklabels()[-1].set_color(CARE)
    ax.get_xticklabels()[-1].set_weight("bold")
    fig.legend(handles=[Patch(facecolor=REGRESSED, label="Regressed (n = 2,756)"),
                        Patch(facecolor=RESCUED, label="Rescued (n = 1,144)")],
               loc="upper center", bbox_to_anchor=(0.525, 0.995), ncol=2,
               frameon=False, handlelength=0.9, handletextpad=0.42,
               columnspacing=0.9, borderaxespad=0.0, fontsize=7.3)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            clipped.append(text.get_text())
    assert not clipped, f"Clipped text: {clipped}"
    overlaps = []
    for index, first in enumerate(value_text):
        for second in value_text[index + 1:]:
            if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                overlaps.append([first.get_text(), second.get_text()])
    assert not overlaps, f"Overlapping value labels: {overlaps}"
    for label in net_text:
        label_box = label.get_window_extent(renderer)
        assert ax.bbox.contains(label_box.x0, label_box.y0)
        assert ax.bbox.contains(label_box.x1, label_box.y1)
        for count_label in value_text:
            if count_label not in net_text:
                assert not label_box.overlaps(count_label.get_window_extent(renderer)), \
                    "A Net annotation must not cover a count label."
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{STEM}.{extension}", dpi=450, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    payload = {
        "figure": STEM, "dimensions_inches": list(FIGSIZE), "font": "Arial",
        "minimum_font_points": 7.0, "data": data,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "visual_contract": {
            "format": "wider grouped bars with in-plot net annotations",
            "question": "Which methods rescue more requests than they regress against a fixed reference?",
            "evidence_layer": "fixed-reference descriptive behavior",
            "y_axis": "absolute request count, shared linear scale starting at zero",
            "methods": [row["model"] for row in data],
            "semantic_colors": {"regressed": REGRESSED, "rescued": RESCUED, "method_highlight": CARE,
                                "ink": INK, "secondary_ink": MUTED, "axis": AXIS, "grid": GRID},
            "conditioning_denominators": {"regressed": 2756, "rescued": 1144},
            "net_definition": "rescued - regressed",
            "net_position": "unboxed pair-local text above each pair; no bracket or third quantitative series",
            "limits": ["Different conditioning denominators; bars do not compare conditional rates",
                       "No new statistical inference", "No categorical interpolation"],
        },
        "style_alignment": {
            "method_reference": "../method_diagnostics_abc_v21.png",
            "main_reference": "output/pdf/agent_care_opd_tmm/journal_v37_handwritten_type/fig_care_opd_main_final.png",
            "matched": ["Arial quantitative typography", "dark navy ink", "blue/warm-orange visual family",
                        "thin gray axes", "pale grid", "blue CARE emphasis", "direct unboxed annotations"],
            "omitted_from_main_diagram": ["handwritten display font", "decorative icons"],
            "palette_revision": "The same in-plot layout as v3, with muted apricot and blue instead of orange and green.",
            "layout_revision": "Canvas widened from 3.49 to 4.15 inches, retaining height and physical font sizes.",
        },
        "caption": "Rescues from 1,144 Clean-SFT failures and regressions from its 2,756 successes. Net = rescued minus regressed, using the same fixed reference.",
        "qa": {"source_values_match_original": True, "text_clipping": "PASS",
               "value_label_overlap": "PASS", "zero_based_shared_axis": True,
               "net_annotations_inside_plot": "PASS", "net_count_separation": "PASS",
               "editable_svg_text": True, "vector_pdf_fonttype": 42, "png_dpi": 450},
    }
    previous = json.loads((OUT / "fig_fixed_transitions_grouped_v4.data.json").read_text(encoding="utf-8"))
    assert payload["data"] == previous["data"]
    assert payload["dimensions_inches"][1] == previous["dimensions_inches"][1]
    assert payload["visual_contract"]["semantic_colors"] == previous["visual_contract"]["semantic_colors"]
    payload["qa"]["width_only_against_v4"] = "PASS"
    (OUT / f"{STEM}.data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [f"{STEM}.{ext}" for ext in ["pdf", "png", "svg"]], "qa": payload["qa"]}))


if __name__ == "__main__":
    main()
