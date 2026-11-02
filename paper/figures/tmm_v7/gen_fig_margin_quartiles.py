"""Publication figure from the frozen margin/behavior audit (no new experiment)."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = PROJECT / "runs/mechanism_eval/qzbc_care_opd_v1/margin_behavior/margin_behavior.json"
sys.path.insert(0, str(HERE.parent / "tmm_v4"))
from figure_style import apply_style, CARE, GKD, ORANGE, GRID, INK


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    selected = report["quartile_recovery"]["care_localized"]
    outside = report["quartile_recovery"]["care_outside"]
    behavior = report["behavior"]
    paired_n = behavior["n"]
    care_only = behavior["care_only_rescued"]
    gkd_only = behavior["gkd_only_rescued"]
    both = behavior["care_rescued"] - care_only
    neither = paired_n - both - care_only - gkd_only
    assert both == behavior["gkd_rescued"] - gkd_only
    assert (paired_n, both, care_only, gkd_only, neither) == (200, 132, 42, 3, 23)
    assert [r["n"] for r in selected] == [48, 47, 47, 47]
    assert [r["care_rescued"] for r in selected] == [24, 46, 47, 47]
    assert [r["care_rescued"] for r in outside] == [36, 41, 43, 44]
    assert sum(r["care_rescued"] for r in selected) == 164
    assert sum(r["care_rescued"] for r in outside) == 164
    local_n = sum(r["n"] for r in selected)
    assert local_n == sum(r["n"] for r in outside) == 189
    apply_style()
    fig = plt.figure(figsize=(3.50, 2.88))
    # The equal-width cells are categorical counts, not area-encoded bars.
    # Direct labels avoid a second long legend and keep the small GKD-only
    # count legible without inflating its proportion in a stacked bar.
    fig.text(.035, .972, f"(a) Paired rescue (n={paired_n})", fontsize=7.4,
             fontweight="bold", va="top")
    tile_width, tile_gap = .223, .012
    count_tiles = [
        ("Both", both, "#E5EEF4", INK),
        ("CARE only", care_only, CARE, "white"),
        ("GKD only", gkd_only, GKD, "white"),
        ("Neither", neither, GRID, INK),
    ]
    for i, (label, count, fill, text_color) in enumerate(count_tiles):
        tile_left = .035 + i * (tile_width + tile_gap)
        fig.add_artist(Rectangle((tile_left, .779), tile_width, .143,
                                 transform=fig.transFigure, facecolor=fill,
                                 edgecolor="none", zorder=1))
        fig.text(tile_left + tile_width / 2, .882, str(count), fontsize=10.5,
                 fontweight="bold", color=text_color, ha="center", va="center")
        fig.text(tile_left + tile_width / 2, .820, label, fontsize=7.1,
                 color=text_color, ha="center", va="center")
    fig.text(.035, .735, f"(b) n={local_n}", fontsize=7.4,
             fontweight="bold", va="center")
    ax = fig.add_axes([.145, .165, .815, .485])
    x = np.arange(1, 5)
    y_selected = np.array([r["care_rescue_rate"] for r in selected]) * 100
    y_outside = np.array([r["care_rescue_rate"] for r in outside]) * 100
    ax.plot(x, y_selected, color=CARE, marker="o", markersize=4.8,
            linewidth=1.6, label="Selected fields", zorder=4)
    ax.plot(x, y_outside, color=ORANGE, marker="D", markersize=4.0,
            markerfacecolor="white", markeredgewidth=1.2, linestyle="--",
            linewidth=1.3, label="Outside fields", zorder=3)
    for i, row in enumerate(selected):
        label = f'{row["care_rescued"]}/{row["n"]}'
        # Place Q1 below-right to avoid both its rising line and the y spine.
        ax.annotate(label, (x[i], y_selected[i]),
                    xytext=(6, -2) if i == 0 else (0, 6),
                    textcoords="offset points", ha="left" if i == 0 else "center",
                    va="top" if i == 0 else "bottom",
                    fontsize=7.1, color=CARE)
    for i, row in enumerate(outside):
        label = f'{row["care_rescued"]}/{row["n"]}'
        ax.annotate(label, (x[i], y_outside[i]), xytext=(0, -8),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7.1, color="#975F27")
    ax.set_xlim(.78, 4.22)
    ax.set_ylim(40, 111)
    ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xlabel("Margin quartile (low to high)", labelpad=3)
    ax.set_ylabel("Rescued (%)", labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2.5)
    legend = fig.legend(*ax.get_legend_handles_labels(), loc="center right",
                        bbox_to_anchor=(.965, .735), ncol=2, handlelength=1.35,
                        columnspacing=.8, handletextpad=.4, borderaxespad=0,
                        fontsize=7.1)
    # Fixed-size vector canvas: check every visible label at the intended
    # single-column size, including the scope labels and both count sequences.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    checked_texts = [*fig.texts, ax.xaxis.label, ax.yaxis.label,
                     *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts,
                     *legend.get_texts()]
    for artist in checked_texts:
        bbox = artist.get_window_extent(renderer)
        assert artist.get_fontsize() >= 7.0
        assert bbox.x0 >= 0 and bbox.y0 >= 0, artist.get_text()
        assert bbox.x1 <= fig.bbox.x1 and bbox.y1 <= fig.bbox.y1, artist.get_text()
    # Text-label collisions are checked separately from the enclosing legend.
    for i, first in enumerate(checked_texts):
        first_box = first.get_window_extent(renderer)
        for second in checked_texts[i + 1:]:
            assert not first_box.overlaps(second.get_window_extent(renderer)), (
                first.get_text(), second.get_text())
    out = HERE / "fig_margin_quartiles"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(json.dumps({"source": str(SOURCE), "pdf": str(out.with_suffix('.pdf')),
                      "png": str(out.with_suffix('.png')),
                      "size_inches": [3.50, 2.88], "minimum_font_pt": 7.1,
                      "paired_counts": {"both": both, "care_only": care_only,
                                        "gkd_only": gkd_only, "neither": neither},
                      "qa": "All labels inside canvas; no text collisions; archived counts verified.",
                      "scope": "Paired rescue: 200 frozen failures. Margin quartiles: 189 localizable cases, independently ranked; no inferred intervals."}))


if __name__ == "__main__":
    main()
