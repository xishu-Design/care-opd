"""Six-method Structured clean/outage endpoints from the frozen external audit.

All values are read from the source CSV.  Coordinate pairs are percentages
(clean TSR, outage TSR), with 65 report families per endpoint.  The short
leader strokes associate labels with points; they do not connect methods.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = (
    PROJECT / "competition_project_document/formal/competition_submission_v9"
    / "supporting_public_v9/03_真实场景外测/统一统计合同"
    / "MODEL_TRACK_STATE_METRICS.csv"
)
TRACK = "structured_evidence_faithfulness"
sys.path.insert(0, str(HERE.parent / "tmm_v4"))
from figure_style import apply_style, CARE, GKD, CLEAN, GREEN, ORANGE, MUTED, GRID, INK

# Display positions are label locations only, never displaced data coordinates.
METHODS = [
    ("frozen", "Frozen", MUTED, "o", (41, 81), (20, 53)),
    ("clean_sft", "Clean-SFT", CLEAN, "s", (9, 54), (0, 31)),
    ("recovery_sft", "Rec.-SFT", ORANGE, "^", (12, 97), (2, 64)),
    ("recovery_dpo", "Rec.-DPO", GREEN, "D", (12, 115), (0, 65)),
    ("gkd_full", "GKD-Full", GKD, "v", (59, 18), (45, 0)),
    ("care_opd", "CARE-OPD", CARE, "P", (65, 57), (44, 27)),
]


def load_endpoints():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        selected = [
            row for row in csv.DictReader(handle)
            if row["track"] == TRACK
            and row["variant"] in ("clean", "source_outage")
        ]
    expected_models = {item[0] for item in METHODS}
    assert len(selected) == 12
    assert {row["model"] for row in selected} == expected_models
    data = {}
    for model, _, _, _, _, expected_counts in METHODS:
        values = []
        for variant, expected_successes in zip(("clean", "source_outage"), expected_counts):
            matches = [r for r in selected if r["model"] == model and r["variant"] == variant]
            assert len(matches) == 1, (model, variant)
            row = matches[0]
            assert int(row["families"]) == 65
            assert int(row["task_success_evaluated"]) == 65
            assert int(row["task_successes"]) == expected_successes
            rate = float(row["task_success_rate"])
            assert math.isclose(rate, expected_successes / 65, abs_tol=1e-12)
            values.append(100 * rate)
        data[model] = values
    return data


def main():
    data = load_endpoints()
    apply_style()
    fig, ax = plt.subplots(figsize=(3.50, 2.65))
    fig.subplots_adjust(left=.155, right=.975, bottom=.18, top=.975)
    ax.set_xlim(-5, 102)
    ax.set_ylim(-6, 122)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Clean TSR (%)", labelpad=3.5)
    ax.set_ylabel("Source-outage TSR (%)", labelpad=3.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.3, pad=2.3)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=.45)

    texts = []
    for model, name, color, marker, label_xy, _ in METHODS:
        x, y = data[model]
        is_care = model == "care_opd"
        ax.scatter(
            [x], [y], marker=marker, s=36 if is_care else 25,
            facecolors=color, edgecolors="white", linewidths=.5,
            zorder=5 if is_care else 4,
        )
        # Short, arrow-free leader strokes identify nearby direct labels only.
        ax.annotate(
            "", xy=(x, y), xytext=label_xy, textcoords="data",
            arrowprops={"arrowstyle": "-", "color": color, "lw": .55,
                        "shrinkA": 1.5, "shrinkB": 4.3},
            zorder=2,
        )
        texts.append(ax.text(
            *label_xy, f"{name}\n({x:.2f}, {y:.2f})", ha="left", va="top",
            fontsize=7.2, linespacing=1.08, color=CARE if is_care else INK,
            fontweight="bold" if is_care else "normal", zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": .4},
        ))
    texts.append(ax.text(99, 117, "Structured reports\n65 families / state",
                         ha="right", va="top", fontsize=7.2,
                         color=MUTED, linespacing=1.10))

    # Verify legibility on the exact final-size canvas, before saving.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer) for text in texts]
    for artist in [*texts, ax.xaxis.label, ax.yaxis.label,
                   *ax.get_xticklabels(), *ax.get_yticklabels()]:
        box = artist.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0, artist.get_text()
        assert box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, artist.get_text()
        assert artist.get_fontsize() >= 7
    for i, first in enumerate(boxes):
        for j, second in enumerate(boxes[i + 1:], i + 1):
            assert not first.overlaps(second), (texts[i].get_text(), texts[j].get_text())
    marker_radius = 3.5 * fig.dpi / 72
    for model, (x, y) in data.items():
        px, py = ax.transData.transform((x, y))
        marker_box = Bbox.from_bounds(px - marker_radius, py - marker_radius,
                                     2 * marker_radius, 2 * marker_radius)
        for label, label_box in zip(texts, boxes):
            assert not label_box.overlaps(marker_box), (model, label.get_text())

    output = HERE / "fig_external_balance"
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    print(json.dumps({
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "track": TRACK,
        "endpoint_order": ["clean_task_success_rate", "source_outage_task_success_rate"],
        "percentages": data,
        "families_per_state": 65,
        "figure_inches": [3.50, 2.65],
        "png_pixels": [1050, 795],
        "minimum_font_pt": 7.2,
        "outputs": [str(output.with_suffix(ext)) for ext in (".pdf", ".png", ".svg")],
        "qa": "All six methods; source-count checks; labels inside canvas and non-overlapping; no inter-method lines or inferred uncertainty.",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
