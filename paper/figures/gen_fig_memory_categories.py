#!/usr/bin/env python3
"""Replot the reported LongMemEval category values, without new evaluation.

Values are transcribed at the integer-percent precision shown in Figure 11 of
the source DOCX. Subgroup denominators and uncertainty are not supplied there.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parents[2] / "competition_project_document/formal/实验结果.docx"
MEMBER = "word/media/image14.png"
OUT = Path(__file__).resolve().parent
STEM = "fig_memory_categories"
CATEGORIES = ["Update", "Multi", "Assistant", "Preference", "User", "Temporal"]
ROWS = [
    ("Controlled", "M0 Sliding window", [24, 8, 7, 3, 13, 5]),
    ("Controlled", "M1 Summary + recent", [33, 13, 9, 27, 14, 20]),
    ("Controlled", "M2 Mem0", [67, 26, 88, 23, 80, 29]),
    ("Controlled", "M3 Mem0 + Graphiti", [68, 26, 88, 23, 80, 35]),
    ("Real-component isolation", "M2 Mem0", [74, 39, 86, 43, 81, 51]),
    ("Real-component isolation", "M3 Mem0 + Graphiti", [74, 39, 86, 43, 80, 53]),
]


def main() -> None:
    font = Path("C:/Windows/Fonts/arial.ttf")
    if not font.is_file():
        raise FileNotFoundError("Arial is required for this figure.")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 300, "savefig.dpi": 300,
    })
    source_bytes = SOURCE.read_bytes()
    with zipfile.ZipFile(SOURCE) as archive:
        original = archive.read(MEMBER)
    source_image = ROOT / "build_experiments/source_memory_categories.png"
    source_image.parent.mkdir(parents=True, exist_ok=True)
    source_image.write_bytes(original)

    values = np.asarray([row[2] for row in ROWS], dtype=int)
    assert values.shape == (6, 6)
    assert values.min() >= 0 and values.max() <= 100
    cmap = colors.LinearSegmentedColormap.from_list(
        "memory_blue", ["#f7fbff", "#b7dcef", "#56b4e9", "#0072b2", "#063b63"]
    )
    norm = colors.Normalize(0, 100)
    fig = plt.figure(figsize=(7.16, 2.35), facecolor="white")
    ax = fig.add_axes([0.255, 0.12, 0.625, 0.835])
    bottoms = [5.25, 4.25, 3.25, 2.25, 1.0, 0.0]
    for i, (_, _, scores) in enumerate(ROWS):
        for j, value in enumerate(scores):
            rgba = cmap(norm(value))
            ax.add_patch(Rectangle((j, bottoms[i]), 1, 1, facecolor=rgba,
                                   edgecolor="white", linewidth=0.7))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j + 0.5, bottoms[i] + 0.5, str(value), ha="center", va="center",
                    fontsize=8, color="white" if luminance < 0.50 else "#111111")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6.25)
    ax.set_xticks(np.arange(6) + 0.5, CATEGORIES)
    ax.set_yticks([b + 0.5 for b in bottoms], [row[1] for row in ROWS])
    ax.tick_params(axis="both", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # A physical gap and rule separate protocols; no cross-protocol trajectory.
    ax.axhline(2.125, color="#66717b", linewidth=0.45)
    fig.text(0.025, 0.6878, "Controlled", ha="center", va="center", rotation=90)
    fig.text(0.025, 0.2536, "Real isolated", ha="center", va="center", rotation=90)
    cax = fig.add_axes([0.904, 0.12, 0.017, 0.835])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, 50, 100])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, width=0.5, pad=3)
    bar.set_label("Judge accuracy (%)", labelpad=4)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{STEM}.{extension}", dpi=300)
    plt.close(fig)

    payload = {
        "schema": "reported-memory-category-heatmap-v1",
        "source_docx": str(SOURCE),
        "source_docx_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_figure": "Figure 11", "source_member": MEMBER,
        "source_image_sha256": hashlib.sha256(original).hexdigest(),
        "source_image_copy": str(source_image),
        "benchmark": "LongMemEval-S-Cleaned",
        "protocol_total_questions_each": 500,
        "metric": "project non-official judge accuracy, percent",
        "display_precision": "integer percentages transcribed from source figure",
        "category_sample_sizes": None,
        "confidence_intervals": None,
        "columns": CATEGORIES,
        "rows": [{"protocol": p, "configuration": name, "values_percent": scores}
                 for p, name, scores in ROWS],
        "additional_text_only_precision": {
            "real_temporal_mem0_percent": 51.13,
            "real_temporal_mem0_graphiti_percent": 53.38,
            "note": "Shown only in source text; heatmap consistently retains image precision.",
        },
        "comparability": [
            "Controlled and real-component protocols must not be pooled or read as before-after.",
            "Overall 500 questions is not a category denominator.",
            "Values do not establish category-specific significance or universal Graphiti benefit.",
            "This is a system-memory evaluation, not a CARE-OPD training gain.",
        ],
        "render": {"size_inches": [7.16, 2.35], "font": "Arial", "font_points": 8,
                   "png_dpi": 300, "color_limits_percent": [0, 100]},
    }
    (OUT / f"{STEM}.data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Source image: {source_image}")
    print(f"Wrote {STEM}.pdf/.png/.svg/.data.json ({values.size} reported cells).")


if __name__ == "__main__":
    main()
