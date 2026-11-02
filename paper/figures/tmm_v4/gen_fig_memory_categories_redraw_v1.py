"""Expand a source-bound memory heatmap by transposing its table geometry.

Uses the annotated-cell grammar from the CCFA heatmap_matrix recipe,
translated to Matplotlib for matched publication styling and vector output.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
SOURCE = OUT.parent / "fig_memory_categories.data.json"
STEM = "fig_memory_categories_redraw_v1"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_agent"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np

from figure_style import apply_style

INK = "#20314B"
RULE = "#A4B3C4"


def luminance(rgb):
    linear = [(v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4)
              for v in colors.to_rgb(rgb)]
    return sum(v * w for v, w in zip(linear, (.2126, .7152, .0722)))


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    previous = json.loads((OUT / "fig_memory_categories.source.json").read_text(encoding="utf-8"))
    assert payload["rows"] == previous["data"]["rows"]
    assert payload["columns"] == previous["data"]["columns"]
    rows = payload["rows"]
    values = np.asarray([r["values_percent"] for r in rows], dtype=int)
    assert values.shape == (6, 6)
    assert [r["protocol"] for r in rows] == ["Controlled"] * 4 + ["Real-component isolation"] * 2
    displayed = values.T
    assert np.array_equal(displayed.T, values)
    labels = ["Sliding\nwindow", "Summary\n+ recent", "Mem0", "Mem0 +\nGraphiti",
              "Mem0", "Mem0 +\nGraphiti"]

    apply_style()
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.grid": False, "savefig.bbox": None,
                         "svg.hashsalt": STEM})
    fig = plt.figure(figsize=(7.16, 2.64), dpi=350)
    ax = fig.add_axes([.103, .027, .818, .770])
    x_left = np.array([0., 1., 2., 3., 4.15, 5.15])
    cmap = plt.get_cmap("Blues")
    norm = colors.Normalize(vmin=0, vmax=100)
    annotations = []
    cells = []
    contrasts = []
    for i, category in enumerate(payload["columns"]):
        for j, row in enumerate(rows):
            value = int(displayed[i, j])
            y_bottom = 5 - i
            bg = cmap(norm(value))
            lum = luminance(bg[:3])
            dark = (lum + .05) / (luminance(INK) + .05)
            white = 1.05 / (lum + .05)
            fg = "white" if white > dark else INK
            contrast = max(dark, white)
            if contrast < 4.5:
                fg = "#111111"
                contrast = (lum + .05) / (luminance(fg) + .05)
            assert contrast >= 4.5, (value, contrast)
            contrasts.append(contrast)
            cell = Rectangle((x_left[j], y_bottom), 1, 1, facecolor=bg,
                             edgecolor="white", linewidth=.85)
            ax.add_patch(cell)
            cells.append(cell)
            annotations.append(ax.text(x_left[j] + .5, y_bottom + .5, str(value),
                                       ha="center", va="center", fontsize=9.4, color=fg))
    ax.set(xlim=(0, 6.15), ylim=(0, 6))
    ax.set_xticks(x_left + .5, labels)
    ax.set_yticks(np.arange(5.5, 0, -1), payload["columns"])
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False,
                   length=0, pad=4.5, labelsize=8.1)
    ax.tick_params(axis="y", length=0, pad=4.5, labelsize=8.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(4.075, color=RULE, lw=.55, zorder=1)
    groups = []
    for start, end, title in [(0, 4, "Controlled"), (4.15, 6.15, "Real-component isolation")]:
        center = (start + end) / (2 * 6.15)
        groups.append(ax.text(center, 1.174, title, transform=ax.transAxes, ha="center",
                              va="bottom", fontsize=8.7, fontweight="bold", clip_on=False))
        ax.plot([start / 6.15, end / 6.15], [1.155, 1.155], transform=ax.transAxes,
                color=RULE, linewidth=.65, clip_on=False)

    cax = fig.add_axes([.939, .027, .013, .770])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       ticks=[0, 25, 50, 75, 100])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, width=.55, pad=2.5, labelsize=7.5)
    cax.set_title("%", fontsize=8, pad=4, color=INK)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    all_text = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    boxes = [(t, t.get_window_extent(renderer)) for t in all_text]
    for t, box in boxes:
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            clipped.append(t.get_text())
    assert not clipped, clipped
    collisions = []
    for i, (t, box) in enumerate(boxes):
        for u, other in boxes[i + 1:]:
            if box.overlaps(other):
                collisions.append((t.get_text(), u.get_text()))
    assert not collisions, collisions
    for t, cell in zip(annotations, cells):
        tb, cb = t.get_window_extent(renderer), cell.get_window_extent(renderer)
        assert cb.contains(tb.x0, tb.y0) and cb.contains(tb.x1, tb.y1), t.get_text()
    assert len(annotations) == 36
    assert [int(t.get_text()) for t in annotations] == displayed.flatten().tolist()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=350)
    plt.close(fig)

    manifest = {
        "schema": "memory-category-transposed-heatmap-v1",
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(),
                   "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "source_rows": rows, "category_rows": payload["columns"],
        "display_values_percent": displayed.tolist(),
        "render": {"size_inches": [7.16, 2.64], "matrix_width_fraction": .818,
                   "previous_matrix_width_fraction": .625,
                   "colormap": "Blues", "limits": [0, 100],
                   "font": "Arial", "numeric_label_pt": 9.4},
        "protocol_total_questions_each": 500,
        "metric": payload["metric"],
        "category_sample_sizes": None, "confidence_intervals": None,
        "claim_boundary": payload["comparability"],
        "qa": {"numeric_labels": 36, "clipped_text": clipped, "text_collisions": collisions,
               "minimum_number_contrast": round(min(contrasts), 3),
               "transposition_verified": True}
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"saved": STEM, "qa": manifest["qa"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
