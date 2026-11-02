"""Compact transposed heatmap from archived paired rescue counts.

Adapts the CCFA heatmap_matrix annotated-cell recipe and the existing
failure-rescue source loader, without recalculating a raw-record evaluation.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
STEM = "fig_failure_rescue_redraw_v1"
os.environ.setdefault("MPLCONFIGDIR", str(OUT.parent / "tmm_v4/.mplconfig_agent"))
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np

from gen_fig_failure_rescue import load_data, PALETTE, CSV_REL, JSON_REL, luminance
from figure_style import apply_style, INK, CARE


def main():
    source_manifest = OUT / "fig_failure_rescue.data.json"
    previous = json.loads(source_manifest.read_text(encoding="utf-8"))
    data, raw_status = load_data()
    assert data == previous["data"]
    values = np.asarray([[cell["percent"] for cell in row["cells"]] for row in data])
    matrix = values.T
    assert matrix.shape == (4, 5) and np.array_equal(matrix.T, values)
    labels = ["Boundary", "Invalid\noutput", "Invented\nID", "Missing\ntext", "Missing\nID"]
    methods = ["Recovery\nSFT", "Recovery\nDPO", "GKD-Full", "CARE-OPD"]
    assert [row["n"] for row in data] == [39, 48, 1028, 49, 3]

    apply_style()
    plt.rcParams.update({"axes.grid": False, "savefig.bbox": None,
                         "svg.hashsalt": STEM})
    fig = plt.figure(figsize=(3.49, 2.20), dpi=450)
    bounds = [.192, .170, .790, .640]
    ax = fig.add_axes(bounds)
    cmap = LinearSegmentedColormap.from_list("rescue_blue_green", PALETTE, N=512)
    norm = Normalize(0, 100)
    value_text = []
    cells = []
    contrasts = []
    for i in range(4):
        for j in range(5):
            val = float(matrix[i, j])
            bg = cmap(norm(val))
            lum = luminance(bg)
            white_contrast = 1.05 / (lum + .05)
            ink_contrast = (lum + .05) / (luminance(INK) + .05)
            fg = "white" if white_contrast > ink_contrast else INK
            contrast = max(white_contrast, ink_contrast)
            if contrast < 4.5:
                fg = "#000000"
                contrast = (lum + .05) / (luminance(fg) + .05)
            assert contrast >= 4.5, (val, contrast)
            contrasts.append(contrast)
            cell = Rectangle((j, 3-i), 1, 1, facecolor=bg,
                             edgecolor="white", linewidth=.75)
            ax.add_patch(cell)
            cells.append(cell)
            value_text.append(ax.text(j+.5, 3.5-i, f"{val:.2f}",
                                      ha="center", va="center", fontsize=8.0, color=fg))
    ax.set(xlim=(0, 5), ylim=(0, 4), xticks=[])
    ax.set_yticks([3.5, 2.5, 1.5, .5], methods)
    ax.tick_params(axis="y", length=0, pad=4, labelsize=7.2)
    for t in ax.get_yticklabels():
        t.set_linespacing(1.02)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_color(CARE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    headers, counts = [], []
    for j, (label, row) in enumerate(zip(labels, data)):
        xp = bounds[0] + bounds[2] * (j+.5) / 5
        headers.append(fig.text(xp, .889, label, ha="center", va="bottom",
                                fontsize=7.2, linespacing=1.02))
        counts.append(fig.text(xp, .846, f"n={row['n']:,}", ha="center", va="center",
                               fontsize=6.8, color="#596779"))
    cax = fig.add_axes([.205, .077, .754, .026])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=1.7, width=.55, pad=1.5, labelsize=7.0)
    fig.text(.015, .09, "Rescue\n(%)", ha="left", va="center", fontsize=7.0, linespacing=1.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = [(t, t.get_window_extent(renderer)) for t in fig.findobj(Text)
                  if t.get_visible() and t.get_text()]
    clipped = [t.get_text() for t, box in text_boxes
               if box.x0 < -.5 or box.y0 < -.5
               or box.x1 > fig.bbox.width+.5 or box.y1 > fig.bbox.height+.5]
    assert not clipped, clipped
    overlaps = [(a.get_text(), b.get_text())
                for k, (a, box) in enumerate(text_boxes)
                for b, other in text_boxes[k+1:] if box.overlaps(other)]
    assert not overlaps, overlaps
    for t, cell in zip(value_text, cells):
        tb, cb = t.get_window_extent(renderer), cell.get_window_extent(renderer)
        assert cb.contains(tb.x0, tb.y0) and cb.contains(tb.x1, tb.y1), t.get_text()
    assert len(value_text) == 20
    assert [t.get_text() for t in value_text] == [f"{v:.2f}" for v in matrix.flatten()]
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=450)
    plt.close(fig)

    sources = [source_manifest, ROOT/CSV_REL, ROOT/JSON_REL]
    manifest = {
        "schema": "compact-failure-stratum-rescue-v1",
        "sources": [{"path": p.relative_to(ROOT).as_posix(),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        "data": data, "display_method_rows": [c["display"] for c in data[0]["cells"]],
        "display_failure_columns": [r["label"] for r in data],
        "display_values_percent": matrix.tolist(),
        "metric": previous["metric"], "population": previous["population"],
        "claim_limits": previous["claim_limits"],
        "raw_record_status": {"full_local_join_recomputed": False,
                              "limitation": previous["raw_record_status"]["limitation"]},
        "render": {"size_inches": [3.49, 2.20], "axes_bounds": bounds,
                   "old_axes_bounds": [.285, .265, .688, .565],
                   "palette": PALETTE, "limits": [0,100], "font": "Arial",
                   "numeric_label_pt": 8.0, "precision_decimals": 2},
        "qa": {"transposition_verified": True, "numeric_labels": 20,
               "clipped_text": clipped, "text_collisions": overlaps,
               "minimum_number_contrast": round(min(contrasts),3)}
    }
    (OUT / f"{STEM}.source.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",
                                           encoding="utf-8")
    print(json.dumps({"saved":STEM,"qa":manifest["qa"]}))


if __name__ == "__main__":
    main()
