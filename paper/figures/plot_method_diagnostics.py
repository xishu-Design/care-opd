"""Reproducible, full-width paper graphic from the two completed diagnostic CSVs.

Run with a Python environment containing matplotlib. No inference or resampling.
The output suffix keeps the previous figure and the raster style draft intact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def text_contrast_ratio(foreground, background):
    """Return the sRGB relative-luminance contrast for two opaque colors."""
    def luminance(color):
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return sum(weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), linear))

    low, high = sorted((luminance(foreground), luminance(background)))
    return round((high + 0.05) / (low + 0.05), 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[3])
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    source = args.repo_root / "runs/mechanism_eval/qzbc_care_opd_v1"
    localization_path = source / "localization/localization_analysis.csv"
    margin_path = source / "recovery_margin/recovery_margin_summary.csv"
    overall = next(r for r in read_csv(localization_path) if r["scope"] == "overall")
    by_model = {r["model_label"]: r for r in read_csv(margin_path)}
    total = int(overall["full_response_tokens"])
    selected = int(overall["selected_tokens"])
    outside = int(overall["outside_selected_tokens"])
    assert selected + outside == total
    assert abs(selected / total - float(overall["token_coverage_micro"])) < 1e-12
    model_keys = ["clean_sft", "gkd_full", "care_opd"]
    scopes = ["full_margin_mean", "localized_margin_mean", "outside_margin_mean"]
    values = [[float(by_model[key][scope]) for scope in scopes] for key in model_keys]
    assert all(int(by_model[key]["examples"]) == 200 for key in model_keys)
    assert all(int(by_model[key]["localized_examples"]) == 189 for key in model_keys)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "care-opd-method-diagnostics-v12",
        "text.color": "#1F2D40",
        "axes.labelcolor": "#1F2D40",
        "xtick.color": "#1F2D40",
        "ytick.color": "#1F2D40",
        "hatch.linewidth": 0.35,
    })
    colors = ["#5B8FD9", "#F2A64A", "#4FB28F"]
    ink, muted, grid = "#1F2D40", "#596779", "#E4E9ED"
    # Darker count text clears 4.5:1 on the approved saturated blue fill.
    count_ink = "#172638"
    assert min(text_contrast_ratio(count_ink, color) for color in colors) >= 4.5
    fig = plt.figure(figsize=(7.3, 2.80), facecolor="white")
    # Reserve a genuine categorical-label column to the left of the y-axis.
    # Both plot rectangles have exactly the same physical width and height.
    left = fig.add_axes([0.140, 0.240, 0.360, 0.580])
    right = fig.add_axes([0.600, 0.240, 0.360, 0.580])

    fig.text(0.140, 0.933, "(a) Responsibility coverage", size=9, weight="bold")
    fig.text(0.600, 0.933, "(b) Validated-target preference", size=9, weight="bold")

    # Three explicitly nested counts, not three disjoint categories.
    counts = [total, selected, outside]
    names = ["Full response", "Selected fields", "Outside fields"]
    ys = [2, 1, 0]
    category_labels = []
    count_labels = []
    count_bars = []
    for y, n, name, color in zip(ys, counts, names, colors):
        bar = left.barh(y, n / 1000, height=0.68, color=color, edgecolor="none", zorder=3)[0]
        count_bars.append(bar)
        suffix = "100%" if n == total else f"{n / total * 100:.2f}%"
        category_labels.append(f"{name}\n({suffix})")
        count_labels.append(left.text(n / 1000 - 8, y, f"{n:,}", size=7.3,
                                      va="center", ha="right", color=count_ink))
    left.set_xlim(0, 420)
    left.set_ylim(-0.5, 2.5)
    left.set_xticks([0, 100, 200, 300, 400])
    left.set_yticks(ys, category_labels)
    left.set_xlabel("Response tokens (×10³)", labelpad=4)
    left.tick_params(axis="x", length=2.5, width=0.6, pad=3)
    left.tick_params(axis="y", length=2.5, width=0.6, pad=5, labelsize=7.5)
    for ticklabel in left.get_yticklabels():
        ticklabel.set_horizontalalignment("right")
        ticklabel.set_multialignment("right")
        ticklabel.set_linespacing(1.2)
    for side in ["top", "right"]:
        left.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        left.spines[side].set_visible(True)
        left.spines[side].set_color("#6C747C")
        left.spines[side].set_linewidth(0.6)
    left.set_axisbelow(True)
    # No background grid behind the category labels.
    fig.text(0.140, 0.865, f"{int(overall['rows']):,} held-out failures", size=7.5, color=muted)
    # A ratio has no trailing multiplication sign. Use a complete approximate
    # equality because the displayed multiplier is rounded from the CSV counts.
    ratio_equation = (r"N_{\mathrm{full}}\approx "
                      + f"{total / selected:.3f}"
                      + r"\,N_{\mathrm{selected}}")
    left_note = fig.text(0.320, 0.047, "Pre-trust: $" + ratio_equation + "$",
                         size=7.5, color=muted, ha="center")

    # Common baseline and range for all three checkpoints and all three scopes.
    width = 0.325
    labels = []
    for j, (name, color) in enumerate(zip(names, colors)):
        xs = [i + (j - 1) * width for i in range(3)]
        heights = [row[j] for row in values]
        right.bar(xs, heights, width=width * 0.80, color=color,
                  edgecolor="none", label=name, zorder=3)
        for x, value in zip(xs, heights):
            label = f"{value:.2f}".replace("-", "−")
            text = right.annotate(label, (x, value), xytext=(0, 3 if value >= 0 else -3),
                                 textcoords="offset points", ha="center",
                                 va="bottom" if value >= 0 else "top", size=7.2)
            labels.append(text)
    right.axhline(0, color="#545D65", linewidth=0.8, zorder=4)
    right.set_ylim(-5.10, 1.14)
    right.set_xlim(-0.52, 2.52)
    right.yaxis.set_major_locator(MultipleLocator(1))
    right.set_xticks([0, 1, 2], ["Clean-SFT", "GKD-full", "CARE-OPD"])
    right.set_ylabel("Oracle − rejected\nmean log-probability margin", labelpad=6, linespacing=1.15)
    right.tick_params(axis="x", length=0, pad=6)
    right.tick_params(axis="y", length=2.5, width=0.6, pad=3)
    right.grid(axis="y", color=grid, linewidth=0.45, zorder=0)
    right.set_axisbelow(True)
    for side in ["top", "right"]:
        right.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        right.spines[side].set_color("#6C747C")
    handles, legend_labels = right.get_legend_handles_labels()
    # Figure-level legend occupies a reserved strip, never the plotting axes.
    legend = fig.legend(handles, legend_labels, loc="lower left",
                        bbox_to_anchor=(0.600, 0.849), ncol=3,
                        frameon=False, borderaxespad=0, handlelength=1.1,
                        handletextpad=0.4, columnspacing=0.6, fontsize=7.3)
    right_note = fig.text(0.780, 0.047, "n = 200 (full); n = 189 (selected / outside)",
                          size=7.3, color=muted, ha="center")

    # Check the most collision-prone direct labels after actual font layout.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in labels]
    collisions = [(i, j) for i, a in enumerate(boxes)
                  for j, b in enumerate(boxes) if j > i and a.overlaps(b)]
    assert not collisions, f"Value labels overlap: {collisions}"
    for label, bar in zip(count_labels, count_bars):
        text_box = label.get_window_extent(renderer)
        bar_box = bar.get_window_extent(renderer)
        assert bar_box.contains(text_box.x0, text_box.y0)
        assert bar_box.contains(text_box.x1, text_box.y1)
    assert left.spines["left"].get_visible()
    assert left.get_position().y0 == right.get_position().y0
    assert left.get_position().y1 == right.get_position().y1
    assert abs(left.get_position().width - right.get_position().width) < 1e-12
    assert "\n" not in left_note.get_text() and "\n" not in right_note.get_text()

    # Crop the union of all visible artwork, not merely the axes rectangles.
    # Use one measured bbox for every output format and preserve two points.
    crop_padding_points = 2
    export_bbox = fig.get_tightbbox(renderer).padded(crop_padding_points / 72)
    export_bbox_px = export_bbox.transformed(fig.dpi_scale_trans)
    safety_px = fig.dpi * crop_padding_points / 72
    edge_violations = []
    text_artists = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    # Unused tick objects beyond the actual axis limits are not drawn.
    active_ticks = {id(t.label1) for axis in [left.xaxis, left.yaxis, right.xaxis, right.yaxis]
                    for t in axis._update_ticks()}
    all_ticks = {id(t.label1) for axis in [left.xaxis, left.yaxis, right.xaxis, right.yaxis]
                 for t in axis.get_major_ticks() + axis.get_minor_ticks()}
    text_artists = [t for t in text_artists if id(t) not in all_ticks or id(t) in active_ticks]
    for item in text_artists + [legend]:
        box = item.get_window_extent(renderer)
        if (box.x0 < export_bbox_px.x0 + safety_px - 0.1
                or box.y0 < export_bbox_px.y0 + safety_px - 0.1
                or box.x1 > export_bbox_px.x1 - safety_px + 0.1
                or box.y1 > export_bbox_px.y1 - safety_px + 0.1):
            edge_violations.append(getattr(item, "get_text", lambda: "legend")())
    assert not edge_violations, f"Export safety margin violated: {edge_violations}"

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "method_localization_and_preference_v12"
    for fmt in ["pdf", "svg", "png"]:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=400, facecolor="white",
                    bbox_inches=export_bbox, pad_inches=0)
    # Portable SVG cannot shift labels if the target machine lacks Arial.
    # Keep the normal SVG editable and provide an outlined copy separately.
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(stem.with_name(stem.name + "_outlined").with_suffix(".svg"),
                    facecolor="white", bbox_inches=export_bbox, pad_inches=0)
    manifest = {
        "figure": stem.name,
        "size_inches": [export_bbox.width, export_bbox.height],
        "working_canvas_inches": [fig.get_figwidth(), fig.get_figheight()],
        "plot_rectangles_inches": [[ax.get_position().width * fig.get_figwidth(),
                                    ax.get_position().height * fig.get_figheight()]
                                   for ax in [left, right]],
        "sources": [{"path": str(p.relative_to(args.repo_root)),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in [localization_path, margin_path]],
        "counts": {"responses": int(overall["rows"]), "full_tokens": total,
                   "selected_tokens": selected, "outside_tokens": outside},
        "selected_fraction_micro": selected / total,
        "full_to_selected_ratio": total / selected,
        "ratio_equation_latex": ratio_equation,
        "margin_models": model_keys,
        "margin_scopes": scopes,
        "margin_values": values,
        "care_full_display": f"{values[2][0]:.2f}",
        "palette": {"full_response": colors[0], "selected_fields": colors[1],
                    "outside_fields": colors[2], "ink": ink,
                    "secondary": muted, "grid": grid, "count_label_ink": count_ink},
        "text_contrast": {
            "metric": "sRGB relative-luminance contrast ratio for opaque colors",
            "ink_on_white": text_contrast_ratio(ink, "#FFFFFF"),
            "secondary_on_white": text_contrast_ratio(muted, "#FFFFFF"),
            "ink_on_full_response": text_contrast_ratio(ink, colors[0]),
            "ink_on_selected_fields": text_contrast_ratio(ink, colors[1]),
            "ink_on_outside_fields": text_contrast_ratio(ink, colors[2]),
            "count_labels_on_full_response": text_contrast_ratio(count_ink, colors[0]),
            "count_labels_on_selected_fields": text_contrast_ratio(count_ink, colors[1]),
            "count_labels_on_outside_fields": text_contrast_ratio(count_ink, colors[2]),
        },
        "checks": {"token_partition": True, "source_micro_ratio": True,
                   "value_label_overlap": False, "legend_entries": len(legend_labels),
                   "visible_text_count": len(text_artists), "edge_violations": edge_violations,
                   "crop_padding_points": crop_padding_points, "left_y_axis_visible": True,
                   "count_labels_inside_bars": True, "plot_top_bottom_alignment": True,
                   "equal_plot_widths": True, "single_line_panel_notes": True},
        "note": "Descriptive diagnostics. No new inference, significance tests, or invented intervals.",
    }
    stem.with_suffix(".data.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"output": str(stem), "checks": manifest["checks"],
                      "care_full_display": manifest["care_full_display"]}, indent=2))


if __name__ == "__main__":
    main()
