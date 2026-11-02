#!/usr/bin/env python3
"""Render external-field and model-call latency figures from archived CSVs.

The grouped-bar grammar follows the CCFA plotting-recipe contract, ported to
Matplotlib for exact physical sizes, type-42 PDF, and editable SVG text.
This script performs no training, inference, or new statistical estimation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_external_deployment"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
from figure_style import apply_style, AXIS, GRID, INK, MUTED, CLEAN, GKD, CARE

METHODS = ["clean_sft", "gkd_full", "care_opd"]
LABELS = {"clean_sft": "Clean-SFT", "gkd_full": "GKD-Full", "care_opd": "CARE-OPD"}
COLORS = {"clean_sft": CLEAN, "gkd_full": GKD, "care_opd": CARE}
HATCHES = {"clean_sft": "", "gkd_full": "///", "care_opd": ".."}
EXTERNAL = {
    "Structured": REPO / "eval_direct_gold_structured_six_models_v5/public/secondary_field_metrics.csv",
    "Raw strict": REPO / "eval_direct_gold_raw_six_models_v5/public/secondary_field_metrics.csv",
}
DEPLOYMENT = REPO / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/02_完整统计/DEPLOYMENT_AUDIT_FULL.csv"
STRATA = ["variant:clean", "variant:bad_history", "overall"]
SCOPES = ["clean", "bad_history", "clean+bad_history"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def provenance(path: Path) -> dict:
    return {"path": path.relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def style() -> None:
    apply_style()


def clean_axes(ax, grid_axis: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.45)
    ax.tick_params(length=2.5, pad=2.5)


def save(fig, stem: str, data: dict) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            assert box.x0 >= -0.5 and box.y0 >= -0.5 and box.x1 <= fig.bbox.width + 0.5 and box.y1 <= fig.bbox.height + 0.5, (stem, item.get_text(), box)
    # Do not use bbox_inches='tight': it silently changes physical dimensions.
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300, bbox_inches=None)
    data.update({"figure_size_inches": fig.get_size_inches().tolist(),
                 "style": {"font": "Arial", "axis_font_pt": 8.0,
                           "tick_font_pt": 7.6, "pdf_fonttype": 42,
                           "colors": COLORS, "reference": "Method diagnostics v21",
                           "ink": INK, "axis": AXIS, "grid": GRID}})
    (OUT / f"{stem}.data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)


def external_data() -> dict:
    result = {}
    for track, path in EXTERNAL.items():
        rows = load_csv(path)
        selected = []
        for method in METHODS:
            triples = []
            for stratum, scope in zip(STRATA, SCOPES):
                matches = [row for row in rows if row["model"] == method and row["stratum"] == stratum]
                assert len(matches) == 1, (track, method, stratum)
                row = matches[0]
                assert row["scope"] == scope
                record = {"model": method, "stratum": stratum, "scope": scope,
                          "source_csv_line": rows.index(row) + 2,
                          "field_micro_f1": float(row["field_micro_f1"])}
                for key in ("reports", "eligible_reports", "eligible_fields", "field_tp", "field_fp", "field_fn"):
                    record[key] = int(row[key])
                tp, fp, fn = (record[key] for key in ("field_tp", "field_fp", "field_fn"))
                assert math.isclose(2 * tp / (2 * tp + fp + fn), record["field_micro_f1"], abs_tol=1e-12)
                triples.append(record)
            for key in ("reports", "eligible_reports", "eligible_fields", "field_tp", "field_fp", "field_fn"):
                assert triples[0][key] + triples[1][key] == triples[2][key], (track, method, key)
            selected.extend(triples)
        result[track] = {"source": provenance(path), "rows": selected,
                         "families": 65 if track == "Structured" else 54,
                         "combined_validation": "overall TP/FP/FN and denominators equal clean plus bad_history; micro-F1 verified from pooled counts"}
    return result


def plot_external() -> None:
    data = external_data()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.80), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.988, bottom=0.17, top=0.75, wspace=0.19)
    positions = np.arange(3)
    bar_width = 0.235
    for idx, (ax, (track, payload)) in enumerate(zip(axes, data.items())):
        for j, method in enumerate(METHODS):
            values = [100 * next(row["field_micro_f1"] for row in payload["rows"]
                                 if row["model"] == method and row["stratum"] == stratum)
                      for stratum in STRATA]
            x = positions + (j - 1) * bar_width
            bars = ax.bar(x, values, width=bar_width * 0.93, color=COLORS[method],
                          edgecolor="white", linewidth=0.45, hatch=HATCHES[method], zorder=3)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, value + 1.4, f"{value:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks(positions, ["Clean", "History", "Clean + History"])
        ax.set_title(f"({chr(97 + idx)}) {track} ({payload['families']} families)", loc="left", pad=5, fontweight="bold")
        ax.tick_params(axis="x", length=0)
        clean_axes(ax, "y")
    axes[0].set_ylabel("Field micro-F1 (%)", labelpad=3)
    # Identical numerical axes make the much lower Raw strict absolute level visible.
    axes[1].tick_params(axis="y", labelleft=True)
    handles = [Patch(facecolor=COLORS[m], edgecolor="white", linewidth=0.4,
                     hatch=HATCHES[m], label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 0.996),
               handlelength=1.55, columnspacing=2.1, handletextpad=0.55)
    save(fig, "fig_external_faithfulness", {
        "metric": "secondary_field_metrics.field_micro_f1", "display_unit": "percent",
        "display_precision_decimals": 1, "tracks": data,
        "interpretation": "Clean+History is pooled field micro-F1, not the mean of two F1 scores. Cohorts are distinct overlapping tracks, not an intervention pair. No confidence intervals are plotted.",
        "caption": "External field faithfulness on Structured (65 families) and Raw strict (54 families), using the same 0--100% scale. History denotes misleading history. Clean+History pools field TP/FP/FN across the two states rather than averaging their F1 scores. The fixed Clean-SFT and GKD-Full comparators are shown with CARE-OPD; the supplement retains the complete six-model state results. These tracks differ in eligible fields and do not establish a causal preprocessing benefit."
    })


def plot_deployment(*, height: float = 1.55, bottom: float = 0.24,
                    top: float = 0.82) -> None:
    rows = load_csv(DEPLOYMENT)
    data = []
    fig, ax = plt.subplots(figsize=(3.49, height))
    fig.subplots_adjust(left=0.215, right=0.976, bottom=bottom, top=top)
    for idx, method in enumerate(METHODS):
        matching = [row for row in rows if row["model"] == method]
        assert len(matching) == 1
        row = matching[0]
        median, p95 = float(row["p50_s"]), float(row["p95_s"])
        assert int(row["n"]) == 200 and median <= p95
        y = 2 - idx
        ax.plot([median, p95], [y, y], color=COLORS[method], linewidth=1.5, zorder=2)
        ax.scatter([median], [y], s=29, marker="o", facecolor=COLORS[method],
                   edgecolor=COLORS[method], linewidth=0.8, zorder=3)
        ax.scatter([p95], [y], s=28, marker="s", facecolor="white",
                   edgecolor=COLORS[method], linewidth=1.0, zorder=3)
        for value in (median, p95):
            ax.annotate(f"{value:.2f}", (value, y), xytext=(0, -7.0),
                        textcoords="offset points", ha="center", va="top", fontsize=7.5)
        data.append({"model": method, "median_s": median, "p95_s": p95,
                     "n": int(row["n"]), "source_csv_line": rows.index(row) + 2,
                     "mean_output_tokens": float(row["mean_output_tokens"]),
                     "parameters": int(row["parameters"]), "weights_gib": float(row["weights_gib"])})
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(-0.85, 2.45)
    ax.set_yticks([2, 1, 0], [LABELS[m] for m in METHODS])
    ax.set_xlabel("Model-call latency (s)", labelpad=4)
    clean_axes(ax, "x")
    ax.tick_params(axis="y", length=0, pad=5)
    handles = [Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=4.5, label="Median"),
               Line2D([], [], color=MUTED, marker="s", markerfacecolor="white", linestyle="none", markersize=4.5, label="P95")]
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.61, 0.985),
               handletextpad=0.5, columnspacing=1.6)
    save(fig, "fig_deployment", {
        "source": provenance(DEPLOYMENT), "rows": data,
        "display_unit": "seconds", "display_precision_decimals": 2,
        "interval_semantics": "Segment connects the sample median and 95th percentile. It is not a confidence interval.",
        "caption": "Isolated model-call latency on one A800 GPU for three 4.2058B-parameter checkpoints, measured on 200 requests per checkpoint after five warm-ups. Filled circles mark medians and hollow squares mark 95th percentiles; connecting segments are not confidence intervals. Measurements exclude retrieval, tool execution, memory, and human confirmation and therefore are not end-to-end Agent latency."
    })


if __name__ == "__main__":
    style()
    plot_external()
    plot_deployment()
    print("Rendered fig_external_faithfulness and fig_deployment from validated archived CSV rows.")

