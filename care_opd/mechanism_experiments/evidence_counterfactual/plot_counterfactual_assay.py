#!/usr/bin/env python3
"""Render the CARE-OPD evidence-state counterfactual assay.

The plot consumes a frozen JSON source-data contract.  It does not score
rollouts, reconstruct missing model outputs, or substitute values from partial
archives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CARE = "#009E73"       # Okabe-Ito bluish green
GKD = "#0072B2"        # Okabe-Ito blue
CLEAN = "#7A7A7A"      # neutral baseline
INK = "#20262E"
MUTED = "#65717D"
GRID = "#D9DEE3"
HIGHLIGHT = "#E7F5F0"
UNSAFE_BG = "#FFF4E5"


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: numeric value required")
    return float(value)


def _validate(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    rows = payload.get("absolute_metrics")
    deltas = payload.get("paired_deltas")
    if not isinstance(rows, list) or not isinstance(deltas, list):
        raise ValueError("source JSON requires absolute_metrics and paired_deltas lists")
    indexed = {str(row.get("model")): row for row in rows if isinstance(row, Mapping)}
    required_models = ("clean_sft", "gkd_full", "care_opd")
    if set(indexed) != set(required_models):
        raise ValueError("absolute_metrics must contain exactly clean_sft, gkd_full, care_opd")
    metric_keys = (
        "evidence_present",
        "critical_safety",
        "restored",
        "irrelevant_history",
        "joint_four_state",
        "unsafe_green",
    )
    for model in required_models:
        row = indexed[model]
        if int(_number(row.get("groups"), f"{model}.groups")) != 200:
            raise ValueError(f"{model}.groups must be 200")
        for key in metric_keys:
            value = _number(row.get(key), f"{model}.{key}")
            if not 0 <= value <= 100:
                raise ValueError(f"{model}.{key}: percentage outside [0, 100]")
    if len(deltas) != 8:
        raise ValueError("paired_deltas must contain eight all-family comparisons")
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(deltas):
        if not isinstance(row, Mapping):
            raise ValueError(f"paired_deltas[{index}] must be an object")
        baseline = str(row.get("baseline"))
        metric = str(row.get("metric"))
        if baseline not in {"clean_sft", "gkd_full"}:
            raise ValueError(f"paired_deltas[{index}].baseline invalid")
        seen.add((baseline, metric))
        estimate = _number(row.get("delta"), f"paired_deltas[{index}].delta")
        low = _number(row.get("ci95_lower"), f"paired_deltas[{index}].ci95_lower")
        high = _number(row.get("ci95_upper"), f"paired_deltas[{index}].ci95_upper")
        if not low <= estimate <= high:
            raise ValueError(f"paired_deltas[{index}]: CI does not contain estimate")
        if int(_number(row.get("paired_groups"), f"paired_deltas[{index}].paired_groups")) != 200:
            raise ValueError(f"paired_deltas[{index}].paired_groups must be 200")
    if len(seen) != 8:
        raise ValueError("paired_deltas contains duplicate baseline/metric pairs")
    return [indexed[key] for key in required_models], list(deltas)


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#7C8791",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _draw_absolute_panel(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[tuple[str, str]],
    display: Mapping[str, str],
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    np: Any,
    *,
    title: str | None = None,
) -> tuple[list[Any], list[str]]:
    """Draw the six absolute endpoints on one Word-width axis."""
    y = np.arange(len(metrics))
    ax.axhspan(0.55, 1.45, color=HIGHLIGHT, zorder=0)
    ax.axhspan(3.55, 4.45, color=HIGHLIGHT, zorder=0)
    ax.axhspan(4.55, 5.45, color=UNSAFE_BG, zorder=0)
    for yi in y:
        ax.hlines(yi, 0, 100, color=GRID, linewidth=0.7, zorder=0)
    offsets = {"clean_sft": -0.18, "gkd_full": 0.0, "care_opd": 0.18}
    for row in rows:
        model = str(row["model"])
        values = [_number(row[key], f"{model}.{key}") for key, _ in metrics]
        yy = y + offsets[model]
        ax.scatter(
            values,
            yy,
            s=48 if model == "care_opd" else 36,
            marker=markers[model],
            color=colors[model],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label=display[model],
        )
        for (key, _), value, ypos in zip(metrics, values, yy, strict=True):
            # GKD and CARE differ by only 0.5 pp on unsafe green.  A small
            # label-only vertical nudge preserves the exact values and keeps
            # both annotations legible at Word rendering size.
            label_y = ypos
            if key == "unsafe_green":
                label_y += {"gkd_full": -0.07, "care_opd": 0.07}.get(model, 0.0)
            ax.text(
                value + 1.4,
                label_y,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color=colors[model],
                fontweight="bold" if model == "care_opd" else "normal",
            )
    ax.set_xlim(0, 108)
    ax.set_ylim(5.55, -0.55)
    ax.set_yticks(y, [label for _, label in metrics])
    for tick, (key, _) in zip(ax.get_yticklabels(), metrics, strict=True):
        if key in {"critical_safety", "joint_four_state"}:
            tick.set_fontweight("bold")
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Task success / error rate (%)")
    if title:
        ax.set_title(title, loc="left", pad=12)
    return ax.get_legend_handles_labels()


def _draw_effects_panel(
    ax: Any,
    deltas: Sequence[Mapping[str, Any]],
    np: Any,
    *,
    title: str | None = None,
) -> tuple[list[Any], list[str]]:
    """Draw paired CARE-minus-baseline confidence intervals on one axis."""
    from matplotlib.lines import Line2D

    metric_order = [
        ("critical_safety_tsr", "Critical safety"),
        ("four_state_joint_tsr", "Four-state joint"),
        ("restoration_retention_given_present", "Restoration retention"),
        ("irrelevant_history_retention_given_present", "History retention"),
    ]
    delta_index = {(str(r["baseline"]), str(r["metric"])): r for r in deltas}
    positions = np.arange(len(metric_order), dtype=float)
    comparison_offsets = {"clean_sft": -0.14, "gkd_full": 0.14}
    for metric_index, (metric, _metric_label) in enumerate(metric_order):
        for baseline in ("clean_sft", "gkd_full"):
            row = delta_index[(baseline, metric)]
            estimate = _number(row["delta"], "delta")
            low = _number(row["ci95_lower"], "ci95_lower")
            high = _number(row["ci95_upper"], "ci95_upper")
            color = CLEAN if baseline == "clean_sft" else GKD
            marker = "o" if baseline == "clean_sft" else "s"
            pos = positions[metric_index] + comparison_offsets[baseline]
            ax.errorbar(
                estimate,
                pos,
                xerr=[[estimate - low], [high - estimate]],
                fmt=marker,
                markersize=6.2,
                color=color,
                ecolor=color,
                elinewidth=1.6,
                capsize=3.0,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
            ax.text(
                high + 1.0,
                pos,
                f"{estimate:+.1f} [{low:+.1f}, {high:+.1f}]",
                va="center",
                fontsize=7.3,
                color=color,
            )
    ax.axvline(0, color=INK, linewidth=1.0, linestyle="--", zorder=1)
    ax.axvspan(0, 58, color=HIGHLIGHT, alpha=0.55, zorder=0)
    ax.set_yticks(positions, [label for _, label in metric_order])
    ax.set_ylim(positions[-1] + 0.55, positions[0] - 0.55)
    ax.set_xlim(-7, 58)
    ax.set_xticks(np.arange(-5, 56, 10))
    ax.set_xlabel("CARE-OPD minus baseline (percentage points)")
    if title:
        ax.set_title(title, loc="left", pad=12)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6.2,
            markerfacecolor=CLEAN,
            markeredgecolor="white",
            label="vs Clean-SFT",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=6.2,
            markerfacecolor=GKD,
            markeredgecolor="white",
            label="vs GKD-full",
        ),
    ]
    return handles, [handle.get_label() for handle in handles]


def _save_figure(fig: Any, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=320)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))


def plot(source: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    payload = _load(source)
    rows, deltas = _validate(payload)
    display = {"clean_sft": "Clean-SFT", "gkd_full": "GKD-full", "care_opd": "CARE-OPD"}
    colors = {"clean_sft": CLEAN, "gkd_full": GKD, "care_opd": CARE}
    markers = {"clean_sft": "o", "gkd_full": "s", "care_opd": "D"}

    metrics = [
        ("evidence_present", "Evidence present"),
        ("critical_safety", "Critical safety"),
        ("restored", "Evidence restored"),
        ("irrelevant_history", "Irrelevant history"),
        ("joint_four_state", "Four-state joint"),
        ("unsafe_green", "Unsafe green ↓"),
    ]

    _style(plt)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standalone figure 1: absolute outcomes.  This is the version intended for
    # Word; it does not depend on a neighboring panel for labels or legends.
    fig_absolute = plt.figure(figsize=(7.5, 5.0), constrained_layout=False)
    absolute_grid = fig_absolute.add_gridspec(
        1, 1, left=0.20, right=0.965, top=0.68, bottom=0.16
    )
    ax_absolute = fig_absolute.add_subplot(absolute_grid[0, 0])
    absolute_handles, absolute_labels = _draw_absolute_panel(
        ax_absolute, rows, metrics, display, colors, markers, np
    )
    fig_absolute.suptitle(
        "Absolute evidence-state behavior",
        x=0.20,
        y=0.965,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
    )
    fig_absolute.text(
        0.20,
        0.885,
        "CARE-OPD raises critical-safety and four-state joint success while ordinary-state performance remains high.",
        ha="left",
        fontsize=8.8,
        color=MUTED,
    )
    fig_absolute.legend(
        absolute_handles,
        absolute_labels,
        loc="upper left",
        bbox_to_anchor=(0.195, 0.80),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    absolute_stem = output_dir / "fig_evidence_state_counterfactual_absolute"
    _save_figure(fig_absolute, absolute_stem)
    plt.close(fig_absolute)

    # Standalone figure 2: paired effects.  Grouping both baselines within each
    # endpoint avoids long repeated y labels and preserves CI readability.
    fig_effects = plt.figure(figsize=(7.5, 4.65), constrained_layout=False)
    effects_grid = fig_effects.add_gridspec(
        1, 1, left=0.20, right=0.965, top=0.69, bottom=0.18
    )
    ax_effects = fig_effects.add_subplot(effects_grid[0, 0])
    effects_handles, effects_labels = _draw_effects_panel(ax_effects, deltas, np)
    fig_effects.suptitle(
        "Paired CARE-minus-baseline effects (95% CI)",
        x=0.20,
        y=0.96,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
    )
    fig_effects.text(
        0.20,
        0.875,
        "Group bootstrap over 200 matched groups; positive values favor CARE-OPD.",
        ha="left",
        fontsize=8.8,
        color=MUTED,
    )
    fig_effects.legend(
        effects_handles,
        effects_labels,
        loc="upper left",
        bbox_to_anchor=(0.195, 0.80),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=8.0,
    )
    effects_stem = output_dir / "fig_evidence_state_counterfactual_effects"
    _save_figure(fig_effects, effects_stem)
    plt.close(fig_effects)

    # Compatibility composite.  It is retained for existing references but is
    # not the artifact to embed in Word.
    fig = plt.figure(figsize=(7.5, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.05, 0.95],
        hspace=0.62,
        left=0.20,
        right=0.965,
        top=0.73,
        bottom=0.085,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    composite_absolute_handles, composite_absolute_labels = _draw_absolute_panel(
        ax_a,
        rows,
        metrics,
        display,
        colors,
        markers,
        np,
        title="a  Absolute evidence-state behavior",
    )
    fig.legend(
        composite_absolute_handles,
        composite_absolute_labels,
        loc="upper left",
        bbox_to_anchor=(0.195, 0.815),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    composite_effects_handles, _ = _draw_effects_panel(
        ax_b, deltas, np, title="b  Paired group-bootstrap effects (95% CI)"
    )
    ax_b.legend(
        handles=composite_effects_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.14),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=8.0,
    )
    fig.suptitle(
        "CARE-OPD improves safety-sensitive\nevidence-state control",
        x=0.20,
        y=0.975,
        ha="left",
        fontsize=14.2,
        fontweight="bold",
        linespacing=1.0,
    )
    fig.text(
        0.20,
        0.885,
        "Largest gains occur on critical safety and four-state consistency;\n"
        "ordinary-state retention remains high.",
        ha="left",
        fontsize=9.2,
        color=MUTED,
        linespacing=1.25,
    )
    fig.text(
        0.20,
        0.018,
        "Controlled state-sensitivity assay; verifier outcomes are not clinical evidence. Values are read from the frozen source-data contract.",
        ha="left",
        fontsize=7.5,
        color=MUTED,
    )
    _save_figure(fig, output_dir / "fig_evidence_state_counterfactual")
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plot(args.source, args.output_dir)
    root = args.output_dir.resolve()
    print(f"figure_absolute={root / 'fig_evidence_state_counterfactual_absolute'}")
    print(f"figure_effects={root / 'fig_evidence_state_counterfactual_effects'}")
    print(f"figure_compat={root / 'fig_evidence_state_counterfactual'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
