#!/usr/bin/env python3
"""Render publication figures for the additional CARE-OPD mechanism assays.

The script only reads completed JSON summaries.  It never substitutes missing
results with placeholders: each requested figure is either fully generated or
fails with a precise input-contract error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


OUR = "#D55E00"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
GRAY = "#9AA4AE"
INK = "#27313A"


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _rate(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: finite rate required")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{where}: rate outside [0, 1]")
    return result


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: Any, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{stem}.pdf")
    fig.savefig(root / f"{stem}.png", dpi=300)


def plot_causal(path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    payload = _load(path)
    models = payload.get("models")
    relative = payload.get("reference_conditioned_request_metrics")
    if not isinstance(models, Mapping) or not isinstance(relative, Mapping):
        raise ValueError(f"{path}: causal summary lacks models/reference metrics")
    preferred = [
        "full_selected",
        "without_retention",
        "matched_full_response",
        "outside_only",
        "without_risk",
        "without_trust",
    ]
    labels = [label for label in preferred if label in models]
    if len(labels) < 2 or set(labels) != set(models):
        raise ValueError(f"{path}: unexpected or incomplete causal arm set")
    display = {
        "full_selected": "CARE joint",
        "without_retention": "w/o retention",
        "matched_full_response": "full response",
        "outside_only": "outside only",
        "without_risk": "w/o risk",
        "without_trust": "w/o trust",
    }
    clean, perturbed, retention, rescue, regression = [], [], [], [], []
    for label in labels:
        metric = models[label].get("unified_metrics")
        rel = relative.get(label)
        if not isinstance(metric, Mapping) or not isinstance(rel, Mapping):
            raise ValueError(f"{path}: missing metrics for {label}")
        clean.append(100 * _rate(metric.get("clean_task_success_rate"), f"{label}.clean"))
        perturbed.append(100 * _rate(metric.get("perturbed_task_success_rate"), f"{label}.perturbed"))
        retention.append(100 * _rate(metric.get("conditional_retention"), f"{label}.retention"))
        rescue.append(100 * _rate(rel.get("rescue_rate"), f"{label}.rescue"))
        regression.append(100 * _rate(rel.get("regression_rate"), f"{label}.regression"))

    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.25
    axes[0].bar(x - width, clean, width, label="Clean TSR", color=SKY)
    axes[0].bar(x, perturbed, width, label="Perturbed TSR", color=OUR)
    axes[0].bar(x + width, retention, width, label="Retention", color=GREEN)
    axes[0].set_xticks(x, [display[label] for label in labels], rotation=24, ha="right")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("A  Equal-budget one-round ablation")
    axes[0].legend(ncol=3, loc="upper center")

    for index, label in enumerate(labels):
        color = OUR if label == "full_selected" else BLUE
        axes[1].scatter(regression[index], rescue[index], s=72, color=color, zorder=3)
        axes[1].annotate(
            display[label],
            (regression[index], rescue[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    max_axis = max([1.0, *rescue, *regression]) * 1.14
    axes[1].plot([0, max_axis], [0, max_axis], color=GRAY, linestyle="--", linewidth=1)
    axes[1].set_xlim(-0.5, max_axis)
    axes[1].set_ylim(-0.5, max_axis)
    axes[1].set_xlabel("Regression vs CARE joint (%)  ← better")
    axes[1].set_ylabel("Rescue vs CARE joint (%)  → better")
    axes[1].set_title("B  Rescue–regression trade-off")
    axes[1].text(
        0.98,
        0.03,
        "Above diagonal = positive net change",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=7.5,
        color=INK,
    )
    _save(fig, output_dir, "fig_causal_ablation")
    plt.close(fig)


def plot_counterfactual(path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    payload = _load(path)
    rows = payload.get("metrics")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: metrics list required")
    all_rows = {str(row.get("model")): row for row in rows if isinstance(row, Mapping) and row.get("family") == "all"}
    labels = [label for label in ("clean_sft", "gkd_full", "care_opd") if label in all_rows]
    if len(labels) != 3:
        raise ValueError(f"{path}: all-family rows for three models required")
    metrics = [
        ("evidence_present_tsr", "Evidence\npresent"),
        ("critical_safety_tsr", "Critical\nsafety"),
        ("evidence_restored_tsr", "Evidence\nrestored"),
        ("irrelevant_history_tsr", "Irrelevant\nhistory"),
        ("four_state_joint_tsr", "Four-state\njoint"),
    ]
    colors = {"clean_sft": GRAY, "gkd_full": BLUE, "care_opd": OUR}
    display = {"clean_sft": "Clean-SFT", "gkd_full": "GKD-full", "care_opd": "CARE-OPD"}
    _style(plt)
    fig, ax = plt.subplots(figsize=(7.5, 3.7), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.24
    for index, label in enumerate(labels):
        values = [100 * _rate(all_rows[label].get(key), f"{label}.{key}") for key, _ in metrics]
        bars = ax.bar(x + (index - 1) * width, values, width, label=display[label], color=colors[label])
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Verifier task success (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Matched evidence-state counterfactual assay")
    ax.legend(ncol=3, loc="upper center")
    ax.text(
        0.0,
        -0.22,
        "Four-state joint requires A/B/C/D all correct within the same evidence group.",
        transform=ax.transAxes,
        fontsize=8,
        color=INK,
    )
    _save(fig, output_dir, "fig_evidence_counterfactual")
    plt.close(fig)


def plot_efficiency(path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    payload = _load(path)
    rows = payload.get("models")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"{path}: models list required")
    indexed = {str(row.get("label")): row for row in rows if isinstance(row, Mapping)}
    labels = [label for label in ("clean_sft", "gkd_full", "care_opd") if label in indexed]
    if len(labels) != 3:
        raise ValueError(f"{path}: clean_sft/gkd_full/care_opd required")
    latency, throughput, memory = [], [], []
    for label in labels:
        row = indexed[label]
        measurements = row.get("measurements")
        mem = row.get("memory")
        if not isinstance(measurements, Mapping) or not isinstance(mem, Mapping):
            raise ValueError(f"{path}: incomplete efficiency row for {label}")
        timing = measurements.get("generation_latency_seconds")
        if not isinstance(timing, Mapping):
            raise ValueError(f"{path}: missing latency for {label}")
        latency.append(float(timing["p95"]))
        throughput.append(float(measurements["generated_tokens_per_second"]))
        memory.append(float(mem["peak_allocated_bytes"]) / 2**30)

    display = ["Clean-SFT", "GKD-full", "CARE-OPD"]
    colors = [GRAY, BLUE, OUR]
    _style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.25), constrained_layout=True)
    panels = [
        (latency, "p95 latency (s)", "A  Tail latency", "lower"),
        (throughput, "Generated tokens/s", "B  Throughput", "higher"),
        (memory, "Peak allocated (GiB)", "C  Peak memory", "lower"),
    ]
    x = np.arange(3)
    for ax, (values, ylabel, title, direction) in zip(axes, panels, strict=True):
        bars = ax.bar(x, values, color=colors, width=0.62)
        ax.set_xticks(x, display, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        upper = max(values) * 1.22 if max(values) > 0 else 1
        ax.set_ylim(0, upper)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + upper * 0.025, f"{value:.2f}", ha="center", fontsize=7.5)
        ax.text(0.98, 0.95, f"{direction} is better", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK)
    _save(fig, output_dir, "fig_deployment_efficiency")
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-summary", type=Path)
    parser.add_argument("--counterfactual-summary", type=Path)
    parser.add_argument("--efficiency-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requested = 0
    if args.causal_summary:
        plot_causal(args.causal_summary, args.output_dir)
        requested += 1
    if args.counterfactual_summary:
        plot_counterfactual(args.counterfactual_summary, args.output_dir)
        requested += 1
    if args.efficiency_summary:
        plot_efficiency(args.efficiency_summary, args.output_dir)
        requested += 1
    if not requested:
        raise SystemExit("provide at least one completed summary JSON")
    print(f"figures={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
