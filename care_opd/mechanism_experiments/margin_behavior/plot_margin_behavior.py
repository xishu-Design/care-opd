#!/usr/bin/env python3
"""Plot the completed CARE-OPD margin-to-behavior analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "care_localized": "#D55E00",
    "care_outside": "#0072B2",
    "care_minus_gkd_localized": "#009E73",
}


def _finite(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected numeric value")
    return float(value)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(payload: Mapping[str, Any], output_dir: Path) -> None:
    if payload.get("schema") != "care-opd-margin-behavior-v1":
        raise ValueError("unsupported analysis schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.05))
    quartiles = payload["quartile_recovery"]
    x = np.arange(1, 5)
    series = (
        ("care_localized", "CARE localized margin", "o"),
        ("care_outside", "CARE outside margin", "s"),
        ("care_minus_gkd_localized", "CARE−GKD localized delta", "^"),
    )
    for key, label, marker in series:
        rates = [
            100.0 * _finite(row["care_rescue_rate"], f"{key}.care_rescue_rate")
            for row in quartiles[key]
        ]
        axes[0].plot(
            x,
            rates,
            marker=marker,
            linewidth=2.0,
            markersize=5.5,
            color=COLORS[key],
            label=label,
        )
    axes[0].set_xticks(x, ["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    axes[0].set_ylabel("Actual CARE rescue rate (%)")
    axes[0].set_xlabel("Within-score rank quartile")
    axes[0].set_title("A. Parameter preference and realized rescue")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right")

    delta_rows = {row["region"]: row for row in payload["care_minus_gkd"]}
    regions = ["full", "localized", "outside"]
    labels = ["Full", "Localized", "Outside"]
    means = [_finite(delta_rows[key]["mean_delta"], f"{key}.mean_delta") for key in regions]
    lowers = [_finite(delta_rows[key]["ci95_lower"], f"{key}.ci95_lower") for key in regions]
    uppers = [_finite(delta_rows[key]["ci95_upper"], f"{key}.ci95_upper") for key in regions]
    errors = np.array(
        [[mean - lower for mean, lower in zip(means, lowers)],
         [upper - mean for mean, upper in zip(means, uppers)]]
    )
    bars = axes[1].bar(
        np.arange(3),
        means,
        yerr=errors,
        capsize=4,
        color=["#7A7A7A", "#D55E00", "#0072B2"],
        width=0.64,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].axhline(0, color="black", linewidth=1.0)
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylabel("CARE−GKD margin delta")
    axes[1].set_title("B. Paired preference gain (95% bootstrap CI)")
    for bar, mean in zip(bars, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:+.2f}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=8.5,
        )

    behavior = payload["behavior"]
    figure.suptitle(
        "CARE-OPD margin-to-behavior closure  |  "
        f"n={behavior['n']}, CARE rescue={100 * behavior['care_rescue_rate']:.1f}%",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"fig_margin_behavior_closure.{suffix}")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    plot(payload, args.output_dir)
    print(args.output_dir / "fig_margin_behavior_closure.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
