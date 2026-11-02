#!/usr/bin/env python3
"""Generate publication-ready CARE-OPD mechanism figures and tables.

The script contains only the frozen aggregate values reported by the completed
mechanism diagnostics.  It does not rerun models or invent confidence bounds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["Frozen", "Clean-SFT", "Recovery-SFT", "Recovery-DPO", "GKD-full", "CARE-OPD"]
COLORS = {
    "Frozen": "#9AA0A6",
    "Clean-SFT": "#56B4E9",
    "Recovery-SFT": "#009E73",
    "Recovery-DPO": "#E69F00",
    "GKD-full": "#0072B2",
    "CARE-OPD": "#D55E00",
}

RESCUE = [36.1888, 0.0, 36.1014, 57.2552, 70.4545, 91.6084]
REGRESSION = [74.7097, 0.0, 18.9768, 28.7736, 40.7475, 19.5573]
PASS_RATE = [28.4872, 70.6667, 67.8462, 67.1282, 62.5385, 83.7179]
CANDIDATE_PASSED = [1111, 2756, 2646, 2618, 2439, 3265]
RESCUED_N = [414, 0, 413, 655, 806, 1048]
REGRESSED_N = [2059, 0, 523, 793, 1123, 539]

MARGIN_METHODS = ["Clean-SFT", "GKD-full", "CARE-OPD"]
FULL_MARGIN = [-4.1842, -1.1055, 0.3150]
LOCAL_MARGIN = [-3.7073, -1.1368, 0.4963]
OUTSIDE_MARGIN = [-4.3732, -0.8385, 0.1495]

ROUNDS = ["Round 1", "Round 2", "Round 3"]
PASSED = [8650, 9004, 9269]
FORMAT = [94, 328, 54]
SEMANTIC = [3256, 2668, 2677]
PERSISTENT_ALL = [259, 1241]
DISAPPEARED_ALL = [3091, 1755]
NEW_ALL = [2737, 1490]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def fig_behavior(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.15), gridspec_kw={"width_ratios": [1.15, 1]},
                             layout="constrained")
    retention = 100 - np.array(REGRESSION)
    for i, method in enumerate(METHODS):
        axes[0].scatter(RESCUE[i], retention[i], s=78 if method == "CARE-OPD" else 48,
                        color=COLORS[method], edgecolor="white", linewidth=0.7, zorder=3)
    label_offsets = {
        "Frozen": (3, -1), "Clean-SFT": (3, -2), "Recovery-SFT": (-2, 5),
        "Recovery-DPO": (-16, 10), "GKD-full": (3, -6), "CARE-OPD": (0, 0),
    }
    for i, method in enumerate(METHODS):
        if method == "CARE-OPD":
            continue
        dx, dy = label_offsets[method]
        axes[0].annotate(method, (RESCUE[i], retention[i]), xytext=(dx, dy),
                         textcoords="offset points", fontsize=7.5,
                         fontweight="bold" if method == "CARE-OPD" else "normal")
    axes[0].set_xlabel("Rescue on Clean-SFT failures (%)")
    axes[0].set_ylabel("Retention of Clean-SFT successes (%)")
    axes[0].set_xlim(-4, 101)
    axes[0].set_ylim(18, 104)
    axes[0].set_title("(a) Recovery–retention trade-off")
    axes[0].annotate("91.61% rescue\n80.44% retention", (RESCUE[-1], retention[-1]),
                     xytext=(-73, 30), textcoords="offset points",
                     arrowprops={"arrowstyle": "->", "color": "#555"}, color="#8B2500", fontsize=7.5)

    net = np.array(RESCUED_N) - np.array(REGRESSED_N)
    bar_colors = [COLORS[m] for m in METHODS]
    bars = axes[1].barh(np.arange(len(METHODS)), net, color=bar_colors, height=0.58)
    axes[1].axvline(0, color="#444", linewidth=0.9)
    axes[1].set_yticks(np.arange(len(METHODS)), METHODS)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1800, 650)
    axes[1].set_xlabel("Net additional successes vs Clean-SFT")
    axes[1].set_title("(b) Rescued minus regressed")
    for bar, value in zip(bars, net):
        axes[1].text(value + 28, bar.get_y() + bar.get_height()/2,
                     f"{value:+d}", va="center", ha="left",
                     fontsize=8, fontweight="bold" if value > 0 else "normal")
    fig.suptitle("CARE-OPD uniquely yields a positive matched net gain", fontsize=11.5, fontweight="bold")
    save(fig, out, "fig1_behavior_rescue_regression")


def fig_localization_margin(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15), layout="constrained")
    selected = 63.6658
    outside = 36.3342
    axes[0].barh([0], [selected], color="#2A9D8F", height=0.5, label="Verifier-selected")
    axes[0].barh([0], [outside], left=[selected], color="#D0D5D8", height=0.5, label="Outside selected fields")
    axes[0].text(selected / 2, 0, "63.67%\nselected", ha="center", va="center", color="white", fontweight="bold")
    axes[0].text(selected + outside / 2, 0, "36.33%\nexcluded", ha="center", va="center", color="#444")
    axes[0].set_xlim(0, 100)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Response tokens (%)")
    axes[0].set_title("(a) Verifier-localized supervision")
    axes[0].text(50, -0.28, "Full response = 1.571× selected-token supervision",
                 ha="center", fontsize=8, transform=axes[0].get_xaxis_transform())

    x = np.arange(len(MARGIN_METHODS))
    width = 0.25
    for offset, values, label, color in [
        (-width, FULL_MARGIN, "Full response", "#56B4E9"),
        (0, LOCAL_MARGIN, "Selected fields", "#D55E00"),
        (width, OUTSIDE_MARGIN, "Outside fields", "#9AA0A6"),
    ]:
        bars = axes[1].bar(x + offset, values, width * 0.9, label=label, color=color)
        for b, v in zip(bars, values):
            axes[1].text(b.get_x() + b.get_width()/2, v + (0.10 if v >= 0 else -0.14), f"{v:.2f}",
                         ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    axes[1].axhline(0, color="#222", linewidth=1.0)
    axes[1].set_xticks(x, MARGIN_METHODS)
    axes[1].set_ylabel("Oracle − rejected margin")
    axes[1].set_title("(b) Frozen-parameter recovery preference")
    axes[1].legend(loc="lower right", fontsize=7.5)
    axes[1].set_ylim(-4.9, 1.15)
    fig.suptitle("Localized supervision and parameter-level preference reversal", fontsize=11.5, fontweight="bold")
    save(fig, out, "fig2_localization_and_margin")


def fig_rounds(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15), layout="constrained")
    x = np.arange(3)
    pass_rate = np.array(PASSED) / 120
    semantic_rate = np.array(SEMANTIC) / 120
    format_rate = np.array(FORMAT) / 120
    axes[0].plot(x, pass_rate, marker="o", color="#D55E00", label="Pass rate")
    axes[0].plot(x, semantic_rate, marker="s", color="#0072B2", label="Semantic failure")
    axes[0].plot(x, format_rate, marker="^", color="#9AA0A6", label="Format failure")
    axes[0].set_xticks(x, ROUNDS)
    axes[0].set_ylabel("Share of 12,000 rollouts (%)")
    axes[0].set_ylim(0, 82)
    axes[0].set_title("(a) Per-round outcome rates")
    axes[0].legend(loc="center right")
    for i, v in enumerate(pass_rate):
        axes[0].text(i, v + 1.5, f"{v:.2f}%", ha="center", color="#8B2500", fontsize=8)

    trans = ["R1 → R2", "R2 → R3"]
    y = np.arange(2)
    persistent = np.array(PERSISTENT_ALL, dtype=float)
    new = np.array(NEW_ALL, dtype=float)
    current = persistent + new
    persistent_pct = 100 * persistent / current
    new_pct = 100 * new / current
    axes[1].barh(y, persistent_pct, color="#0072B2", label="Persistent from prior round")
    axes[1].barh(y, new_pct, left=persistent_pct, color="#E69F00", label="New in current round")
    axes[1].set_yticks(y, trans)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Composition of current-round failures (%)")
    axes[1].set_title("(b) High failure-set turnover")
    closures = [92.27, 58.58]
    for i in range(2):
        persistent_label = f"{persistent_pct[i]:.2f}%" if i == 0 else f"{persistent_pct[i]:.2f}%\npersistent"
        axes[1].text(persistent_pct[i] / 2, i, persistent_label,
                     ha="center", va="center", color="white", fontsize=7.4)
        axes[1].text(persistent_pct[i] + new_pct[i] / 2, i, f"{new_pct[i]:.2f}%\nnew",
                     ha="center", va="center", color="#3B2A00", fontsize=7.4, fontweight="bold")
        axes[1].text(100, i - 0.34, f"Prior-failure closure: {closures[i]:.2f}%",
                     ha="right", va="center", fontsize=7.2, color="#555")
    fig.suptitle("Online rounds improve aggregate success while failures keep migrating", fontsize=11.5, fontweight="bold")
    save(fig, out, "fig3_online_round_dynamics")


def write_manifest(out: Path) -> None:
    manifest = {
        "schema": "care-opd-mechanism-figure-pack-v1",
        "claim_boundary": "aggregate descriptive figures; no invented confidence intervals",
        "figures": [
            "fig1_behavior_rescue_regression", "fig2_localization_and_margin", "fig3_online_round_dynamics"
        ],
        "presentation": "figures_only",
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    fig_behavior(args.output_dir)
    fig_localization_margin(args.output_dir)
    fig_rounds(args.output_dir)
    write_manifest(args.output_dir)
    print(f"Mechanism figure pack: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
