#!/usr/bin/env python3
"""Replot the archived localization audit using unchanged data extraction."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_localization"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

from gen_agent_figures import style, GREEN, INK, REPO

OUTSIDE_BLUE, OUTSIDE_FILL = "#0072B2", "#DCEEF8"


def main() -> None:
    original_script = OUT.parent / "gen_fig_mechanism_detail.py"
    spec = importlib.util.spec_from_file_location("archived_mechanism_detail", original_script)
    archived = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(archived)
    localization, quartiles = archived.extract_data()
    style()
    fig = plt.figure(figsize=(7.16, 2.60))
    a = fig.add_axes([0.145, 0.225, 0.275, 0.600])
    b = fig.add_axes([0.680, 0.225, 0.290, 0.600])
    fig.text(0.025, 0.967, "(a) Responsibility-token coverage", va="top", fontsize=8.5)
    fig.text(0.596, 0.967, "(b) CARE-OPD rescue by margin rank", va="top", fontsize=8.5)

    y = np.arange(len(localization))
    selected = [r["selected_percent"] for r in localization]
    outside = [r["outside_percent"] for r in localization]
    a.barh(y, selected, height=0.61, color=GREEN, edgecolor="white", linewidth=0.4, zorder=3)
    a.barh(y, outside, left=selected, height=0.61, color=OUTSIDE_FILL,
           edgecolor=OUTSIDE_BLUE, linewidth=0.35, hatch="///", zorder=3)
    a.set_yticks(y, [r["label"] for r in localization])
    a.set_xlim(0, 100)
    a.set_ylim(3.65, -0.65)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel("Response tokens (%)", labelpad=3)
    a.tick_params(axis="y", length=0, pad=4)
    a.tick_params(axis="x", length=2.5, pad=2)
    a.grid(axis="x", color="#D9D9D9", linewidth=0.45, zorder=0)
    for spine in ("left", "right", "top"):
        a.spines[spine].set_visible(False)
    for pos, row in enumerate(localization):
        a.text(row["selected_percent"] / 2, pos, f'{row["selected_percent"]:.1f}',
               ha="center", va="center", fontsize=8, color=INK, zorder=4)
        a.text(row["selected_percent"] + row["outside_percent"] / 2, pos,
               f'{row["outside_percent"]:.1f}', ha="center", va="center", fontsize=8, color=INK,
               bbox={"facecolor": OUTSIDE_FILL, "edgecolor": "none", "pad": 0.12}, zorder=4)
        a.text(1.105, pos, str(row["n_failure_rows"]), transform=a.get_yaxis_transform(),
               ha="center", va="center", fontsize=8, clip_on=False)
        a.text(1.360, pos, f'{row["full_fallback_percent"]:.1f}%', transform=a.get_yaxis_transform(),
               ha="center", va="center", fontsize=8, clip_on=False)
    a.text(1.105, 1.025, "n", transform=a.transAxes, ha="center", va="bottom", fontsize=8)
    a.text(1.360, 1.025, "Fallback", transform=a.transAxes, ha="center", va="bottom", fontsize=8)
    fig.legend(handles=[Patch(facecolor=GREEN, label="Selected"),
                        Patch(facecolor=OUTSIDE_FILL, edgecolor=OUTSIDE_BLUE, hatch="///", label="Outside")],
               loc="upper left", bbox_to_anchor=(0.145, 0.912), ncol=2, borderaxespad=0,
               handlelength=1.1, handletextpad=0.4, columnspacing=0.9, fontsize=8)

    for score, color, marker, offset, label in [
        ("care_localized", GREEN, "o", -0.10, "Selected score"),
        ("care_outside", OUTSIDE_BLUE, "s", 0.10, "Outside score"),
    ]:
        series = quartiles[score]
        xs = np.asarray([r["quartile"] for r in series], dtype=float) + offset
        ys = [r["rescue_percent"] for r in series]
        b.scatter(xs, ys, marker=marker, s=22, edgecolor=color, linewidth=0.9,
                  facecolor=color if score == "care_localized" else "white", zorder=4)
        for x, value in zip(xs, ys):
            label_y = 4 if score == "care_localized" else -6
            b.annotate(f"{value:.1f}", (x, value), xytext=(0, label_y), textcoords="offset points",
                       ha="center", va="bottom" if label_y > 0 else "top",
                       fontsize=7.5, color=INK)
    b.set_xlim(0.62, 4.38)
    b.set_ylim(0, 114)
    b.set_xticks([1, 2, 3, 4], ["Q1\nn=48", "Q2\nn=47", "Q3\nn=47", "Q4\nn=47"])
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_xlabel("Quartile within each score", labelpad=3)
    b.set_ylabel("Rescued (%)", labelpad=3)
    b.tick_params(length=2.5, pad=2)
    b.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)
    b.set_axisbelow(True)
    for spine in ("top", "right"):
        b.spines[spine].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=GREEN, marker="o", linestyle="none", markersize=4,
                               label="Selected score"),
                        Line2D([], [], color=OUTSIDE_BLUE, marker="s", markerfacecolor="white", linestyle="none",
                               markersize=4, label="Outside score")],
               loc="upper left", bbox_to_anchor=(0.650, 0.912), ncol=2, borderaxespad=0,
               handlelength=1, handletextpad=0.3, columnspacing=0.8, fontsize=8)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    clipped = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > width + 0.5 or box.y1 > height + 0.5:
            clipped.append(artist.get_text())
    assert not clipped, f"Text extends beyond canvas: {clipped}"
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"fig_localization_audit.{extension}", dpi=300)
    payload = {
        "schema": "tmm-localization-audit-v3",
        "sources": [archived.source_metadata(archived.LOCALIZATION, ["scope", "group", "rows", "selected_tokens", "outside_selected_tokens", "full_response_tokens", "full_fallback_rows"]),
                    archived.source_metadata(archived.MARGINS, ["section", "score", "quartile", "n", "care_rescued", "care_rescue_rate", "score_min", "score_mean", "score_max"])],
        "extraction_script": {"path": original_script.relative_to(REPO).as_posix(),
                              "sha256": hashlib.sha256(original_script.read_bytes()).hexdigest()},
        "size_inches": [7.16, 2.60], "font": "Times New Roman", "minimum_text_pt": 7.5,
        "color_encoding": {"selected": GREEN, "outside": OUTSIDE_BLUE, "outside_fill": OUTSIDE_FILL},
        "panel_a": {"rows": localization, "coverage_unit": "token-weighted within evidence state",
                    "n_unit": "failed Clean-SFT responses", "n_total": 1144,
                    "claim_boundary": "Mask implementation audit, not a performance ablation."},
        "panel_b": {"series": quartiles, "ranking": "Independent within-score quartiles; archived request-ID tie break.",
                    "n_localizable": 189, "rescued": 164, "unrescued": 25,
                    "interpolation": "None: offset discrete markers only; no connecting lines or smoothing.",
                    "confidence_intervals": None,
                    "claim_boundary": "Descriptive failure-conditioned association, not causal mediation or calibrated confidence."},
        "checks": {"original_extraction_reused": True, "rates_match_counts": True,
                   "no_text_clipped": True, "no_new_training_or_inference": True},
    }
    (OUT / "fig_localization_audit.source.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print("Wrote fig_localization_audit.pdf/png/svg/source.json; source data unchanged.")


if __name__ == "__main__":
    main()


