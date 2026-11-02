#!/usr/bin/env python3
"""Render archived localization and margin-rank diagnostics without inference.

Outputs: fig_mechanism_detail.{pdf,png,svg,data.json}, beside this script.
All numerical values are read from frozen CSV sources; no fitted or smoothed
curves, new bootstrap estimates, or invented confidence intervals are used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "care_opd_mechanism_detail_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
LOCALIZATION = ROOT / "runs/mechanism_eval/qzbc_care_opd_v1/localization/localization_analysis.csv"
MARGINS = ROOT / "runs/mechanism_eval/qzbc_care_opd_v1/margin_behavior/margin_behavior_summary.csv"
STEM = OUT / "fig_mechanism_detail"
SELECTED = "#0072B2"
OUTSIDE = "#009E73"
INK = "#1F2D40"
MUTED = "#5C6570"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_metadata(path: Path, fields: list[str]) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fields": fields,
    }


def extract_data() -> tuple[list[dict], dict[str, list[dict]]]:
    source_rows = read_csv(LOCALIZATION)
    order = [
        ("bad_history", "Bad history"),
        ("clean", "Clean"),
        ("evidence_conflict", "Evidence conflict"),
        ("source_outage", "Source outage"),
    ]
    localization = []
    for group, label in order:
        matches = [r for r in source_rows if r["scope"] == "by_variant" and r["group"] == group]
        assert len(matches) == 1, (group, len(matches))
        row = matches[0]
        total = int(row["full_response_tokens"])
        selected = int(row["selected_tokens"])
        outside = int(row["outside_selected_tokens"])
        n = int(row["rows"])
        fallback = int(row["full_fallback_rows"])
        assert selected + outside == total
        assert abs(selected / total - float(row["token_coverage_micro"])) < 1e-12
        assert abs(fallback / n - float(row["full_fallback_rate"])) < 1e-12
        localization.append({
            "group": group, "label": label, "n_failure_rows": n,
            "full_response_tokens": total, "selected_tokens": selected,
            "outside_tokens": outside, "selected_percent": 100 * selected / total,
            "outside_percent": 100 * outside / total,
            "full_fallback_rows": fallback, "full_fallback_percent": 100 * fallback / n,
        })

    margin_rows = read_csv(MARGINS)
    quartiles = {}
    for score in ["care_localized", "care_outside"]:
        matched = [r for r in margin_rows if r["section"] == "quartile" and r["score"] == score]
        matched.sort(key=lambda r: int(r["quartile"]))
        assert [int(r["quartile"]) for r in matched] == [1, 2, 3, 4]
        series = []
        for row in matched:
            n = int(row["n"])
            rescued = int(row["care_rescued"])
            rate = float(row["care_rescue_rate"])
            assert abs(rescued / n - rate) < 1e-12
            series.append({
                "quartile": int(row["quartile"]), "n": n, "rescued": rescued,
                "rescue_rate": rate, "rescue_percent": 100 * rate,
                "score_min": float(row["score_min"]),
                "score_max": float(row["score_max"]),
                "score_mean": float(row["score_mean"]),
            })
        assert [r["n"] for r in series] == [48, 47, 47, 47]
        assert sum(r["rescued"] for r in series) == 164
        quartiles[score] = series
    assert sum(r["n_failure_rows"] for r in localization) == 1144
    return localization, quartiles


def main() -> None:
    localization, quartiles = extract_data()
    arial_path = font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.edgecolor": "#6C747C",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "care-opd-mechanism-detail-v1",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 300, "savefig.dpi": 300,
    })
    fig = plt.figure(figsize=(7.16, 2.7), facecolor="white")
    a = fig.add_axes([0.145, 0.285, 0.292, 0.510])
    b = fig.add_axes([0.667, 0.285, 0.306, 0.510])
    fig.text(0.025, 0.946, "(a) Responsibility coverage by evidence state", weight="bold", va="top")
    fig.text(0.617, 0.946, "(b) CARE-OPD rescue by margin rank", weight="bold", va="top")

    # Panel a: exact token partition, not a causal outcome comparison.
    y = list(range(len(localization)))
    selected = [r["selected_percent"] for r in localization]
    outside = [r["outside_percent"] for r in localization]
    a.barh(y, selected, height=0.61, color=SELECTED, edgecolor="white", linewidth=0.45, zorder=3)
    a.barh(y, outside, left=selected, height=0.61, color=OUTSIDE,
           edgecolor="white", linewidth=0.45, hatch="//", zorder=3)
    a.set_yticks(y, [r["label"] for r in localization])
    a.tick_params(axis="y", length=0, pad=5)
    a.set_xlim(0, 100)
    a.set_ylim(3.65, -0.65)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel("Response tokens (%)", labelpad=5)
    a.spines["left"].set_visible(False)
    a.grid(axis="x", color="#E4E9ED", linewidth=0.45, zorder=0)
    for pos, row in enumerate(localization):
        a.text(row["selected_percent"] / 2, pos, f'{row["selected_percent"]:.1f}',
               ha="center", va="center", color="white")
        a.text(row["selected_percent"] + row["outside_percent"] / 2, pos,
               f'{row["outside_percent"]:.1f}', ha="center", va="center", color="white",
               bbox={"facecolor": OUTSIDE, "edgecolor": "none", "pad": 0.15})
        a.text(1.105, pos, str(row["n_failure_rows"]), transform=a.get_yaxis_transform(),
               ha="center", va="center", clip_on=False)
        a.text(1.365, pos, f'{row["full_fallback_percent"]:.1f}%', transform=a.get_yaxis_transform(),
               ha="center", va="center", clip_on=False)
    a.text(1.105, 1.022, "n", transform=a.transAxes, ha="center", va="bottom", color=MUTED)
    a.text(1.365, 1.022, "Fallback", transform=a.transAxes, ha="center", va="bottom", color=MUTED)
    a.legend(handles=[Patch(facecolor=SELECTED, label="Selected"),
                      Patch(facecolor=OUTSIDE, edgecolor="white", hatch="//", label="Outside")],
             loc="lower left", bbox_to_anchor=(-0.01, 1.008), ncol=2,
             frameon=False, handlelength=1.15, handletextpad=0.4,
             columnspacing=0.9, borderaxespad=0)

    # Panel b: independently ranked quartile groups for each actual score.
    for score, color, marker, linestyle, label in [
        ("care_localized", SELECTED, "o", "-", "Selected"),
        ("care_outside", OUTSIDE, "s", "--", "Outside"),
    ]:
        series = quartiles[score]
        xs = [r["quartile"] for r in series]
        ys = [r["rescue_percent"] for r in series]
        b.plot(xs, ys, color=color, marker=marker, linestyle=linestyle,
               linewidth=1.25, markersize=3.8, markerfacecolor="white",
               markeredgewidth=0.9, label=label, zorder=3)
        for x, value in zip(xs, ys):
            delta = 5 if score == "care_localized" else -11
            if score == "care_localized" and x == 1:
                delta = -12
            txt = f"{value:.1f}" if value != 100 else "100"
            b.annotate(txt, (x, value), xytext=(0, delta), textcoords="offset points",
                       ha="center", va="bottom" if delta > 0 else "top", color=color)
    b.set_xlim(0.75, 4.25)
    b.set_ylim(0, 115)
    b.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylabel("Rescued (%)", labelpad=4)
    b.set_xlabel("Within-score margin quartile", labelpad=5)
    b.grid(axis="y", color="#E4E9ED", linewidth=0.45, zorder=0)
    b.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.008), frameon=False,
             ncol=2, handlelength=1.5, handletextpad=0.4, columnspacing=0.8, borderaxespad=0)

    fig.text(0.317, 0.091, "1,144 Clean-SFT failures; token-weighted coverage", ha="center", color=MUTED)
    fig.text(0.806, 0.091, "n = 48 / 47 / 47 / 47 per score", ha="center", color=MUTED)
    fig.text(0.806, 0.037, "189 paired failures; descriptive association", ha="center", color=MUTED)

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
        fig.savefig(STEM.with_suffix("." + extension), dpi=300, bbox_inches=None, pad_inches=0)

    payload = {
        "schema": "care-opd-mechanism-detail-v1",
        "sources": [
            source_metadata(LOCALIZATION, ["scope", "group", "rows", "full_response_tokens", "selected_tokens",
                                          "outside_selected_tokens", "token_coverage_micro", "full_fallback_rows", "full_fallback_rate"]),
            source_metadata(MARGINS, ["section", "score", "quartile", "n", "care_rescued", "care_rescue_rate",
                                     "score_min", "score_mean", "score_max"]),
        ],
        "figure_size_inches": [7.16, 2.7], "font": {"family": "Arial", "size_pt": 8, "resolved_path": arial_path},
        "colors": {"selected": SELECTED, "outside": OUTSIDE},
        "panel_a": {"filter": "scope=by_variant", "rows": localization,
                    "coverage_unit": "tokens, micro-average within state",
                    "n_unit": "failed Clean-SFT responses", "fallback_unit": "failed responses",
                    "claim_boundary": "Descriptive mask implementation audit, not a performance ablation."},
        "panel_b": {"filter": "section=quartile; score in care_localized,care_outside", "series": quartiles,
                    "ranking": "Independent within-score rank quartiles; ties resolved by request ID in the archived analysis.",
                    "n_localizable": 189, "rescued": 164, "unrescued": 25,
                    "interpolation": "Straight connections between four observed values only; no smoothing.",
                    "confidence_intervals": "None computed or displayed.",
                    "claim_boundary": "Failure-conditioned association on a frozen cohort; not causal mediation or calibrated confidence."},
        "checks": {"token_partition": True, "rates_match_counts": True, "quartile_counts_match": True,
                   "no_text_clipped": True, "no_new_training_or_inference": True},
    }
    STEM.with_suffix(".data.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plt.close(fig)
    print("Saved fig_mechanism_detail.pdf/png/svg/data.json; no text clipped.")


if __name__ == "__main__":
    main()
