#!/usr/bin/env python3
"""Plot archived fixed-Clean-SFT transitions as absolute diverging counts.

Visual contract: five methods, one common reference partition, rescued counts
to the right and regressed counts to the left; all tick labels are nonnegative
magnitudes. A separate net-count text column preserves the main comparison.
The two conditioning denominators are explicit. This adapts the CCFA direct-
label categorical-bar recipe; no rates are pooled and no inference is added.

Only sibling fig_fixed_transitions.pdf/png/svg/data.json files are written.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "care_opd_tmm_v2_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
SOURCE_REL = ("competition_project_document/formal/competition_submission_v9/"
              "supporting_public_v9/02_完整统计/QZBC_STATE_AND_TRANSITION_FULL.csv")
MODELS = ["Frozen", "Recovery-SFT", "Recovery-DPO", "GKD-Full", "CARE-OPD"]
GREEN, ORANGE, INK = "#009E73", "#E69F00", "#222222"
CAPTION = (
    "Transitions relative to one fixed Clean-SFT partition of the 3,900 QZBC "
    "requests. Rightward green bars count rescued requests among 1,144 reference "
    "failures; leftward orange bars count regressed requests among 2,756 reference "
    "successes. Axis labels report absolute counts: the leftward direction does "
    "not denote negative counts. Net is rescued minus regressed, yielding +509 "
    "for CARE-OPD. The two subsets have different denominators, so these counts "
    "are not compared as conditional rates or pooled into a conditional rate."
)


def main():
    source = ROOT / SOURCE_REL
    with source.open(encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.DictReader(f))
    records = {r["model"]: r for r in all_rows
               if r["record_type"] == "transition" and r["state_or_slice"] == "vs_Clean-SFT_all_four_states"}
    ref = records["Clean-SFT"]
    assert int(ref["successes"]) == 2756 and int(ref["failures"]) == 1144
    data = []
    for model in MODELS:
        row = records[model]
        item = {"model": model, **{key: int(row[key]) for key in
                ["n", "successes", "failures", "reference_failures", "rescued",
                 "reference_successes", "regressed", "net_success_change"]}}
        assert item["reference_failures"] == 1144 and item["reference_successes"] == 2756
        assert item["reference_failures"] + item["reference_successes"] == item["n"] == 3900
        assert 0 <= item["rescued"] <= 1144 and 0 <= item["regressed"] <= 2756
        assert item["rescued"] - item["regressed"] == item["net_success_change"]
        assert item["successes"] - int(ref["successes"]) == item["net_success_change"]
        assert item["successes"] + item["failures"] == 3900
        data.append(item)
    assert data[-1]["net_success_change"] == 509

    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "care-opd-tmm-v2-fixed-transitions", "savefig.dpi": 300,
        "savefig.bbox": None, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(3.49, 2.6), dpi=300)
    ax = fig.add_axes([0.285, 0.265, 0.545, 0.575])
    ys = list(range(4, -1, -1))
    ax.axvline(0, color="#555555", lw=0.65, zorder=1)
    for y, row in zip(ys, data):
        rescued, regressed = row["rescued"], row["regressed"]
        ax.barh(y, -regressed, height=0.57, color=ORANGE, edgecolor="white", lw=0.6, zorder=2)
        ax.barh(y, rescued, height=0.57, color=GREEN, edgecolor="white", lw=0.6, zorder=2)
        ax.text(-regressed / 2, y, f"{regressed:,}", ha="center", va="center", fontsize=7.5, color=INK)
        ax.text(rescued / 2, y, f"{rescued:,}", ha="center", va="center", fontsize=7.5, color=INK)
        ax.text(1.24, y, f'{row["net_success_change"]:+,}', transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5,
                color=GREEN if row["model"] == "CARE-OPD" else INK,
                weight="bold" if row["model"] == "CARE-OPD" else "normal")
    ax.set(xlim=(-2400, 1350), ylim=(-0.58, 4.75), yticks=ys, yticklabels=MODELS,
           xticks=[-2000, -1000, 0, 1000], xlabel="Absolute request count")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):,.0f}"))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=2.5, width=0.6, pad=2)
    for label in ax.get_yticklabels():
        if label.get_text() == "CARE-OPD":
            label.set_color(GREEN)
            label.set_weight("bold")
    ax.text(1.24, 4.68, "Net", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5)
    fig.legend(handles=[Patch(facecolor=ORANGE, label="Regressed (n = 2,756)"),
                        Patch(facecolor=GREEN, label="Rescued (n = 1,144)")],
               loc="upper center", bbox_to_anchor=(0.515, 0.985), ncol=2,
               frameon=False, handlelength=1.0, handletextpad=0.45,
               columnspacing=1.0, borderaxespad=0.1)
    fig.text(0.52, 0.06, "Fixed Clean-SFT reference; counts, not rates", ha="center", va="center", fontsize=7.5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            clipped.append(text.get_text())
    assert not clipped, f"Clipped text: {clipped}"
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"fig_fixed_transitions.{ext}", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    payload = {
        "figure": "fig_fixed_transitions", "dimensions_inches": [3.49, 2.6],
        "font": "Times New Roman", "body_points": 8.5, "minimum_points": 7.5,
        "source_files": [{"path": SOURCE_REL, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "row_filter": "record_type=transition; state_or_slice=vs_Clean-SFT_all_four_states; exclude Clean-SFT identity row",
            "fields": ["model", "n", "reference_failures", "rescued", "reference_successes", "regressed", "net_success_change"]}],
        "data": data,
        "visual_contract": {"evidence_layer": "fixed-reference descriptive behavior",
            "chart": "diverging absolute-count horizontal bars and net-count text column",
            "left": "regressed nonnegative counts, displayed leftward", "right": "rescued nonnegative counts, displayed rightward",
            "conditioning_denominators": {"rescued": 1144, "regressed": 2756},
            "claim_limits": ["Different conditioning denominators", "No pooled conditional rate", "No new statistical inference"]},
        "caption": CAPTION,
        "qa": {"source_consistency_checks": "passed", "text_clipping_check": "passed",
            "vector_pdf_fonttype": 42, "editable_svg_text": True, "png_dpi": 300}}
    (OUT / "fig_fixed_transitions.data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved fig_fixed_transitions.pdf/png/svg/data.json; source and text-bound checks passed.")


if __name__ == "__main__":
    main()
