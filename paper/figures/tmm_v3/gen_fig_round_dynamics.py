#!/usr/bin/env python3
"""Plot archived on-policy failure composition, without causal extrapolation.

Visual contract: a single-column stacked-bar view shows semantic/format
failures at each of three collections. New failures relative to the previous
collection appear as text, not a second axis. This adapts the CCFA grouped-bar
recipe to a composition comparison; no fit, smoothing or new statistics.
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
from matplotlib.text import Text

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
CSV_REL = "runs/mechanism_eval/qzbc_care_opd_v1/round_dynamics/round_dynamics.csv"
JSON_REL = "runs/mechanism_eval/qzbc_care_opd_v1/round_dynamics/round_dynamics.json"
SEMANTIC_COLOR, FORMAT_COLOR = "#0072B2", "#D55E00"
CAPTION = (
    "Failure composition in the three on-policy collections, each containing "
    "12,000 stochastic rollouts. Semantic and format failures partition the "
    "3,350, 2,996, and 2,731 failures. Format failures contain an invalid generated "
    "response; semantic failures are the remaining failed responses. The labels "
    "below rounds 2 and 3 count failures absent from the immediately preceding "
    "collection (2,737 and 1,490). These are descriptive collection diagnostics: "
    "they neither isolate the benefit of online resampling nor constitute a "
    "matched one-round-versus-three-round evaluation."
)


def main():
    with (ROOT / CSV_REL).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    detail = json.loads((ROOT / JSON_REL).read_text(encoding="utf-8-sig"))
    rounds = []
    for row in rows:
        if row["row_type"] != "round":
            continue
        r = {key: int(row[key]) for key in ["round", "failed", "format_failures", "semantic_failures", "rollouts", "passed"]}
        assert r["semantic_failures"] + r["format_failures"] == r["failed"]
        assert r["rollouts"] == r["failed"] + r["passed"] == 12000
        original = next(q for q in detail["rounds"] if int(q["label"]) == r["round"])["recomputed"]
        assert all(r[key] == original[key] for key in r if key != "round")
        transition = next((q for q in rows if q["row_type"] == "transition" and q["failure_class"] == "all"
                           and int(q["to"]) == r["round"]), None)
        r["new_failures_vs_previous"] = int(transition["new"]) if transition else None
        r["persistent_failures_vs_previous"] = int(transition["persistent"]) if transition else None
        if transition:
            assert r["new_failures_vs_previous"] + r["persistent_failures_vs_previous"] == r["failed"]
        rounds.append(r)
    assert [r["round"] for r in rounds] == [1, 2, 3]
    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8.5,
        "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#222222",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "care-opd-tmm-v3-round-dynamics", "savefig.dpi": 300,
        "savefig.bbox": None, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(3.49, 1.9), dpi=300)
    ax = fig.add_axes([0.19, 0.22, 0.775, 0.64])
    xs = [1, 2, 3]
    semantic = [r["semantic_failures"] for r in rounds]
    formats = [r["format_failures"] for r in rounds]
    ax.bar(xs, semantic, width=0.57, color=SEMANTIC_COLOR, edgecolor="white", lw=0.6, label="Semantic")
    ax.bar(xs, formats, width=0.57, bottom=semantic, color=FORMAT_COLOR, edgecolor="white", lw=0.6,
           hatch="////", label="Format")
    for x, row in zip(xs, rounds):
        ax.text(x, row["semantic_failures"] * 0.50, f'{row["semantic_failures"]:,}',
                color="white", ha="center", va="center", fontsize=8.5)
        ax.text(x, row["failed"] + 140, f'{row["failed"]:,}', ha="center", va="bottom", fontsize=8.5)
        cap_middle = row["semantic_failures"] + row["format_failures"] * 0.5
        ax.annotate(str(row["format_failures"]), (x + 0.29, cap_middle),
                    xytext=(x + 0.38, cap_middle), va="center", ha="left", fontsize=7.5,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": FORMAT_COLOR, "shrinkA": 1, "shrinkB": 0})
    ax.set(xlim=(0.5, 3.63), ylim=(0, 4050), ylabel="Failed rollouts (count)",
           xticks=xs, xticklabels=["Round 1", "Round 2", "Round 3"], yticks=[0, 1000, 2000, 3000, 4000])
    ax.tick_params(length=2.5, width=0.6, pad=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.46, 1.17), ncol=2,
              frameon=False, handlelength=1.2, columnspacing=1.3, borderaxespad=0)
    for x, row in zip(xs, rounds):
        if row["new_failures_vs_previous"] is not None:
            ax.text(x, -0.19, f'New: {row["new_failures_vs_previous"]:,}',
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        b = obj.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1:
            clipped.append(obj.get_text())
    assert not clipped, f"Clipped text: {clipped}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig_round_dynamics.{ext}", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    payload = {"figure": "fig_round_dynamics", "dimensions_inches": [3.49, 1.9],
        "font": "Times New Roman", "body_points": 8.5, "tick_points": 7.5,
        "color_encoding": {"semantic": SEMANTIC_COLOR, "format": FORMAT_COLOR,
                           "format_hatch": "////", "meaning": "failure classes, not model identities"},
        "source_files": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest(),
                          "fields": (["row_type", "round", "failed", "semantic_failures", "format_failures", "new", "persistent", "rollouts"]
                                     if p == CSV_REL else ["rounds.*.recomputed", "definitions"])} for p in [CSV_REL, JSON_REL]],
        "data": rounds, "definitions": detail["definitions"], "caption": CAPTION,
        "visual_contract": {"evidence_layer": "descriptive collection diagnostic", "chart": "three stacked bars plus new-failure count annotations",
            "claim_limits": ["No causal identification of online resampling", "Not a matched one-round vs three-round evaluation",
                             "No second axis, smoothing, interpolation, or invented uncertainty"]},
        "qa": {"text_clipping_check": "passed", "source_json_crosscheck": "passed", "vector_pdf_fonttype": 42,
               "editable_svg_text": True, "png_dpi": 300}}
    (OUT / "fig_round_dynamics.data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved fig_round_dynamics.pdf/png/svg/data.json; source checks and text bounds passed.")


if __name__ == "__main__":
    main()


