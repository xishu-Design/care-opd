"""Matched ablation: paired bars plus vertical paired-effect intervals.

Adapts the CCFA grouped_bar_chart and direct-annotation grammar to the
existing matched-ablation record. No new statistical estimation.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
SOURCE = OUT / "fig_ablation.data.json"
STEM = "fig_ablation_redraw_v1"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig_agent"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

from figure_style import apply_style, INK, GRID, AXIS

BLUE, TEAL, ORANGE = "#6E96C9", "#45A88D", "#D99447"
CARE = "#1768A1"


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    for source in payload["source_files"]:
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], path.name
    arms = {a["arm"]: a for a in payload["data"]["arms"]}
    order = ["without_retention", "matched_full_response", "outside_only",
             "without_trust", "full_selected"]
    assert set(order) == set(arms)
    clean = np.array([arms[k]["clean_tsr_percent"] for k in order])
    pert = np.array([arms[k]["perturbed_tsr_percent"] for k in order])
    for k in order:
        a = arms[k]
        assert abs(a["clean_tsr_percent"] - 100*a["clean_success_count"]/a["clean_n"]) < 1e-10
        assert abs(a["perturbed_tsr_percent"] - 100*a["perturbed_success_count"]/a["perturbed_n"]) < 1e-10
    effects = payload["data"]["paired_effects"]
    assert len(effects) == 4
    summaries = {}
    for row in effects:
        assert abs(row["delta_pp"] -
                   (arms[row["left_arm"]]["perturbed_tsr_percent"] -
                    arms[row["right_arm"]]["perturbed_tsr_percent"])) < 1e-10
        rel = row["source_summary"]
        if rel not in summaries:
            summaries[rel] = json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))
        c = summaries[rel]["paired_cluster_bootstrap_comparisons"][row["source_comparison_key"]]
        raw = c["metrics"]["perturbed_task_success_rate"]
        assert np.allclose([row["delta_pp"], *row["ci95_pp"]],
                           [-100*raw["delta"], -100*raw["ci95_upper"], -100*raw["ci95_lower"]])
        assert c["replicates"] == row["bootstrap_replicates"] == 2000

    apply_style()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "savefig.bbox": None, "svg.hashsalt": STEM})
    fig = plt.figure(figsize=(7.16, 2.35), dpi=350)
    ax = fig.add_axes([.064, .245, .450, .610])
    eff = fig.add_axes([.603, .245, .375, .610])
    fig.text(.064, .970, "(a) Matched outcomes", ha="left", va="top",
             fontsize=8.7, fontweight="bold")
    fig.text(.603, .970, "(b) Paired perturbed-TSR effects", ha="left", va="top",
             fontsize=8.7, fontweight="bold")
    x = np.arange(5)
    ax.axvspan(3.55, 4.52, color="#EFF5FB", lw=0, zorder=0)
    bars_a = ax.bar(x-.18, clean, width=.33, color=BLUE, edgecolor="white",
                    linewidth=.4, label="Clean", zorder=3)
    bars_b = ax.bar(x+.18, pert, width=.33, color=TEAL, edgecolor="white",
                    linewidth=.4, label="Perturbed", zorder=3)
    number_labels = []
    for bars in [bars_a, bars_b]:
        for bar in bars:
            v = bar.get_height()
            number_labels.append(ax.text(bar.get_x()+bar.get_width()/2, v+1.8, f"{v:.2f}",
                                          ha="center", va="bottom", fontsize=7.1, zorder=5))
    ax.set(xlim=(-.6,4.6), ylim=(0,104), yticks=[0,25,50,75,100], ylabel="TSR (%)")
    ax.set_xticks(x, ["Selected", "Full-response\nKD", "Outside\nonly", "No trust", "CARE-OPD"])
    ax.tick_params(axis="x", length=0, pad=4, labelsize=7.4)
    ax.tick_params(axis="y", length=2.5, pad=2.5, labelsize=7.4)
    ax.yaxis.labelpad = 3
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.get_xticklabels()[-1].set_color(CARE)
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(.012,1.005), ncol=2,
                       borderaxespad=0, handlelength=1.25, handletextpad=.4,
                       columnspacing=1.0, fontsize=7.5)
    ax.plot([-.40,-.40,2.40,2.40], [-.205,-.23,-.23,-.205],
            transform=ax.get_xaxis_transform(), clip_on=False, color="#8697A9", lw=.65)
    ax.text(1, -.253, "Protect off", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.2, color="#526478")

    eff.axhline(0, color=AXIS, lw=.75, ls=(0,(3,2)), zorder=2)
    eff.axvline(1.5, color=GRID, lw=.65, ls=(0,(2,2)), zorder=0)
    effect_colors = [ORANGE, ORANGE, TEAL, BLUE]
    markers = ["o","o","s","D"]
    whisker_segments = []
    for i, (row, color, marker) in enumerate(zip(effects, effect_colors, markers)):
        d = row["delta_pp"]
        lo, hi = row["ci95_pp"]
        eff.vlines(i, 0, d, color=color, alpha=.28, lw=3.3, zorder=2)
        eff.errorbar(i, d, yerr=[[d-lo],[hi-d]], fmt=marker, markersize=5.5,
                     color=color, markeredgecolor="white", markeredgewidth=.55,
                     elinewidth=1.45, capsize=4, capthick=1.05, zorder=4)
        number_labels.append(eff.text(i,hi+.40,f"{d:+.2f}",ha="center",va="bottom",
                                      fontsize=8.0,color=INK))
        whisker_segments.append((i,lo,hi))
    eff.set(xlim=(-.6,3.6), ylim=(-.6,14.2), yticks=[0,4,8,12],
            ylabel="TSR gain (pp)")
    eff.set_xticks(np.arange(4), ["Selected −\nfull-response", "Selected −\noutside-only",
                                 "CARE-OPD −\nno Protect", "CARE-OPD −\nno trust"])
    eff.tick_params(axis="x", length=0, pad=4, labelsize=7.1)
    eff.tick_params(axis="y", length=2.5, pad=2.5, labelsize=7.4)
    eff.yaxis.labelpad=3
    eff.grid(axis="y", color=GRID, lw=.5, zorder=0)
    eff.set_axisbelow(True)
    for a in (ax, eff):
        for side in ("left","bottom"):
            a.spines[side].set_linewidth(.65)
        a.spines["left"].set_visible(True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t,t.get_window_extent(renderer)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text()]
    clipped = [t.get_text() for t,b in texts if b.x0<-.5 or b.y0<-.5
               or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5]
    assert not clipped, clipped
    overlaps = [(t.get_text(),u.get_text()) for i,(t,b) in enumerate(texts)
                for u,c in texts[i+1:] if b.overlaps(c)]
    assert not overlaps, overlaps
    legend_box = legend.get_window_extent(renderer)
    assert not any(t.get_window_extent(renderer).overlaps(legend_box) for t in number_labels)
    assert len(number_labels)==14
    for ext in ("pdf","png","svg"):
        fig.savefig(OUT/f"{STEM}.{ext}",dpi=350)
    plt.close(fig)
    record = {
        "figure":STEM, "source":{"path":SOURCE.relative_to(ROOT).as_posix(),
                                "sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "data":payload["data"],"display_arm_order":order,
        "matched_budget":payload["matched_budget"],
        "supplemental_omissions":payload["supplemental_omissions"],
        "claim_limits":payload["visual_contract"]["claim_limits"],
        "render":{"size_inches":[7.16,2.35],"panel_a":"grouped clean/perturbed bars",
                  "panel_b":"vertical paired-effect intervals","font":"Arial",
                  "clean_color":BLUE,"perturbed_color":TEAL,
                  "rate_axis_limits":[0,104],"paired_effect_axis_limits":[-.6,14.2]},
        "qa":{"all_upstream_hashes_match":True,"rates_match_counts":True,
              "paired_effects_and_cis_match_archives":True,"clipped_text":clipped,
              "text_collisions":overlaps,"numeric_labels":14,"no_new_resampling":True}}
    (OUT/f"{STEM}.source.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",
                                         encoding="utf-8")
    print(json.dumps({"saved":STEM,"qa":record["qa"]},ensure_ascii=True))


if __name__=="__main__":
    main()
