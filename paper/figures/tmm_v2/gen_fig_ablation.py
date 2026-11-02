#!/usr/bin/env python3
"""Render the archived, matched single-round CARE-OPD ablation (no inference).

Visual contract: panel a shows joint clean/perturbed behavior of five displayed
arms; panel b shows four pre-existing paired perturbed-TSR differences. Records
and update counts are matched, effective supervised-token mass is not. The
CCFA grouped-comparison/direct-label recipe is adapted to scatter + forest
grammar, retaining the full 0--100 range. The complete six-arm results, including
the negative fixed-risk effect, are preserved in fig_ablation_full.

Run with a Python environment containing matplotlib and numpy. Only sibling
fig_ablation.{pdf,png,svg,data.json} files are written.
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
GREEN, INK = "#009E73", "#222222"
ARM_COLORS = {"full_selected": GREEN, "without_retention": "#D55E00",
              "matched_full_response": "#0072B2", "outside_only": "#CC79A7",
              "without_trust": "#56B4E9", "without_risk": "#E69F00"}
EFFECT_COLORS = {"region": GREEN, "protect": GREEN,
                 "trust": "#0072B2", "risk": "#D55E00"}
ARM_ORDER = ["full_selected", "without_retention", "matched_full_response",
             "outside_only", "without_trust", "without_risk"]
LABELS = {"full_selected": "Full CARE-OPD", "without_retention": "w/o Protect",
          "matched_full_response": "Full-response KD", "outside_only": "Outside-only",
          "without_trust": "w/o trust", "without_risk": "w/o risk"}
CAPTION = (
    "Matched single-round ablation on QZBC. (a) Clean and perturbed task-success "
    "rates for five displayed arms with the same 6,350 records, record order, initialization, "
    "and 199 optimizer updates (one epoch, training seed 42). Each point is one "
    "observed arm; there is no interpolated frontier. (b) Archived perturbed-TSR "
    "differences and family-paired 95% percentile-bootstrap intervals (975 families, "
    "2,000 replicates). The two region contrasts disable Protect in both arms. "
    "The Protect switch removes both outside-region and clean-control reference "
    "anchoring; it is not an isolated complement-KL intervention. All arms retain "
    "Oracle CE weight 0.25. Effective supervised-token mass is not matched. The "
    "complete six-arm results, including the negative fixed-risk effect, are "
    "preserved in the companion supplementary figure fig_ablation_full. "
    "Intervals reflect evaluation-sample "
    "uncertainty, not variation across training seeds."
)


def read_csv(rel):
    with (ROOT / rel).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def source(rel, fields):
    return {"path": rel, "sha256": hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(),
            "fields": fields}


def style():
    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8.5,
        "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "care-opd-tmm-v2-ablation", "savefig.dpi": 300,
        "savefig.bbox": None, "savefig.facecolor": "white"})


def main():
    csv_rel = "summary_vs_full_selected/summary.csv"
    full_rel = "summary_vs_full_selected/summary.json"
    selected_rel = "summary_vs_without_retention/summary.json"
    plan_rel = "single_round_all6_compat_20260813_222104/experiment_plan.json"
    archive_rel = ("competition_project_document/formal/competition_submission_v9/"
                   "supporting_public_v9/02_完整统计/QZBC_ABLATION_PAIRED_EFFECTS.csv")
    csv_rows = {r["model"]: r for r in read_csv(csv_rel)}
    summaries = {"full": read_json(full_rel), "selected": read_json(selected_rel)}
    plan = read_json(plan_rel)
    budget = plan["budget_contract"]
    assert budget["max_steps_per_arm"] == 199 and budget["seed"] == 42
    assert budget["same_row_ids_and_order"] and budget["epochs_cap"] == 1
    assert len({a["id_order_sha256"] for a in plan["arms"]}) == 1
    sources = [source(csv_rel, ["model", "clean_tsr", "perturbed_tsr"]),
               source(full_rel, ["models.*.unified_metrics", "paired_cluster_bootstrap_comparisons"]),
               source(selected_rel, ["paired_cluster_bootstrap_comparisons"]),
               source(plan_rel, ["arms.rows", "arms.id_order_sha256", "budget_contract"]),
               source(archive_rel, ["contrast", "endpoint", "delta_pp", "ci95_low", "ci95_high"])]
    arms = []
    for name in ARM_ORDER:
        meta_rel = f"single_round_all6_compat_20260813_222104/{name}/final/run_metadata.json"
        meta = read_json(meta_rel)
        assert meta["global_step"] == 199
        assert meta["opd"]["hard_oracle_ce_weight"] == 0.25
        assert meta["training"]["learning_rate"] == 2e-5
        assert meta["training"]["seed"] == 42
        assert next(a for a in plan["arms"] if a["name"] == name)["rows"] == 6350
        metrics = summaries["full"]["models"][name]["unified_metrics"]
        clean = float(csv_rows[name]["clean_tsr"])
        perturbed = float(csv_rows[name]["perturbed_tsr"])
        assert abs(clean - metrics["clean_task_success_rate"]) < 1e-12
        assert abs(perturbed - metrics["perturbed_task_success_rate"]) < 1e-12
        arms.append({"arm": name, "label": LABELS[name], "clean_tsr_percent": clean * 100,
                     "perturbed_tsr_percent": perturbed * 100,
                     "clean_success_count": metrics["clean_unique_successes"],
                     "clean_n": metrics["clean_unique_units"],
                     "perturbed_success_count": metrics["perturbed_success_count"],
                     "perturbed_n": metrics["paired_units"]})
        sources.append(source(meta_rel, ["global_step", "opd.hard_oracle_ce_weight", "training.learning_rate", "training.seed"]))

    specs = [
        ("selected", "matched_full_response_vs_without_retention", "Selected − full response", "selected-minus-full_response", "region"),
        ("selected", "outside_only_vs_without_retention", "Selected − outside-only", "selected-minus-outside", "region"),
        ("full", "without_retention_vs_full_selected", "Full − w/o Protect", "full-minus-without_retention", "protect"),
        ("full", "without_trust_vs_full_selected", "Full − w/o trust", "full-minus-without_trust", "trust"),
        ("full", "without_risk_vs_full_selected", "Full − w/o risk", "full-minus-without_risk", "risk"),
    ]
    archived = {r["contrast"]: r for r in read_csv(archive_rel) if r["endpoint"] == "perturbed_tsr"}
    effects = []
    for which, key, label, check_key, group in specs:
        comp = summaries[which]["paired_cluster_bootstrap_comparisons"][key]
        raw = comp["metrics"]["perturbed_task_success_rate"]
        # The archived comparison is right-minus-left for this visual; negate
        # the estimate and swap/negate bounds. This is not new resampling.
        delta = -100 * raw["delta"]
        low, high = -100 * raw["ci95_upper"], -100 * raw["ci95_lower"]
        assert raw["replicates_used"] == comp["replicates"] == 2000
        check = archived[check_key]
        for actual, field in [(delta, "delta_pp"), (low, "ci95_low"), (high, "ci95_high")]:
            assert abs(actual - float(check[field])) <= 0.00501
        effects.append({"label": label, "group": group, "delta_pp": delta,
                        "ci95_pp": [low, high], "left_arm": comp["reference_arm"],
                        "right_arm": comp["left_arm"], "source_comparison_key": key,
                        "source_summary": full_rel if which == "full" else selected_rel,
                        "transform": "negate estimate; negate and swap CI bounds",
                        "method": comp["method"], "bootstrap_replicates": 2000})

    supplemental_omissions = {
        "arms": [row for row in arms if row["arm"] == "without_risk"],
        "paired_effects": [row for row in effects if row["group"] == "risk"],
        "reason": "Moved to the complete supplementary ablation figure, not discarded or reinterpreted",
        "complete_figure": "fig_ablation_full.pdf",
        "complete_data": "fig_ablation_full.data.json",
    }
    arms = [row for row in arms if row["arm"] != "without_risk"]
    effects = [row for row in effects if row["group"] != "risk"]
    assert len(arms) == 5 and len(effects) == 4

    style()
    fig = plt.figure(figsize=(7.16, 2.65), dpi=300)
    ax = fig.add_axes([0.075, 0.245, 0.325, 0.635])
    forest = fig.add_axes([0.673, 0.245, 0.305, 0.635])
    fig.text(0.075, 0.94, "(a) Matched five-arm outcomes", weight="normal", va="top")
    fig.text(0.443, 0.94, "(b) Perturbed TSR: paired effects", weight="normal", va="top")
    placements = {
        "full_selected": ((64, 84), "center", "D"),
        "without_retention": ((5, 80), "left", "s"),
        "matched_full_response": ((43, 39), "center", "^"),
        "outside_only": ((5, 49), "left", "v"),
        "without_trust": ((81, 48), "center", "o"),
        "without_risk": ((99, 94), "right", "P"),
    }
    for row in arms:
        name = row["arm"]
        xy = (row["clean_tsr_percent"], row["perturbed_tsr_percent"])
        textxy, align, marker = placements[name]
        color = ARM_COLORS[name]
        ax.plot(*xy, marker=marker, ms=5.5 if name == "full_selected" else 4.5,
                color=color, mec="white", mew=0.6, ls="none", zorder=4, clip_on=False)
        ax.annotate(row["label"], xy, xytext=textxy, textcoords="data",
                    ha=align, va="center", fontsize=7.5,
                    color=GREEN if name == "full_selected" else INK,
                    fontweight="bold" if name == "full_selected" else "normal",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55,
                                "shrinkA": 2.5, "shrinkB": 3.0}, zorder=3)
    ax.set(xlim=(0, 100), ylim=(0, 100), xticks=[0, 25, 50, 75, 100],
           yticks=[0, 25, 50, 75, 100], xlabel="Clean TSR (%)", ylabel="Perturbed TSR (%)")
    ax.tick_params(length=2.5, width=0.6, pad=2)

    ys = [3.65, 2.72, 1.45, 0.35]
    forest.axvline(0, color="#A6A6A6", lw=0.65, ls=(0, (3, 2)), zorder=0)
    forest.axhline(2.05, color="#D9D9D9", lw=0.55, xmin=0, xmax=1)
    for y, row in zip(ys, effects):
        d, (lo, hi) = row["delta_pp"], row["ci95_pp"]
        color = EFFECT_COLORS[row["group"]]
        forest.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=color,
                        ms=4.5, elinewidth=1.25, capsize=2.5, capthick=0.8, zorder=3)
        forest.text(-0.03, y, row["label"], transform=forest.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8.0,
                    weight="bold" if row["group"] == "protect" else "normal")
        forest.text(d, y + 0.21, f"{d:+.2f}", ha="center", va="bottom", fontsize=7.5)
    forest.set(xlim=(-4.5, 13.3), ylim=(-0.18, 4.23), xticks=[-4, 0, 4, 8, 12],
               yticks=ys, xlabel="Difference (percentage points)")
    forest.spines["left"].set_visible(True)
    forest.tick_params(axis="y", left=True, labelleft=False, length=2.5, width=0.6, direction="out")
    forest.tick_params(axis="x", length=2.5, width=0.6, pad=2)
    fig.text(0.5, 0.060,
             "One round · 6,350 records · 199 updates · seed 42; region contrasts: Protect off in both arms",
             ha="center", va="center", fontsize=7.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text_obj in fig.findobj(Text):
        if not text_obj.get_visible() or not text_obj.get_text():
            continue
        bbox = text_obj.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width + 1 or bbox.y1 > fig.bbox.height + 1:
            clipped.append(text_obj.get_text())
    assert not clipped, f"Clipped text: {clipped}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig_ablation.{ext}", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    payload = {"figure": "fig_ablation", "dimensions_inches": [7.16, 2.65],
               "font": "Times New Roman", "body_points": 8.5, "tick_points": 7.5,
               "color_encoding": {"arms": ARM_COLORS, "paired_effect_groups": EFFECT_COLORS,
                                  "palette": "Okabe-Ito; arm labels and distinct markers retained"},
               "source_files": sources, "data": {"arms": arms, "paired_effects": effects},
               "supplemental_omissions": supplemental_omissions,
               "matched_budget": {"rows": 6350, "failure_records": 3350, "clean_controls": 3000,
                   "optimizer_updates": 199, "epochs": 1, "training_seed": 42,
                   "same_row_ids_and_order": True, "supervised_token_mass_matched": False,
                   "oracle_ce_weight_all_arms": 0.25},
               "visual_contract": {"evidence_layer": "matched ablation", "panel_a": "five displayed arms, full-range axes",
                   "panel_b": "four archived paired perturbed-TSR CIs", "claim_limits": [
                       "No Pareto-optimality claim or interpolation", "No new inference or resampling",
                       "Single training seed", "Oracle lacks isolated control",
                       "Protect includes clean-control and outside-region reference anchoring"]},
               "caption": CAPTION, "qa": {"text_clipping_check": "passed", "vector_pdf_fonttype": 42,
                   "editable_svg_text": True, "png_dpi": 300, "archived_csv_ci_crosscheck": "passed"}}
    (OUT / "fig_ablation.data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved fig_ablation.pdf/png/svg/data.json; archived-value checks and text bounds passed.")


if __name__ == "__main__":
    main()
