#!/usr/bin/env python3
"""Three archived training-stage endpoints in four evidence-state panels.

The CCFA small-multiple and direct-label recipes are adapted to three discrete
training stages. Straight segments join endpoints, not sampled training steps.
The prior paired decomposition remains in the data manifest for traceability.
Only sibling fig_training_before_after PDF/PNG/SVG/data.json are written.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "care_opd_tmm_v5_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

sys.path.insert(0, str(OUT.parent / "tmm_v4"))
from figure_style import apply_style, INK, GRID, CLEAN, CARE, BLUE, GREEN, ORANGE

PAIRED_REL = "runs/mechanism_eval/qzbc_care_opd_v1/failure_decomposition/failure_decomposition.csv"
DETAIL_REL = "runs/mechanism_eval/qzbc_care_opd_v1/failure_decomposition/failure_decomposition.json"
MARGINAL_REL = ("competition_project_document/formal/competition_submission_v9/"
                "supporting_public_v9/02_完整统计/QZBC_STATE_AND_TRANSITION_FULL.csv")
ANALYSIS_REL = "care_opd/mechanism_eval/analyze_failure_decomposition.py"
STATES = ["clean", "bad_history", "source_outage", "evidence_conflict"]
LABELS = ["Clean", "Bad history", "Source outage", "Evidence conflict"]
STAGES = [("Frozen", "Initial", "o", CLEAN), ("Clean-SFT", "Clean-SFT", "s", CLEAN),
          ("CARE-OPD", "CARE-OPD", "D", CARE)]
TRANSITIONS = [("both_pass", "Retained", BLUE), ("rescued", "Rescued", GREEN),
               ("regressed", "Regressed", ORANGE), ("both_fail", "Still failed", CLEAN)]


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source(relative):
    path = ROOT / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def extract_data():
    archive = read_csv(PAIRED_REL)
    rows = {r["stratum"]: r for r in archive
            if r["dimension"] == "variant" and r["model"] == "care_opd"
            and r["reference"] == "clean_sft"}
    assert set(rows) == set(STATES)
    detail = json.loads((ROOT / DETAIL_REL).read_text(encoding="utf-8-sig"))
    json_rows = {r["stratum"]: r for r in detail["rows"]
                 if r["dimension"] == "variant" and r["model"] == "care_opd"
                 and r["reference"] == "clean_sft"}
    marginals = {(r["model"], r["state_or_slice"]): r for r in read_csv(MARGINAL_REL)
                 if r["record_type"] == "state"}
    fields = ["n", "reference_passed", "reference_failed", "candidate_passed",
              "both_pass", "rescued", "regressed", "both_fail"]
    data = []
    for state, label in zip(STATES, LABELS):
        r = {key: int(rows[state][key]) for key in fields}
        assert all(r[key] == json_rows[state][key] for key in fields)
        assert r["n"] == 975 == sum(r[key] for key, _, _ in TRANSITIONS)
        assert r["reference_passed"] == r["both_pass"] + r["regressed"]
        assert r["reference_failed"] == r["rescued"] + r["both_fail"]
        assert r["candidate_passed"] == r["both_pass"] + r["rescued"]
        assert r["reference_passed"] == int(marginals["Clean-SFT", state]["successes"])
        assert r["candidate_passed"] == int(marginals["CARE-OPD", state]["successes"])
        r.update(state=state, label=label,
                 before_percent=100 * r["reference_passed"] / r["n"],
                 after_percent=100 * r["candidate_passed"] / r["n"],
                 net_success_change=r["rescued"] - r["regressed"])
        r["stage_points"] = []
        for model, stage, _, _ in STAGES:
            marginal = marginals[model, state]
            n, successes = int(marginal["n"]), int(marginal["successes"])
            assert n == 975 and abs(successes / n - float(marginal["success_rate"])) < 1e-12
            r["stage_points"].append({"archived_model": model, "stage": stage,
                                      "n": n, "successes": successes,
                                      "task_success_percent": 100 * successes / n})
        data.append(r)
    totals = {key: sum(row[key] for row in data) for key in fields}
    overall = next(r for r in archive if r["dimension"] == "overall" and r["stratum"] == "all"
                   and r["model"] == "care_opd" and r["reference"] == "clean_sft")
    assert all(totals[key] == int(overall[key]) for key in fields)
    assert totals["n"] == 3900 and totals["rescued"] - totals["regressed"] == 509
    assert [totals[key] for key, _, _ in TRANSITIONS] == [2217, 1048, 539, 96]
    raw_status = []
    for model in ("clean_sft", "care_opd"):
        relative = ("runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal/"
                    f"family_clean_views/{model}.jsonl")
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        raw_status.append({"model": model, "local_path": relative,
                           "local_bytes": path.stat().st_size, "local_sha256": digest,
                           "archived_sha256": detail["inputs"]["rollouts"][model]["sha256"],
                           "matches_archived_hash": digest == detail["inputs"]["rollouts"][model]["sha256"]})
    return data, totals, raw_status


def main():
    data, totals, raw_status = extract_data()
    font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
    apply_style()
    plt.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "care-opd-tmm-v5-before-after"})
    fig = plt.figure(figsize=(7.16, 2.0), dpi=300)
    for panel, row in enumerate(data):
        ax = fig.add_axes([0.075 + 0.235 * panel, 0.255, 0.200, 0.590])
        values = [point["task_success_percent"] for point in row["stage_points"]]
        ax.plot([0, 1], values[:2], color=CLEAN, lw=1.0, zorder=2)
        ax.plot([1, 2], values[1:], color=CARE, lw=1.0, zorder=2)
        for x, (value, (_, _, marker, color)) in enumerate(zip(values, STAGES)):
            ax.plot(x, value, marker=marker, ls="none", ms=4.5, color=color,
                    mec="white", mew=0.55, zorder=4, clip_on=False)
            dy = -7 if value >= 85 else 6
            ax.annotate(f"{value:.2f}", (x, value), xytext=(0, dy),
                        textcoords="offset points", ha="center",
                        va="top" if dy < 0 else "bottom", fontsize=7.5,
                        color=CARE if x == 2 else INK,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15}, zorder=5)
        ax.set(xlim=(-0.25, 2.25), ylim=(0, 100), yticks=[0, 25, 50, 75, 100],
               xticks=[0, 1, 2], xticklabels=[stage for _, stage, _, _ in STAGES])
        ax.set_title(f"({chr(97 + panel)}) {row['label']}", fontsize=8.2, pad=11)
        ax.grid(axis="y", color=GRID, lw=0.45)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=2.5, pad=2, labelsize=7.5)
        for spine in ("right", "top"):
            ax.spines[spine].set_visible(False)
    fig.text(0.014, 0.55, "Task success (%)", rotation=90, ha="center", va="center", fontsize=8.0)
    fig.text(0.529, 0.075, "Training stage", ha="center", va="center", fontsize=8.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        b = artist.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1:
            clipped.append(artist.get_text())
    assert not clipped, clipped
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig_training_before_after.{ext}", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    payload = {
        "figure": "fig_training_before_after", "schema": "archived-three-stage-endpoints-v2",
        "dimensions_inches": [7.16, 2.0], "font": "Arial", "minimum_text_points": 7.5,
        "style_version": "method-v21-aligned-v4",
        "style_source": {"path": "../tmm_v4/figure_style.py", "sha256": hashlib.sha256((OUT.parent / "tmm_v4/figure_style.py").read_bytes()).hexdigest()},
        "sources": [source(p) for p in (PAIRED_REL, DETAIL_REL, MARGINAL_REL, ANALYSIS_REL)],
        "source_filter": {"dimension": "variant", "model": "care_opd", "reference": "clean_sft"},
        "displayed_source_filter": {"record_type": "state", "model": [s[0] for s in STAGES], "state_or_slice": STATES},
        "fields": ["stratum", "n", "reference_passed", "candidate_passed", "both_pass", "rescued", "regressed", "both_fail"],
        "data": data, "totals": totals, "net_success_change": 509,
        "transition_definitions": {"Retained": "Clean-SFT passes and CARE-OPD passes", "Rescued": "Clean-SFT fails and CARE-OPD passes",
                                   "Regressed": "Clean-SFT passes and CARE-OPD fails", "Still failed": "Both models fail"},
        "color_encoding": {"Initial": CLEAN, "Clean-SFT": CLEAN, "CARE-OPD": CARE,
                           "Initial_to_Clean-SFT_segment": CLEAN, "Clean-SFT_to_CARE-OPD_segment": CARE,
                           "stage_markers": {stage: marker for _, stage, marker, _ in STAGES}},
        "archived_paired_counts_displayed": False,
        "caption_note": "Labels denote training stages: Initial is the frozen starting model, followed by Clean-SFT and the CARE-OPD-trained student. Straight segments connect observed stage endpoints, not training steps; each state has 975 requests.",
        "raw_record_status": {"full_local_join_recomputed": False,
                              "limitation": "Local raw JSONL copies are truncated; use complete archived request-paired CSV/JSON, not a claimed fresh full-record recomputation.",
                              "files": raw_status},
        "checks": {"paired_csv_matches_paired_json": True, "state_marginals_match_public_archive": True,
                   "four_state_n_each": 975, "total_n": 3900, "net_equals_1048_minus_539": True,
                   "no_text_clipped": True, "new_inference_or_resampling": False,
                   "pdf_fonttype": 42, "editable_svg_text": True, "png_dpi": 300},
        "claim_limits": ["Endpoint comparison, not a training-step trajectory or component-level causal attribution",
                         "Derived controlled contract evaluation, not independent clinical efficacy",
                         "All four states retained, including clean, history, and conflict net regressions",
                         "No new uncertainty intervals, smoothing, or unobserved training-step interpolation",
                         "Archived Clean-SFT-to-CARE paired counts retained in manifest but not displayed"],
    }
    (OUT / "fig_training_before_after.data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved fig_training_before_after PDF/PNG/SVG/data.json; archived counts, marginals, totals, and text bounds checked.")


if __name__ == "__main__":
    main()
