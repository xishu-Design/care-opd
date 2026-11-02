"""Plot archived full-response rescue by reference-defined failure type.

Visual contract
---------------
Artifact: fig_failure_rescue; single-column 3.49 x 2.35 inches.
Question: which kinds of Clean-SFT failures are rescued by each trained method?
Evidence: a descriptive stratum analysis of the existing 3,900-request paired
evaluation, not an independent experiment or an error-tag disappearance rate.
Grammar: all five archived failure types x all four recovery-trained methods,
counts beside row labels, exact percentages, shared untruncated 0--100 scale.
Statistics: archived integer rescue counts divided by reference failures; no
new inference, resampling, confidence interval, or significance claim.
Traceability: CSV checked against its companion JSON; hashes and raw-copy
limitations are preserved in the adjacent manifest. Reference labels overlap.
The CCFA heatmap recipe is adapted to Matplotlib and the existing V4 theme.
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
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.text import Text
import numpy as np

sys.path.insert(0, str(OUT.parent / "tmm_v4"))
from figure_style import apply_style, INK, GREEN, CARE

CSV_REL = "runs/mechanism_eval/qzbc_care_opd_v1/failure_decomposition/failure_decomposition.csv"
JSON_REL = "runs/mechanism_eval/qzbc_care_opd_v1/failure_decomposition/failure_decomposition.json"
ANALYZER_REL = "care_opd/mechanism_eval/analyze_failure_decomposition.py"
TYPES = [
    ("failed_evidence_boundary_recovery", "Boundary"),
    ("invalid_generated_response", "Invalid output"),
    ("invented_source_id", "Invented ID"),
    ("missing_required_content", "Missing text"),
    ("missing_required_source_id", "Missing ID"),
]
METHODS = [
    ("recovery_sft", "Recovery\nSFT"),
    ("recovery_dpo", "Recovery\nDPO"),
    ("gkd_full", "GKD\nFull"),
    ("care_opd", "CARE\nOPD"),
]
PALETTE = ["#EEF7F4", "#8ECDBC", GREEN, "#3C99B7", CARE, "#174365"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(relative):
    path = ROOT / relative
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def load_data():
    with (ROOT / CSV_REL).open(encoding="utf-8-sig", newline="") as handle:
        archive = list(csv.DictReader(handle))
    detail = json.loads((ROOT / JSON_REL).read_text(encoding="utf-8-sig"))
    filtered = [r for r in archive if r["dimension"] == "failure_type"
                and r["reference"] == "clean_sft"
                and r["stratum"] != "__reference_pass__"]
    assert {r["stratum"] for r in filtered} == {name for name, _ in TYPES}
    keyed = {(r["stratum"], r["model"]): r for r in filtered}
    json_keyed = {(r["stratum"], r["model"]): r for r in detail["rows"]
                  if r["dimension"] == "failure_type"
                  and r["reference"] == "clean_sft"}
    rows = []
    for failure_type, label in TYPES:
        cells = []
        for method, display in METHODS:
            row = keyed[failure_type, method]
            stored = json_keyed[failure_type, method]
            for key in ("n", "reference_failed", "reference_passed", "rescued",
                        "candidate_passed", "both_fail", "both_pass", "regressed"):
                assert int(row[key]) == stored[key], (failure_type, method, key)
            n, rescued = int(row["reference_failed"]), int(row["rescued"])
            assert int(row["n"]) == n and int(row["reference_passed"]) == 0
            assert int(row["candidate_passed"]) == rescued
            assert rescued + int(row["both_fail"]) == n
            assert abs(rescued / n - float(row["rescue_rate"])) < 1e-12
            cells.append({"method": method, "display": display.replace("\n", "-"),
                          "n": n, "rescued": rescued, "percent": 100 * rescued / n,
                          "csv_line": archive.index(row) + 2})
        assert len({cell["n"] for cell in cells}) == 1
        rows.append({"failure_type": failure_type, "label": label,
                     "n": cells[0]["n"], "cells": cells})
    assert [row["n"] for row in rows] == [39, 48, 1028, 49, 3]
    overall = next(r for r in archive if r["dimension"] == "overall"
                   and r["stratum"] == "all" and r["model"] == "care_opd")
    assert int(overall["n"]) == 3900 and int(overall["reference_failed"]) == 1144
    raw = []
    for model in ["clean_sft"] + [m for m, _ in METHODS]:
        relative = ("runs/public_eval/qzbc_grouped_identity_4gpu_v1/formal/"
                    f"family_clean_views/{model}.jsonl")
        path = ROOT / relative
        record = {"model": model, "path": relative, "exists": path.exists(),
                  "archived_rows": detail["inputs"]["rollouts"][model]["rows"],
                  "archived_sha256": detail["inputs"]["rollouts"][model]["sha256"]}
        if path.exists():
            record.update(local_bytes=path.stat().st_size, local_sha256=sha(path))
            record["matches_archived_hash"] = record["local_sha256"] == record["archived_sha256"]
        raw.append(record)
    return rows, raw


def luminance(color):
    channels = [(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
                for v in to_rgb(color)]
    return sum(v * w for v, w in zip(channels, (0.2126, 0.7152, 0.0722)))


def main():
    data, raw_status = load_data()
    font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
    apply_style()
    plt.rcParams.update({"savefig.bbox": None, "svg.hashsalt": "care-opd-failure-rescue-v5"})
    fig = plt.figure(figsize=(3.49, 2.35), dpi=300)
    ax = fig.add_axes([0.285, 0.265, 0.688, 0.565])
    matrix = np.array([[cell["percent"] for cell in row["cells"]] for row in data])
    cmap = LinearSegmentedColormap.from_list("care_rescue_blue_green", PALETTE, N=512)
    img = ax.pcolormesh(np.arange(5) - 0.5, np.arange(6) - 0.5, matrix,
                        cmap=cmap, vmin=0, vmax=100, shading="flat",
                        edgecolors="white", linewidth=0.75, rasterized=False)
    ax.set(xlim=(-0.5, 3.5), ylim=(4.5, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="both", bottom=False, left=False)
    annotations = []
    contrasts = []
    for i, row in enumerate(data):
        yfig = 0.265 + 0.565 * (1 - (i + 0.5) / len(data))
        annotations.append(fig.text(0.012, yfig, row["label"], fontsize=7.5,
                                    ha="left", va="center", color=INK))
        annotations.append(fig.text(0.264, yfig, f"{row['n']:,}", fontsize=7.5,
                                    ha="right", va="center", color=INK))
        for j, cell in enumerate(row["cells"]):
            background = cmap(cell["percent"] / 100)
            lum = luminance(background)
            white_contrast = 1.05 / (lum + 0.05)
            ink_contrast = (lum + 0.05) / (luminance(INK) + 0.05)
            color = "white" if white_contrast >= ink_contrast else INK
            contrasts.append(max(white_contrast, ink_contrast))
            annotations.append(ax.text(j, i, f"{cell['percent']:.2f}",
                                       fontsize=7.5, ha="center", va="center", color=color))
    annotations.append(fig.text(0.012, 0.913, "Failure type", fontsize=7.5,
                                ha="left", va="center", color=INK))
    annotations.append(fig.text(0.264, 0.913, "n", fontsize=7.5,
                                ha="right", va="center", color=INK))
    for j, (_, label) in enumerate(METHODS):
        annotations.append(fig.text(0.285 + 0.688 * (j + 0.5) / 4, 0.913,
                                    label, fontsize=7.5, ha="center", va="center",
                                    linespacing=1.12, color=INK))
    cax = fig.add_axes([0.285, 0.162, 0.688, 0.035])
    colorbar = fig.colorbar(img, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    colorbar.solids.set_rasterized(False)
    colorbar.outline.set_visible(False)
    cax.tick_params(labelsize=7.5, length=2, pad=1.5)
    colorbar.set_label("Rescued requests (%)", fontsize=8.0, labelpad=2.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    text_boxes = []
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                clipped.append(artist.get_text())
            text_boxes.append((artist.get_text(), box))
    assert not clipped, clipped
    overlaps = [(a[0], b[0]) for i, a in enumerate(text_boxes) for b in text_boxes[i + 1:]
                if a[1].overlaps(b[1]) and min(a[1].x1, b[1].x1) - max(a[1].x0, b[1].x0) > 1
                and min(a[1].y1, b[1].y1) - max(a[1].y0, b[1].y0) > 1]
    assert not overlaps, overlaps
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig_failure_rescue.{ext}", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    payload = {
        "figure": "fig_failure_rescue", "schema": "archived-failure-type-rescue-v1",
        "dimensions_inches": [3.49, 2.35], "font": "Arial", "minimum_text_points": 7.5,
        "style_version": "method-v21-aligned-v4",
        "style_source": {"path": "../tmm_v4/figure_style.py",
                         "sha256": sha(OUT.parent / "tmm_v4/figure_style.py")},
        "sources": [source(p) for p in (CSV_REL, JSON_REL, ANALYZER_REL)],
        "source_filter": {"dimension": "failure_type", "reference": "clean_sft",
                          "excluded": ["__reference_pass__"], "methods": [m for m, _ in METHODS]},
        "data": data,
        "display_label_definitions": {
            "Boundary": "failed_evidence_boundary_recovery: failed evidence-boundary recovery",
            "Invalid output": "invalid_generated_response: invalid GeneratedResponse",
            "Invented ID": "invented_source_id: invented source identifier",
            "Missing text": "missing_required_content: missing required content",
            "Missing ID": "missing_required_source_id: missing required source identifier",
        },
        "metric": "100 * rescued / reference_failed; candidate must pass the full-response verifier",
        "display": {"type": "annotated heatmap", "colormap": PALETTE,
                    "label_region_fraction": 0.285, "matrix_width_fraction": 0.688,
                    "color_scale": [0, 100], "precision_decimals": 2,
                    "minimum_annotation_contrast_ratio": min(contrasts),
                    "uncertainty": "No archived stratum intervals are available; none are fabricated."},
        "population": {"same_evaluation_requests": 3900, "reference_failed_requests": 1144,
                       "sum_of_failure_label_counts": sum(r["n"] for r in data),
                       "failure_types_overlap": True, "smallest_stratum_n": 3},
        "raw_record_status": {"full_local_join_recomputed": False,
                              "limitation": "Local raw JSONL copies are truncated and do not match the archived full-record hashes. Values use the archived paired CSV and companion JSON, not a new raw-record recomputation.",
                              "files": raw_status},
        "caption_note": "Full-response rescue among Clean-SFT failures, grouped by its five original failure labels. Labels overlap; n gives each stratum's denominator. The n=3 source-omission stratum is descriptive only. This is a stratum analysis of the same evaluation, not an independent experiment.",
        "checks": {"csv_integer_counts_match_json": True, "percentages_computed_from_counts": True,
                   "all_archived_failure_types_displayed": True, "no_text_clipped": True,
                   "no_text_overlap": True, "pdf_fonttype": 42, "editable_svg_text": True,
                   "png_dpi": 300, "heatmap_and_colorbar_are_vector": True,
                   "no_new_inference_or_resampling": True},
        "claim_limits": ["Full-response verifier success, not disappearance of a particular error label",
                         "Reference-defined overlapping strata; counts must not be summed into unique requests",
                         "Same held-out evaluation, not new external evidence or a causal component intervention",
                         "No reliable ranking from the n=3 stratum",
                         "Existing-success regressions are not shown here; the main paired-outcome figure reports them"],
    }
    (OUT / "fig_failure_rescue.data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved fig_failure_rescue PDF/PNG/SVG/data.json; 20 cells and text bounds checked.")


if __name__ == "__main__":
    main()
