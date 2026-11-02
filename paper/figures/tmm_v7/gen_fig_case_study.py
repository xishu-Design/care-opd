"""Render a compact categorical case comparison from archived response fields.

The cards are English summaries, not new model outputs or successive edits.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = PROJECT / "runs/mechanism_eval/qzbc_care_opd_v1/qualitative_cases/qualitative_cases.json"
CASE_ID = "27ebf6b23cee5bb845871165"
sys.path.insert(0, str(HERE.parent / "tmm_v4"))
from figure_style import apply_style, INK, MUTED, CLEAN, GKD, CARE


def main():
    record = json.loads(SOURCE.read_text(encoding="utf-8"))
    case = next(c for c in record["cases"] if c["id"] == CASE_ID)
    models = case["models"]
    assert case["variant"] == "evidence_conflict"
    assert [models[m]["passed"] for m in ("clean_sft", "gkd_full", "care_opd")] == [False, False, True]
    assert "CONTROLLED_CONFLICT_DO_NOT_CITE" in models["clean_sft"]["response"]["preview"]
    assert "case_forbidden_content" in models["gkd_full"]["failure_types"]
    assert '"source_ids": []' in models["care_opd"]["response"]["preview"]

    apply_style()
    fig = plt.figure(figsize=(7.16, 1.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tracked = []

    def text(x, y, value, *, size=8.2, color=INK, weight="normal", ha="left", family=None, bounds=None):
        kwargs = {} if family is None else {"fontfamily": family}
        item = ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                       ha=ha, va="center", **kwargs)
        tracked.append((item, bounds))
        return item

    ax.add_patch(FancyBboxPatch((.004, .853), .992, .143,
                 boxstyle="round,pad=0,rounding_size=0.012", linewidth=0,
                 facecolor="#F0F4F8"))
    text(.019, .925, "SAME PHYSICIAN REQUEST", size=7.6, weight="bold", color=CARE)
    text(.365, .925, "Conflicting evidence", size=8.1, ha="center")
    text(.977, .925, "All three request manual review", size=8.1, ha="right")

    cards = [
        dict(name="Clean-SFT", x=.004, color=CLEAN, header="#EEF1F5", body="#FFFFFF",
             summary="Evidence restatement", sections="Catalog claims retained",
             sources="Non-citable marker", passed=False),
        dict(name="GKD-Full", x=.340, color=GKD, header="#F2EEF6", body="#FFFFFF",
             summary="Conflict acknowledged", sections="Prohibited catalog claims",
             sources="Catalog IDs retained", passed=False),
        dict(name="CARE-OPD", x=.676, color=CARE, header="#E3F0F8", body="#F8FCFE",
             summary="Conflict acknowledged", sections="No specific medical claims",
             sources="[]", passed=True),
    ]
    width = .320
    for card in cards:
        x = card["x"]
        bounds = (x, .018, x + width, .813)
        ax.add_patch(FancyBboxPatch((x, .018), width, .795,
                     boxstyle="round,pad=0,rounding_size=0.009", linewidth=.65,
                     edgecolor="#C8D3DE", facecolor=card["body"]))
        ax.add_patch(Rectangle((x + .001, .674), width - .002, .129,
                              facecolor=card["header"], edgecolor="none"))
        text(x + .018, .739, card["name"], size=9.0, weight="bold", color=card["color"], bounds=bounds)

        # Explicit words and geometric marks keep verdicts readable without color.
        pass_color = CARE if card["passed"] else "#AD6928"
        icon_x, icon_y = x + .228, .739
        if card["passed"]:
            ax.plot([icon_x-.007, icon_x-.002, icon_x+.008],
                    [icon_y, icon_y-.020, icon_y+.023], color=pass_color,
                    linewidth=1.2, solid_capstyle="round")
        else:
            ax.plot([icon_x-.006, icon_x+.006], [icon_y-.020, icon_y+.020],
                    color=pass_color, linewidth=1.1)
            ax.plot([icon_x-.006, icon_x+.006], [icon_y+.020, icon_y-.020],
                    color=pass_color, linewidth=1.1)
        text(x + .298, .739, "PASS" if card["passed"] else "FAIL", size=7.7,
             weight="bold", color=pass_color, ha="right", bounds=bounds)

        for label, value, y in [
            ("summary", card["summary"], .605),
            ("sections", card["sections"], .415),
            ("source_ids", card["sources"], .225),
        ]:
            text(x + .018, y, label, size=7.1, color=MUTED,
                 family="DejaVu Sans Mono", bounds=bounds)
            text(x + .018, y - .085, value, size=8.2,
                 color=CARE if card["passed"] else INK, bounds=bounds)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for item, bounds in tracked:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, item.get_text()
        if bounds:
            lower = ax.transData.transform(bounds[:2])
            upper = ax.transData.transform(bounds[2:])
            assert box.x0 >= lower[0] and box.x1 <= upper[0] and box.y0 >= lower[1] and box.y1 <= upper[1], item.get_text()
        boxes.append((item.get_text(), box))
    for i, (left_text, left_box) in enumerate(boxes):
        for right_text, right_box in boxes[i+1:]:
            assert not left_box.overlaps(right_box), (left_text, right_text)

    output = HERE / "fig_case_study"
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=300)
    plt.close(fig)
    print(json.dumps({"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      "case_id": CASE_ID, "selection": case["selection_audit"],
                      "output": str(output.with_suffix('.pdf')), "size_inches": [7.16, 1.62],
                      "text_bounds_and_overlaps": "PASS", "scope": "English field summaries; independent generations; illustrative, not prevalence."}))


if __name__ == "__main__":
    main()
