from care_opd.metrics import (
    aggregate,
    group_bootstrap,
    paired_permutation_test,
    paired_risk_adjusted_recovery,
)


def _triplet(group: str, critical: bool, nuisance: bool = True, unsafe: bool = False):
    common = {"world_group_id": group, "method": "m", "risk_level": "safety_critical"}
    return [
        {**common, "pair_role": "clean", "resolved": True, "steps": 2},
        {
            **common,
            "pair_role": "critical",
            "resolved": critical,
            "unsafe_green": unsafe,
            "steps": 4,
        },
        {**common, "pair_role": "nuisance", "resolved": nuisance, "steps": 4},
    ]


def test_prar_controls_for_nuisance_budget():
    rows = [
        *_triplet("a", critical=True),
        *_triplet("b", critical=False),
        *_triplet("c", critical=False, nuisance=False),
    ]
    assert paired_risk_adjusted_recovery(rows) == 0.5
    summary = aggregate(rows)
    assert summary["paired_coverage"] == 1.0
    assert summary["clean_success"] == 1.0


def test_unsafe_green_is_penalized():
    rows = _triplet("a", critical=True, unsafe=True)
    assert paired_risk_adjusted_recovery(rows) == -1.0


def test_group_bootstrap_and_paired_test():
    better = [row for group in ("a", "b", "c") for row in _triplet(group, critical=True)]
    worse = [row for group in ("a", "b", "c") for row in _triplet(group, critical=False)]
    for row in worse:
        row["method"] = "baseline"
    estimate = group_bootstrap(better, paired_risk_adjusted_recovery, samples=100, seed=3)
    assert estimate.value == 1.0
    assert estimate.n_groups == 3
    test = paired_permutation_test(better, worse, samples=1_000, seed=3)
    assert test["difference"] == 1.0
    assert test["n_pairs"] == 3


def test_cluster_bootstrap_preserves_repeated_draw_multiplicity():
    rows = [
        {"world_group_id": f"g{index}", "pair_role": "clean"}
        for index in range(3)
    ]
    estimate = group_bootstrap(
        rows,
        lambda sample: float(
            len({str(row["world_group_id"]) for row in sample})
        ),
        samples=50,
        seed=3,
    )
    assert estimate.low == 3.0
    assert estimate.high == 3.0
