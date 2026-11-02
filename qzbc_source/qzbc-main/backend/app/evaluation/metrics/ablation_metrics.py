from typing import Any


def metric_delta(baseline: dict[str, Any], ablated: dict[str, Any], metric: str) -> float:
    return round(float(ablated.get(metric, 0.0)) - float(baseline.get(metric, 0.0)), 4)


def score_drop(baseline_score: float, ablated_score: float) -> float:
    return round(max(0.0, baseline_score - ablated_score), 4)
