from app.evaluation.schemas import EvaluationCaseResult


def rate(results: list[EvaluationCaseResult], metric_name: str, default: float = 1.0) -> float:
    rows = [row for row in results if row.metric_name == metric_name]
    return round(sum(row.passed for row in rows) / len(rows), 4) if rows else default


def category_rate(results: list[EvaluationCaseResult], category: str, default: float = 1.0) -> float:
    rows = [row for row in results if row.category == category]
    return round(sum(row.passed for row in rows) / len(rows), 4) if rows else default
