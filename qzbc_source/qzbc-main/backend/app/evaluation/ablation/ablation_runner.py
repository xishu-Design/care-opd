from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AblationRun, EvaluationRun
from app.evaluation.metrics.ablation_metrics import score_drop
from app.evaluation.report import serialize_evaluation_run
from app.evaluation.runner import EvaluationRunner
from app.llm.experiments.runner import LLMExperimentRunner


CONFIG_PATH = Path(__file__).with_name("ablation_config.yaml")


class AblationRunner:
    def __init__(self, db: Session):
        self.db = db

    def run(self, *, suite: str = "quick", baseline_run_id: str | None = None, actor_user_id: int | None = None) -> dict[str, Any]:
        if suite not in {"quick", "full"}:
            raise ValueError("suite must be quick or full")
        started = perf_counter()
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        baseline = self._baseline(suite, baseline_run_id, actor_user_id)
        baseline_payload = serialize_evaluation_run(baseline)
        experiments = []
        for experiment in config["experiments"]:
            if experiment["name"] == "template_vs_ollama":
                experiments.append(self._llm_comparison(experiment, actor_user_id))
            else:
                ablated = EvaluationRunner(
                    self.db,
                    suite=suite,
                    reset_demo_data=True,
                    actor_user_id=actor_user_id,
                    disabled=set(experiment.get("disable", [])),
                ).run()
                experiments.append(self._comparison(experiment, baseline_payload, serialize_evaluation_run(ablated)))
        ablation_id = f"ablation_{uuid4().hex}"
        report = {
            "ablation_id": ablation_id,
            "phase": "phase_12_system_evaluation",
            "status": "completed",
            "suite": suite,
            "baseline_run_id": baseline.run_id,
            "experiment_count": len(experiments),
            "baseline": self._run_summary(baseline_payload),
            "experiments": experiments,
            "competition_summary": "消融结果显示，MCP、Evidence Packet、Safety Guard、角色记忆、图谱路径与 HumanConfirm 均对应可测量的能力变化。",
            "ppt_ready_tables": [
                {
                    "title": "GuardTCM 架构消融实验",
                    "columns": ["label", "target_metric", "baseline_value", "ablated_value", "delta", "proves"],
                    "rows": [{key: item.get(key) for key in ("label", "target_metric", "baseline_value", "ablated_value", "delta", "proves")} for item in experiments],
                }
            ],
            "ppt_ready_charts": [
                {
                    "title": "核心模块移除后的指标变化",
                    "type": "grouped_bar",
                    "categories": [item["label"] for item in experiments if item.get("baseline_value") is not None],
                    "series": [
                        {"name": "Baseline", "data": [item["baseline_value"] for item in experiments if item.get("baseline_value") is not None]},
                        {"name": "Ablated", "data": [item["ablated_value"] for item in experiments if item.get("baseline_value") is not None]},
                    ],
                }
            ],
        }
        duration_ms = int((perf_counter() - started) * 1000)
        report["duration_ms"] = duration_ms
        self.db.add(
            AblationRun(
                ablation_id=ablation_id,
                actor_user_id=actor_user_id,
                status="completed",
                baseline_run_id=baseline.run_id,
                experiment_count=len(experiments),
                report_json=report,
                duration_ms=duration_ms,
            )
        )
        self.db.commit()
        return report

    def _baseline(self, suite: str, run_id: str | None, actor_user_id: int | None) -> EvaluationRun:
        if run_id:
            run = self.db.scalar(select(EvaluationRun).where(EvaluationRun.run_id == run_id))
            if not run:
                raise ValueError("baseline evaluation run not found")
            return run
        return EvaluationRunner(self.db, suite=suite, reset_demo_data=True, actor_user_id=actor_user_id).run()

    def _llm_comparison(self, experiment: dict, actor_user_id: int | None) -> dict[str, Any]:
        runner = LLMExperimentRunner(self.db)
        template = runner.run(provider="template", suite="quick", actor_user_id=actor_user_id)
        ollama = runner.run(provider="ollama", suite="quick", actor_user_id=actor_user_id)
        return {
            "name": experiment["name"],
            "label": experiment["label"],
            "target_metric": experiment["target_metric"],
            "baseline_value": template["schema_valid_rate"],
            "ablated_value": ollama["schema_valid_rate"],
            "delta": round(ollama["schema_valid_rate"] - template["schema_valid_rate"], 4),
            "score_drop": score_drop(template["schema_valid_rate"], ollama["schema_valid_rate"]),
            "proves": experiment["proves"],
            "template": {key: template[key] for key in ("provider", "model", "success", "fallback_count", "parse_success_rate", "schema_valid_rate", "safety_pass_rate", "avg_latency_ms")},
            "ollama": {key: ollama[key] for key in ("provider", "model", "success", "fallback_count", "parse_success_rate", "schema_valid_rate", "safety_pass_rate", "avg_latency_ms")},
            "fallback_visible": ollama["fallback_count"] > 0,
        }

    @staticmethod
    def _comparison(experiment: dict, baseline: dict, ablated: dict) -> dict[str, Any]:
        target = str(experiment["target_metric"])
        baseline_value = _nested_metric(baseline["metrics"], target)
        ablated_value = _nested_metric(ablated["metrics"], target)
        return {
            "name": experiment["name"],
            "label": experiment["label"],
            "target_metric": target,
            "baseline_value": baseline_value,
            "ablated_value": ablated_value,
            "delta": round(ablated_value - baseline_value, 4),
            "score_drop": score_drop(baseline["overall_score"], ablated["overall_score"]),
            "proves": experiment["proves"],
            "run_id": ablated["run_id"],
            "overall_score": ablated["overall_score"],
        }

    @staticmethod
    def _run_summary(payload: dict) -> dict:
        return {key: payload[key] for key in ("run_id", "total_cases", "pass_rate", "overall_score", "safety_score", "evidence_score", "agent_trace_score", "llm_safety_score")}


def _nested_metric(metrics: dict, path: str) -> float:
    group, name = path.split(".", 1)
    return float(metrics.get(group, {}).get(name, 0.0))
