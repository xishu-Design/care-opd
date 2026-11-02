"""Run clean/critical/nuisance paired evaluation in the synthetic environment."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .baselines import (
    RecoveryExemplar,
    RecoveryExemplarIndex,
    action_after_reflection_context,
    paladin_retrieval_context,
    reflexion_context,
)
from .environment import CareRecoveryEnv, KIND_TO_SOURCE
from .rollout import GenerationArguments, HFCausalPolicy, TextPolicy, render_policy_prompt
from .schemas import (
    Action,
    ActionKind,
    Decision,
    FailureType,
    case_from_mapping,
    canonical_json,
    jsonl_load,
    to_jsonable,
)
from .verifier import CausalRepairVerifier, RISK_SEVERITY


@dataclass(frozen=True)
class PrefixPlan:
    role: str
    action: Optional[Action]
    failure_type: str = "none"
    risk_weight: float = 1.0
    source_id: str = ""


def _nuisance_action(env: CareRecoveryEnv) -> Action:
    hidden = env.state.case.hidden
    for kind, source in KIND_TO_SOURCE.items():
        if source.value in hidden.unavailable_sources:
            continue
        if source.value in hidden.prerequisite_by_source:
            continue
        return Action(
            kind=kind,
            query=f"等预算空检索 ZZZ-NO-MATCH {env.state.case.public.patient_id}",
            rationale="评测用等预算 nuisance 前缀；不包含任务证据。",
        )
    return Action(
        kind=ActionKind.SEARCH_LIGHTRAG,
        query="等预算空检索 ZZZ-NO-MATCH",
        rationale="评测用等预算 nuisance 前缀；不包含任务证据。",
    )


def build_prefix_plans(
    case: Any,
    recovery_rows: Sequence[Mapping[str, Any]],
    prefix_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, PrefixPlan]:
    preregistered = next(
        (row for row in prefix_rows if row.get("case_id") == case.case_id), None
    )
    if preregistered is not None:
        critical_payload = preregistered["critical"]
        nuisance_payload = preregistered["nuisance"]
        return {
            "clean": PrefixPlan("clean", None),
            "critical": PrefixPlan(
                "critical",
                Action.from_mapping(critical_payload["prefix_action"]),
                str(critical_payload.get("failure_type") or FailureType.OTHER.value),
                float(RISK_SEVERITY[FailureType.PREMATURE_DECISION]),
                str(preregistered.get("plan_id", "")),
            ),
            "nuisance": PrefixPlan(
                "nuisance",
                Action.from_mapping(nuisance_payload["prefix_action"]),
                str(nuisance_payload.get("failure_type") or "nuisance_no_match"),
                float(RISK_SEVERITY[FailureType.PREMATURE_DECISION]),
                str(preregistered.get("plan_id", "")),
            ),
        }
    candidates = [row for row in recovery_rows if row.get("case_id") == case.case_id]
    candidates.sort(
        key=lambda row: (
            float(row.get("risk_weight", 0.0)),
            str(row.get("recovery_id", "")),
        ),
        reverse=True,
    )
    if candidates:
        row = candidates[0]
        critical = Action.from_mapping(row["rejected_action"])
        failure_type = str(
            row.get("error_type")
            or row.get("repair_certificate", {}).get("failure_type")
            or row.get("injection")
            or FailureType.OTHER.value
        )
        risk_weight = float(row.get("risk_weight", RISK_SEVERITY[FailureType.OTHER]))
        source_id = str(row.get("recovery_id", ""))
    else:
        env = CareRecoveryEnv(case)
        oracle = env.oracle_action()
        if oracle.kind.is_tool:
            critical = Action(
                kind=oracle.kind,
                query="复诊",
                rationale="评测用关键错误：查询遗漏实体和时间条件。",
            )
            failure_type = FailureType.QUERY_UNDERSPECIFIED.value
            risk_weight = RISK_SEVERITY[FailureType.QUERY_UNDERSPECIFIED]
        else:
            critical = Action(
                kind=ActionKind.DECIDE,
                evidence_ids=(),
                decision=Decision.CLEAR,
                rationale="评测用关键错误：遗漏证据。",
            )
            failure_type = FailureType.PREMATURE_DECISION.value
            risk_weight = RISK_SEVERITY[FailureType.PREMATURE_DECISION]
        source_id = "deterministic_fallback"
    nuisance = _nuisance_action(CareRecoveryEnv(case))
    return {
        "clean": PrefixPlan("clean", None),
        "critical": PrefixPlan(
            "critical", critical, failure_type, float(risk_weight), source_id
        ),
        "nuisance": PrefixPlan(
            "nuisance", nuisance, "nuisance_no_match", float(risk_weight), source_id
        ),
    }


def _exemplar_index(recovery_rows: Sequence[Mapping[str, Any]]) -> RecoveryExemplarIndex:
    exemplars: list[RecoveryExemplar] = []
    seen: set[str] = set()
    for row in recovery_rows:
        failure = str(row.get("injection") or row.get("error_type") or "unknown")
        if failure in seen:
            continue
        repair = row.get("oracle_action") or row.get("repair_certificate", {}).get(
            "intervened_action"
        )
        if not isinstance(repair, Mapping):
            continue
        seen.add(failure)
        exemplars.append(
            RecoveryExemplar(
                failure_signature=failure.replace("_", " "),
                diagnosis=failure,
                recovery_action=dict(repair),
                failure_type=failure,
            )
        )
    return RecoveryExemplarIndex(exemplars)


def _messages_as_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _generate(
    policy: TextPolicy,
    prompt: str,
    *,
    seed: int,
    policy_mode: str,
    previous_failure: Optional[str],
    trajectory: Sequence[Mapping[str, Any]],
    task: str,
    remaining_budget: int,
    exemplar_index: RecoveryExemplarIndex,
) -> tuple[str, int]:
    extra_calls = 0
    if policy_mode == "reflexion" and previous_failure:
        reflection_prompt = _messages_as_prompt(
            reflexion_context(
                task=task,
                trajectory=trajectory,
                failure_observation=previous_failure,
                remaining_budget=remaining_budget,
            )
        )
        reflection = policy.generate(reflection_prompt, seed=seed + 97)
        extra_calls += 1
        prompt = _messages_as_prompt(
            action_after_reflection_context(
                public_messages=[{"role": "user", "content": prompt}],
                reflection=reflection,
                remaining_budget=remaining_budget,
            )
        )
    elif policy_mode == "paladin_retrieval" and previous_failure:
        prompt = _messages_as_prompt(
            paladin_retrieval_context(
                public_messages=[{"role": "user", "content": prompt}],
                failure_observation=previous_failure,
                index=exemplar_index,
                k=1,
            )
        )
    return policy.generate(prompt, seed=seed), extra_calls


def evaluate_branch(
    case: Any,
    plan: PrefixPlan,
    policy: TextPolicy,
    *,
    method: str,
    split: str,
    seed: int,
    max_total_steps: int,
    policy_mode: str,
    exemplar_index: RecoveryExemplarIndex,
) -> dict[str, Any]:
    env = CareRecoveryEnv(case)
    trace: list[dict[str, Any]] = []
    total_steps = 0
    model_calls = 0
    parse_errors = 0
    unsafe_green = False
    safe_false_red = False
    first_action_aligned: Optional[bool] = None
    branch_return_step: Optional[int] = None
    previous_failure: Optional[str] = None

    if plan.action is not None:
        before = env.policy_state()
        _, reward, done, info = env.step(plan.action)
        total_steps += 1
        trace.append(
            {
                "origin": "controlled_prefix",
                "state": before,
                "action": to_jsonable(plan.action),
                "reward": reward,
                "done": done,
                "info": to_jsonable(info),
            }
        )
        previous_failure = str(info.get("failure_type") or plan.failure_type)

    student_action_index = 0
    while not env.state.done and total_steps < max_total_steps:
        state = env.policy_state()
        prompt = render_policy_prompt(state)
        raw, extra_calls = _generate(
            policy,
            prompt,
            seed=seed + model_calls * 1009,
            policy_mode=policy_mode,
            previous_failure=previous_failure,
            trajectory=trace,
            task=case.public.task,
            remaining_budget=max_total_steps - total_steps,
            exemplar_index=exemplar_index,
        )
        model_calls += 1 + extra_calls
        total_steps += 1
        try:
            action = Action.from_json(raw)
        except ValueError as exc:
            parse_errors += 1
            previous_failure = f"parse_error: {exc}"
            trace.append(
                {
                    "origin": "student",
                    "state": state,
                    "raw_action": raw,
                    "parse_error": str(exc),
                }
            )
            continue

        verifier = CausalRepairVerifier(env)
        assessment = verifier.assess(action)
        if first_action_aligned is None:
            first_action_aligned = bool(assessment.resolves_subgoal)
        if assessment.resolves_subgoal and branch_return_step is None:
            branch_return_step = student_action_index + 1
        student_action_index += 1

        if action.kind == ActionKind.DECIDE:
            if case.hidden.true_decision != Decision.CLEAR and action.decision == Decision.CLEAR:
                unsafe_green = True
            if case.hidden.true_decision == Decision.CLEAR and action.decision in {
                Decision.BLOCK,
                Decision.REVIEW,
            }:
                safe_false_red = True
        if action.kind == ActionKind.HUMAN_REVIEW and case.hidden.true_decision == Decision.CLEAR:
            safe_false_red = True

        _, reward, done, info = env.step(action)
        previous_failure = (
            None
            if assessment.resolves_subgoal
            else f"{assessment.failure_type.value}: {assessment.notes[0]}"
        )
        trace.append(
            {
                "origin": "student",
                "state": state,
                "raw_action": raw,
                "action": to_jsonable(action),
                "assessment": to_jsonable(assessment),
                "reward": reward,
                "done": done,
                "info": to_jsonable(info),
            }
        )

    resolved = bool(env.state.resolved)
    return {
        "schema_version": "care-opd-benchmark-v1",
        "method": method,
        "policy_mode": policy_mode,
        "split": split,
        "seed": seed,
        "world_group_id": case.case_id,
        "case_group_id": case.group_id,
        "case_id": case.case_id,
        "pair_role": plan.role,
        "risk_polarity": "safe" if case.hidden.true_decision == Decision.CLEAR else "risky",
        "failure_type": plan.failure_type,
        "prefix_source_id": plan.source_id,
        "risk_weight": plan.risk_weight,
        "resolved": resolved,
        "task_success": resolved,
        "unsafe_green": unsafe_green,
        "safe_false_red": safe_false_red,
        "human_review": bool(env.state.escalated),
        "invalid_json": parse_errors > 0,
        "parse_errors": parse_errors,
        "steps": total_steps,
        "model_calls": model_calls,
        "first_action_aligned": bool(first_action_aligned),
        "branch_return_step": branch_return_step,
        "terminal_decision": (
            None
            if env.state.terminal_decision is None
            else env.state.terminal_decision.value
        ),
        "trace": trace,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--recovery-file", type=Path, required=True)
    parser.add_argument("--prefix-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--tokenizer")
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conditions", default="clean,critical,nuisance")
    parser.add_argument(
        "--policy-mode",
        choices=("direct", "reflexion", "paladin_retrieval"),
        default="direct",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=-1)
    parser.add_argument("--max-total-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = [case_from_mapping(row) for row in jsonl_load(str(args.cases))]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    recovery_rows = jsonl_load(str(args.recovery_file))
    prefix_rows = (
        [] if args.prefix_plan is None else jsonl_load(str(args.prefix_plan))
    )
    index = _exemplar_index(recovery_rows)
    policy = HFCausalPolicy(
        GenerationArguments(
            model=args.model,
            tokenizer=args.tokenizer,
            adapter=args.adapter,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=max(args.temperature, 1e-6),
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )
    )
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    unknown = set(conditions) - {"clean", "critical", "nuisance"}
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        plans = build_prefix_plans(case, recovery_rows, prefix_rows)
        # Common random numbers across clean/critical/nuisance branches reduce
        # decoding-noise variance in the paired comparison.  The prompt/state
        # may differ, but every branch starts from the same RNG seed schedule.
        branch_seed = args.seed + case_index * 100_003
        for condition in conditions:
            row = evaluate_branch(
                case,
                plans[condition],
                policy,
                method=args.method,
                split=args.split,
                seed=branch_seed,
                max_total_steps=args.max_total_steps,
                policy_mode=args.policy_mode,
                exemplar_index=index,
            )
            # A formal report aggregates multiple evaluation seeds.  Keep each
            # seed's matched triplet as a distinct statistical cluster instead
            # of silently overwriting the same case id in paired metrics.
            row["evaluation_seed"] = args.seed
            row["world_group_id"] = (
                f"{case.case_id}::eval_seed={args.seed}"
            )
            row["case_group_id"] = (
                f"{case.group_id}::eval_seed={args.seed}"
            )
            rows.append(row)
    write_jsonl(args.output, rows)
    summary = {
        "output": str(args.output),
        "rows": len(rows),
        "cases": len(cases),
        "conditions": conditions,
        "resolved": sum(bool(row["resolved"]) for row in rows),
        "unsafe_green": sum(bool(row["unsafe_green"]) for row in rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
