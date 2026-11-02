"""Online student rollout collection with verifier-localized OPD exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .schemas import (
    Action,
    CaseSpec,
    FailureType,
    canonical_json,
    case_from_mapping,
    jsonl_load,
    to_jsonable,
)


class TextPolicy(Protocol):
    """Minimal policy interface; fake policies can implement it without torch."""

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        add_special_tokens: bool = True,
    ) -> str:
        ...


@dataclass
class GenerationArguments:
    model: str
    tokenizer: Optional[str] = None
    adapter: Optional[str] = None
    device: Optional[str] = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 384
    temperature: float = 0.9
    top_p: float = 0.95
    do_sample: bool = True
    trust_remote_code: bool = False


@dataclass
class CollectionArguments:
    cases: Path
    output_dir: Path
    max_episodes: int = -1
    max_steps: int = 6
    seed: int = 42
    resume: bool = False
    include_clean_controls: bool = True


@dataclass
class EpisodeResult:
    episode_id: str
    case_id: str
    seed: int
    steps: List[Dict[str, Any]]
    resolved: bool
    escalated: bool
    terminal_decision: Optional[str]
    parse_errors: int
    verifier_rejections: int
    opd_examples: List[Dict[str, Any]] = field(default_factory=list)
    revision_examples: List[Dict[str, Any]] = field(default_factory=list)
    verified_sft_examples: List[Dict[str, Any]] = field(default_factory=list)
    verified_kd_examples: List[Dict[str, Any]] = field(default_factory=list)


POLICY_INSTRUCTION = (
    "你是 CARE-OPD 合成证据恢复策略。仅根据公开状态和工具历史，"
    "输出一个符合 output_contract 的 JSON 动作。不得输出 Markdown，"
    "不得把本仿真用于真实诊疗。"
)


def render_policy_prompt(
    state: Mapping[str, Any],
    *,
    external_feedback: Optional[Mapping[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "instruction": "选择下一步最小、可验证的动作。",
        "policy_state": state,
    }
    if external_feedback:
        payload["last_rejected_output"] = external_feedback
    return POLICY_INSTRUCTION + "\n" + canonical_json(payload)


def render_teacher_prompt(
    student_prompt: str,
    *,
    failure_type: str,
    selected_components: Sequence[str],
    verified_repair: Mapping[str, Any],
    certificate_hash: str,
) -> str:
    privileged = {
        "host_verifier": {
            "failure_type": failure_type,
            "causally_decisive_components": list(selected_components),
            "verified_minimal_repair": verified_repair,
            "certificate_hash": certificate_hash,
        },
        "instruction": (
            "在相同输出前缀上提供正确恢复策略的 token 分布；"
            "只修复已定位组件，保留其他正确组件。"
        ),
    }
    return student_prompt + "\n<privileged_teacher_context>\n" + canonical_json(privileged)


def render_revision_prompt(
    student_prompt: str,
    *,
    rejected_output: str,
    failure_type: str,
    verifier_notes: Sequence[str],
) -> str:
    """Build the public first-error revision context used by SFT baselines.

    Unlike the privileged OPD teacher prompt, this context never contains the
    hidden repair action or hidden evidence.  It exposes only the action that
    was actually rejected and the environment feedback available after that
    transition.
    """

    feedback = {
        "rejected_output": rejected_output,
        "failure_type": failure_type,
        "verifier_notes": list(verifier_notes),
        "instruction": "在相同公开状态上修正首个失败动作；只输出一个 JSON 动作。",
    }
    return student_prompt + "\n<public_transition_feedback>\n" + canonical_json(feedback)


def _sft_messages(prompt: str, target: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]


def _json_value_span(text: str, key: str) -> Optional[Tuple[int, int]]:
    """Locate a top-level JSON value span without normalizing the raw rollout."""

    key_pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*')
    match = key_pattern.search(text)
    if match is None:
        return None
    decoder = json.JSONDecoder()
    try:
        _, consumed = decoder.raw_decode(text[match.end() :])
    except json.JSONDecodeError:
        return None
    return match.end(), match.end() + consumed


def component_char_spans(
    raw_response: str,
    components: Sequence[str],
    *,
    risk_weight: float,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for component in components:
        bounds = _json_value_span(raw_response, component)
        if bounds is None:
            continue
        spans.append(
            {
                "start": bounds[0],
                "end": bounds[1],
                "component": component,
                "risk_weight": risk_weight,
            }
        )
    # A malformed response or a component that cannot be projected safely uses
    # full-response supervision.  This fallback is explicit in the audit.
    if components and not spans:
        spans.append(
            {
                "start": 0,
                "end": max(len(raw_response), 1),
                "component": "__full_response_fallback__",
                "risk_weight": risk_weight,
            }
        )
    return spans


class HFCausalPolicy:
    """Text-only generation wrapper supporting base or PEFT adapter checkpoints."""

    def __init__(self, args: GenerationArguments):
        from .models import ModelArguments, load_causal_lm, load_tokenizer

        model_args = ModelArguments(
            student_model_name_or_path=args.model,
            tokenizer_name_or_path=args.tokenizer,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.dtype,
        )
        self.tokenizer = load_tokenizer(model_args)
        self.model = load_causal_lm(
            args.model,
            model_args,
            trainable=False,
        )
        if args.adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("loading --adapter requires peft") from exc
            self.model = PeftModel.from_pretrained(self.model, args.adapter)
        import torch

        requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        self.model.to(self.device)
        self.model.eval()
        self.args = args

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        add_special_tokens: bool = True,
    ) -> str:
        import torch
        from transformers import set_seed

        set_seed(seed)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.args.do_sample:
            generation_kwargs.update(
                temperature=self.args.temperature,
                top_p=self.args.top_p,
            )
        with torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def collect_case(
    case: CaseSpec,
    policy: TextPolicy,
    *,
    collection_seed: int,
    max_steps: int,
    include_clean_controls: bool,
) -> EpisodeResult:
    """Run one genuine student trajectory and export verified OPD checkpoints."""

    from .environment import CareRecoveryEnv
    from .verifier import CausalRepairVerifier, RISK_SEVERITY

    env = CareRecoveryEnv(case)
    steps: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    revision_examples: List[Dict[str, Any]] = []
    verified_sft_examples: List[Dict[str, Any]] = []
    verified_kd_examples: List[Dict[str, Any]] = []
    external_feedback: Optional[Dict[str, Any]] = None
    parse_errors = 0
    verifier_rejections = 0

    for attempt in range(max_steps):
        state_before = env.policy_state()
        prior_failure_count = sum(
            1
            for entry in state_before.get("history", ())
            if str(
                entry.get("observation", {}).get("failure_type", "none")
            )
            != FailureType.NONE.value
        )
        state_role = (
            "post_failure_recovery"
            if prior_failure_count
            else "pre_failure_or_clean"
        )
        prompt = render_policy_prompt(
            state_before,
            external_feedback=external_feedback,
        )
        step_seed = collection_seed + attempt
        raw = policy.generate(prompt, seed=step_seed).strip()
        verifier = CausalRepairVerifier(env)
        assessment = verifier.assess_raw(raw)
        step_payload: Dict[str, Any] = {
            "attempt": attempt,
            "seed": step_seed,
            "state": state_before,
            "student_prompt": prompt,
            "raw_action": raw,
            "assessment": to_jsonable(assessment),
        }

        try:
            action = Action.from_json(raw)
        except ValueError as exc:
            parse_errors += 1
            risk_weight = float(RISK_SEVERITY[FailureType.PARSE_ERROR])
            oracle = env.oracle_action()
            teacher_prompt = render_teacher_prompt(
                prompt,
                failure_type=FailureType.PARSE_ERROR.value,
                selected_components=("kind", "query", "evidence_ids", "decision", "rationale"),
                verified_repair=to_jsonable(oracle),
                certificate_hash="parse_error_host_fallback",
            )
            examples.append(
                {
                    "example_id": f"{case.case_id}_a{attempt}_parse",
                    "case_id": case.case_id,
                    "group_id": case.group_id,
                    "round_seed": collection_seed,
                    "on_policy_state_role": state_role,
                    "prior_failure_count": prior_failure_count,
                    "student_prompt": prompt,
                    "teacher_prompt": teacher_prompt,
                    "rollout": raw or " ",
                    "repair_spans": [
                        {
                            "start": 0,
                            "end": max(len(raw), 1),
                            "component": "__parse_error_full_response__",
                            "risk_weight": risk_weight,
                        }
                    ],
                    "risk_weight": risk_weight,
                    "error_type": FailureType.PARSE_ERROR.value,
                    "host_repair_verified": True,
                    "host_verification_confidence": 1.0,
                    "teacher_output_verified": False,
                    "teacher_verified": False,
                    "teacher_confidence": 0.0,
                    "clean_control": False,
                    "oracle_action": to_jsonable(oracle),
                }
            )
            if not revision_examples:
                target = canonical_json(to_jsonable(oracle))
                revision_prompt = render_revision_prompt(
                    prompt,
                    rejected_output=raw,
                    failure_type=FailureType.PARSE_ERROR.value,
                    verifier_notes=(str(exc),),
                )
                revision_examples.append(
                    {
                        "example_id": f"{case.case_id}_a{attempt}_first_revision",
                        "case_id": case.case_id,
                        "group_id": case.group_id,
                        "round_seed": collection_seed,
                        "on_policy_state_role": state_role,
                        "prior_failure_count": prior_failure_count,
                        "prompt": revision_prompt,
                        "target": target,
                        "messages": _sft_messages(revision_prompt, target),
                        "rejected": raw,
                        "failure_type": FailureType.PARSE_ERROR.value,
                        "risk_weight": risk_weight,
                        "certificate_hash": "parse_error_host_fallback",
                    }
                )
            external_feedback = {
                "status": "rejected",
                "failure_type": FailureType.PARSE_ERROR.value,
                "message": str(exc),
                "raw_output": raw,
            }
            step_payload["transition"] = external_feedback
            steps.append(step_payload)
            continue

        certificate = verifier.localize(action)
        is_rejected = not certificate.before.resolves_subgoal
        if is_rejected:
            verifier_rejections += 1
            risk_weight = float(certificate.before.risk_severity)
            teacher_prompt = render_teacher_prompt(
                prompt,
                failure_type=certificate.failure_type.value,
                selected_components=certificate.selected_components,
                verified_repair=to_jsonable(certificate.intervened_action),
                certificate_hash=certificate.certificate_hash,
            )
            examples.append(
                {
                    "example_id": f"{case.case_id}_a{attempt}_{certificate.certificate_hash[:8]}",
                    "case_id": case.case_id,
                    "group_id": case.group_id,
                    "round_seed": collection_seed,
                    "on_policy_state_role": state_role,
                    "prior_failure_count": prior_failure_count,
                    "student_prompt": prompt,
                    "teacher_prompt": teacher_prompt,
                    "rollout": raw,
                    "repair_spans": component_char_spans(
                        raw,
                        certificate.selected_components,
                        risk_weight=risk_weight,
                    ),
                    "risk_weight": risk_weight,
                    "error_type": certificate.failure_type.value,
                    "host_repair_verified": bool(
                        certificate.verified_minimal
                    ),
                    "host_verification_confidence": 1.0,
                    "teacher_output_verified": False,
                    "teacher_verified": False,
                    "teacher_confidence": 0.0,
                    "clean_control": False,
                    "certificate_hash": certificate.certificate_hash,
                    "oracle_action": to_jsonable(certificate.intervened_action),
                }
            )
            if not revision_examples:
                target = canonical_json(
                    to_jsonable(certificate.intervened_action)
                )
                revision_prompt = render_revision_prompt(
                    prompt,
                    rejected_output=raw,
                    failure_type=certificate.failure_type.value,
                    verifier_notes=certificate.before.notes,
                )
                revision_examples.append(
                    {
                        "example_id": f"{case.case_id}_a{attempt}_first_revision",
                        "case_id": case.case_id,
                        "group_id": case.group_id,
                        "round_seed": collection_seed,
                        "on_policy_state_role": state_role,
                        "prior_failure_count": prior_failure_count,
                        "prompt": revision_prompt,
                        "target": target,
                        "messages": _sft_messages(revision_prompt, target),
                        "rejected": raw,
                        "failure_type": certificate.failure_type.value,
                        "risk_weight": risk_weight,
                        "certificate_hash": certificate.certificate_hash,
                    }
                )
        elif include_clean_controls:
            examples.append(
                {
                    "example_id": f"{case.case_id}_a{attempt}_clean",
                    "case_id": case.case_id,
                    "group_id": case.group_id,
                    "round_seed": collection_seed,
                    "on_policy_state_role": state_role,
                    "prior_failure_count": prior_failure_count,
                    "student_prompt": prompt,
                    "teacher_prompt": prompt
                    + "\n<privileged_teacher_context>\n"
                    + canonical_json(
                        {
                            "host_verifier": "action_verified_for_current_subgoal",
                            "instruction": "保持当前正确行为分布。",
                        }
                    ),
                    "rollout": raw,
                    "repair_spans": [],
                    "risk_weight": 1.0,
                    "error_type": FailureType.NONE.value,
                    "host_verified": True,
                    "host_verification_confidence": 1.0,
                    "teacher_output_verified": False,
                    "teacher_verified": False,
                    "teacher_confidence": 0.0,
                    "clean_control": True,
                }
            )

        if assessment.resolves_subgoal:
            target = canonical_json(to_jsonable(action))
            verified_sft_examples.append(
                {
                    "example_id": f"{case.case_id}_a{attempt}_verified_sft",
                    "case_id": case.case_id,
                    "group_id": case.group_id,
                    "round_seed": collection_seed,
                    "on_policy_state_role": state_role,
                    "prior_failure_count": prior_failure_count,
                    "prompt": prompt,
                    "target": target,
                    "messages": _sft_messages(prompt, target),
                    "failure_type": FailureType.NONE.value,
                    "risk_weight": 1.0,
                    "host_verified": True,
                }
            )
            verified_kd_examples.append(
                {
                    "example_id": f"{case.case_id}_a{attempt}_verified_kd",
                    "case_id": case.case_id,
                    "group_id": case.group_id,
                    "round_seed": collection_seed,
                    "on_policy_state_role": state_role,
                    "prior_failure_count": prior_failure_count,
                    "student_prompt": prompt,
                    "teacher_prompt": prompt,
                    "rollout": target,
                    "repair_spans": [
                        {
                            "start": 0,
                            "end": max(len(target), 1),
                            "component": "__verified_full_response__",
                            "risk_weight": 1.0,
                        }
                    ],
                    "risk_weight": 1.0,
                    "error_type": FailureType.NONE.value,
                    "host_verification_confidence": 1.0,
                    "teacher_output_verified": False,
                    "teacher_verified": False,
                    "teacher_confidence": 0.0,
                    "clean_control": False,
                    "host_verified": True,
                }
            )

        next_state, reward, done, info = env.step(action)
        step_payload["parsed_action"] = to_jsonable(action)
        step_payload["transition"] = {
            "reward": reward,
            "done": done,
            "info": to_jsonable(info),
            "next_state": next_state,
        }
        steps.append(step_payload)
        external_feedback = None
        if done:
            break

    state = env.state
    return EpisodeResult(
        episode_id=f"{case.case_id}_seed{collection_seed}",
        case_id=case.case_id,
        seed=collection_seed,
        steps=steps,
        resolved=bool(state.resolved),
        escalated=bool(state.escalated),
        terminal_decision=(
            None if state.terminal_decision is None else state.terminal_decision.value
        ),
        parse_errors=parse_errors,
        verifier_rejections=verifier_rejections,
        opd_examples=examples,
        revision_examples=revision_examples,
        verified_sft_examples=verified_sft_examples,
        verified_kd_examples=verified_kd_examples,
    )


def collect_rollouts(
    cases: Sequence[CaseSpec],
    policy: TextPolicy,
    args: CollectionArguments,
) -> List[EpisodeResult]:
    selected = list(cases)
    if args.max_episodes > 0:
        selected = selected[: args.max_episodes]
    results: List[EpisodeResult] = []
    for index, case in enumerate(selected):
        results.append(
            collect_case(
                case,
                policy,
                collection_seed=args.seed + index * 1009,
                max_steps=args.max_steps,
                include_clean_controls=args.include_clean_controls,
            )
        )
    return results


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".partial",
        delete=False,
    ) as handle:
        count = 0
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return count


def write_collection(results: Sequence[EpisodeResult], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    opd_rows = []
    revision_rows = []
    verified_sft_rows = []
    verified_kd_rows = []
    for result in results:
        payload = asdict(result)
        opd_examples = payload.pop("opd_examples")
        revision_examples = payload.pop("revision_examples")
        verified_sft_examples = payload.pop("verified_sft_examples")
        verified_kd_examples = payload.pop("verified_kd_examples")
        raw_rows.append(payload)
        opd_rows.extend(opd_examples)
        revision_rows.extend(revision_examples)
        verified_sft_rows.extend(verified_sft_examples)
        verified_kd_rows.extend(verified_kd_examples)
    raw_path = output_dir / "raw_rollouts.jsonl"
    opd_path = output_dir / "opd_examples.jsonl"
    revision_path = output_dir / "revision_examples.jsonl"
    verified_sft_path = output_dir / "verified_sft_examples.jsonl"
    verified_kd_path = output_dir / "verified_kd_examples.jsonl"
    raw_count = _atomic_jsonl(raw_path, raw_rows)
    opd_count = _atomic_jsonl(opd_path, opd_rows)
    revision_count = _atomic_jsonl(revision_path, revision_rows)
    verified_sft_count = _atomic_jsonl(verified_sft_path, verified_sft_rows)
    verified_kd_count = _atomic_jsonl(verified_kd_path, verified_kd_rows)

    digest = hashlib.sha256()
    for path in (
        raw_path,
        opd_path,
        revision_path,
        verified_sft_path,
        verified_kd_path,
    ):
        digest.update(path.read_bytes())
    audit = {
        "format_version": "care-opd-online-v1",
        "episodes": raw_count,
        "opd_examples": opd_count,
        "revision_examples": revision_count,
        "verified_sft_examples": verified_sft_count,
        "verified_kd_examples": verified_kd_count,
        "resolved": sum(result.resolved for result in results),
        "escalated": sum(result.escalated for result in results),
        "parse_errors": sum(result.parse_errors for result in results),
        "verifier_rejections": sum(result.verifier_rejections for result in results),
        "host_verified_examples": sum(
            bool(
                row.get(
                    "host_repair_verified",
                    row.get("host_verified", False),
                )
            )
            for row in opd_rows
        ),
        "post_failure_recovery_examples": sum(
            row.get("on_policy_state_role") == "post_failure_recovery"
            for row in opd_rows
        ),
        "teacher_output_verified_examples": sum(
            bool(row.get("teacher_output_verified")) for row in opd_rows
        ),
        "collection_sha256": digest.hexdigest(),
    }
    audit_path = output_dir / "collection_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect genuine CARE-OPD student rollouts.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--adapter")
    parser.add_argument("--device")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-clean-controls", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    namespace = _parser().parse_args(argv)
    rows = jsonl_load(str(namespace.cases))
    cases = [case_from_mapping(row) for row in rows]
    generation = GenerationArguments(
        model=namespace.model,
        tokenizer=namespace.tokenizer,
        adapter=namespace.adapter,
        device=namespace.device,
        dtype=namespace.dtype,
        max_new_tokens=namespace.max_new_tokens,
        temperature=namespace.temperature,
        top_p=namespace.top_p,
        do_sample=not namespace.greedy,
        trust_remote_code=namespace.trust_remote_code,
    )
    collection = CollectionArguments(
        cases=namespace.cases,
        output_dir=namespace.output_dir,
        max_episodes=namespace.max_episodes,
        max_steps=namespace.max_steps,
        seed=namespace.seed,
        include_clean_controls=not namespace.no_clean_controls,
    )
    policy = HFCausalPolicy(generation)
    results = collect_rollouts(cases, policy, collection)
    audit = write_collection(results, namespace.output_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
