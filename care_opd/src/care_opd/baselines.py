"""Inference-time baseline wrappers used under the same action budget.

These wrappers only construct additional context. Model calls and environment execution are
owned by :mod:`care_opd.rollout`, which lets every condition share decoding and accounting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TOKEN_PATTERN = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 0.0


@dataclass(frozen=True)
class RecoveryExemplar:
    failure_signature: str
    diagnosis: str
    recovery_action: Mapping[str, Any]
    failure_type: str = "unknown"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveryExemplar":
        return cls(
            failure_signature=str(value["failure_signature"]),
            diagnosis=str(value["diagnosis"]),
            recovery_action=dict(value["recovery_action"]),
            failure_type=str(value.get("failure_type", "unknown")),
        )


class RecoveryExemplarIndex:
    def __init__(self, exemplars: Sequence[RecoveryExemplar]) -> None:
        self.exemplars = list(exemplars)

    @classmethod
    def from_json(cls, path: Path) -> "RecoveryExemplarIndex":
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls([RecoveryExemplar.from_dict(value) for value in values])

    def search(self, failure_observation: str, k: int = 1) -> list[RecoveryExemplar]:
        ranked = sorted(
            self.exemplars,
            key=lambda item: (
                _jaccard(failure_observation, item.failure_signature),
                item.failure_type,
                item.failure_signature,
            ),
            reverse=True,
        )
        return ranked[: max(0, k)]


def reflexion_context(
    *,
    task: str,
    trajectory: Sequence[Mapping[str, Any]],
    failure_observation: str,
    remaining_budget: int,
) -> list[dict[str, str]]:
    """Two-call Reflexion context.

    The rollout harness must charge the reflection generation as one model call and preserve
    the same environment-action budget used by all methods.
    """

    compact_trace = json.dumps(list(trajectory), ensure_ascii=False, separators=(",", ":"))
    reflection_prompt = (
        "你正在分析一次工具智能体失败。只写一条可执行、可验证的短反思，"
        "指出失败原因以及下一次不能重复的动作；不得臆造未观察到的证据。\n"
        f"任务：{task}\n轨迹：{compact_trace}\n环境反馈：{failure_observation}\n"
        f"剩余动作预算：{remaining_budget}"
    )
    return [{"role": "user", "content": reflection_prompt}]


def action_after_reflection_context(
    *,
    public_messages: Sequence[Mapping[str, str]],
    reflection: str,
    remaining_budget: int,
) -> list[dict[str, str]]:
    messages = [dict(message) for message in public_messages]
    messages.append(
        {
            "role": "user",
            "content": (
                "以下是上一轮失败后形成的反思。它不是证据，只能用于选择恢复动作：\n"
                f"{reflection}\n剩余动作预算：{remaining_budget}\n"
                "现在仅输出符合 schema 的下一动作 JSON。"
            ),
        }
    )
    return messages


def paladin_retrieval_context(
    *,
    public_messages: Sequence[Mapping[str, str]],
    failure_observation: str,
    index: RecoveryExemplarIndex,
    k: int = 1,
) -> list[dict[str, str]]:
    """PALADIN-style inference-time retrieval baseline.

    Retrieved records describe recovery strategies, never hidden evidence or a final answer.
    """

    examples = index.search(failure_observation, k=k)
    payload = [
        {
            "failure_signature": item.failure_signature,
            "diagnosis": item.diagnosis,
            "recovery_action": item.recovery_action,
        }
        for item in examples
    ]
    messages = [dict(message) for message in public_messages]
    messages.append(
        {
            "role": "user",
            "content": (
                "从公开恢复案例库检索到以下相似故障。案例不是当前病例证据，"
                "只可参考恢复策略：\n"
                f"{json.dumps(payload, ensure_ascii=False)}\n"
                "结合当前真实环境反馈，仅输出下一动作 JSON。"
            ),
        }
    )
    return messages


def agent_debug_context(
    *,
    task: str,
    trajectory: Sequence[Mapping[str, Any]],
    failure_observation: str,
) -> list[dict[str, str]]:
    compact_trace = json.dumps(list(trajectory), ensure_ascii=False, separators=(",", ":"))
    return [
        {
            "role": "user",
            "content": (
                "定位轨迹中最早的根因错误。返回 JSON，字段为 "
                "first_error_step、module、failure_type、feedback。feedback 不得泄露答案。\n"
                f"任务：{task}\n轨迹：{compact_trace}\n环境反馈：{failure_observation}"
            ),
        }
    ]

