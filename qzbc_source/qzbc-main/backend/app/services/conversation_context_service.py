from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import PatientEducationChat, RoleAIChat


_FOLLOW_UP_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(它|他|她|这个|这种|那个|那种|上述|上面|前面|刚才|其|该)",
        r"(它|这个|这种|那个|上述|上面|前面|刚才|其|该)",
        r"^(还有|然后|那么|那|再|继续|具体|为什么|怎么)(呢|吗|说|讲|解释|分析|介绍|？|\?|$)",
        r"^(有无|是否)(禁忌|副作用|注意事项|区别|联系)",
        r"^(请)?(从|用).{0,12}(角度|方式)(再|重新)?(回答|解释)",
        r"^(重新|换个|再)(回答|解释|说)",
        r"^针对你(上|前)一条回答",
    )
)

_ANSWER_REFERENCE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(上|前)一条回答",
        r"你(刚才|上次|前面)说",
        r"(上述|上面|前面)回答",
        r"第[一二三四五六七八九十\d]+(点|条|段)",
        r"这段(内容|话|回答)",
    )
)

_TOPIC_STOP_FRAGMENTS = {
    "什么", "怎么", "如何", "哪些", "请问", "一下", "详细", "讲解", "解释", "分析", "介绍",
    "回答", "问题", "区别", "联系", "还有", "继续", "具体", "为什么", "时候", "患者", "中医",
    "是否", "可以", "需要", "给出", "复习", "步骤", "进行", "关于", "针对", "上述", "前面", "刚才",
    "的", "是", "我", "你",
}


class ConversationContextService:
    """Build deterministic, role-isolated short-term context from authoritative SQL chat history.

    Mem0 and Graphiti remain the long-term semantic and temporal layers. Recent turns are
    loaded directly because a vector search must never be the only way to remember the
    immediately preceding question.
    """

    def __init__(self, db: Session):
        self.db = db

    def build(self, *, role: str, user_id: int, current_message: str) -> dict[str, Any]:
        settings = get_settings()
        if not settings.conversation_context_enabled:
            return _empty_context(current_message)

        rows = self._recent_rows(role=role, user_id=user_id, limit=settings.conversation_recent_turns)
        available_turns = self._bounded_turns(rows)
        explicit_follow_up = _is_follow_up(current_message, bool(available_turns))
        topic_continuation = bool(available_turns) and _shares_topic(
            current_message,
            str(available_turns[-1].get("question") or ""),
        )
        follow_up = explicit_follow_up or topic_continuation
        turns = _relevant_recent_turns(available_turns) if follow_up else []
        answer_reference = follow_up and _references_previous_answer(current_message)
        context_mode = (
            "no_history"
            if not available_turns
            else "answer_reference"
            if answer_reference
            else "topic_continuation"
            if follow_up
            else "new_topic"
        )
        retrieval_query = _retrieval_query(current_message, turns, follow_up)
        return {
            "strategy": "recent_window+mem0+graphiti",
            "scope": {"role": role, "user_id": user_id},
            "recent_turns": turns,
            "turn_count": len(turns),
            "available_turn_count": len(available_turns),
            "follow_up_detected": follow_up,
            "answer_reference_detected": answer_reference,
            "include_assistant_answers": answer_reference,
            "context_mode": context_mode,
            "retrieval_query": retrieval_query,
        }

    def _recent_rows(self, *, role: str, user_id: int, limit: int) -> list[Any]:
        if role == "patient":
            rows = self.db.scalars(
                select(PatientEducationChat)
                .where(PatientEducationChat.patient_id == user_id)
                .order_by(PatientEducationChat.created_at.desc(), PatientEducationChat.id.desc())
                .limit(limit)
            ).all()
        elif role in {"doctor", "student"}:
            rows = self.db.scalars(
                select(RoleAIChat)
                .where(RoleAIChat.user_id == user_id, RoleAIChat.role == role)
                .order_by(RoleAIChat.created_at.desc(), RoleAIChat.id.desc())
                .limit(limit)
            ).all()
        else:
            rows = []
        return list(reversed(rows))

    def _bounded_turns(self, rows: list[Any]) -> list[dict[str, Any]]:
        settings = get_settings()
        remaining = max(0, settings.conversation_context_max_chars)
        selected: list[dict[str, Any]] = []
        # Spend the budget on the newest turns first, then restore chronological order.
        for row in reversed(rows):
            question = _compact(str(row.question), settings.conversation_question_max_chars)
            answer = _compact(str(row.answer), settings.conversation_answer_max_chars)
            if not question or remaining <= 0:
                continue
            required = len(question) + len(answer)
            if required > remaining:
                answer = _compact(answer, max(0, remaining - len(question)))
            if not answer and len(question) > remaining:
                question = _compact(question, remaining)
            if not question:
                continue
            turn = {
                "turn_id": int(row.id),
                "question": question,
                "answer": answer,
                "created_at": row.created_at.isoformat(),
            }
            if hasattr(row, "allowed"):
                turn["allowed"] = bool(row.allowed)
                grounding = row.llm_result.get("grounding") if isinstance(row.llm_result, dict) else {}
                provider = str(row.llm_result.get("provider") or "") if isinstance(row.llm_result, dict) else ""
                turn["answer_trusted"] = bool(
                    row.allowed
                    and (
                        provider == "patient_safety_policy"
                        or (
                            isinstance(grounding, dict)
                            and int(grounding.get("used_evidence_count") or 0) > 0
                            and not grounding.get("missing_evidence", False)
                        )
                    )
                )
            else:
                # Doctor/student model answers are not authoritative patient
                # records. Only deterministic structured-record responses may
                # be replayed as assistant messages on a later follow-up.
                provider = str(row.llm_result.get("provider") or "") if isinstance(row.llm_result, dict) else ""
                turn["answer_trusted"] = provider == "doctor_business_context_policy"
            selected.append(turn)
            remaining -= len(question) + len(answer)
        return list(reversed(selected))


def context_memory_items(context: dict[str, Any]) -> list[dict[str, Any]]:
    scope = context.get("scope") if isinstance(context.get("scope"), dict) else {}
    role = str(scope.get("role") or "")
    user_id = scope.get("user_id")
    items = []
    for turn in context.get("recent_turns") or []:
        if not isinstance(turn, dict):
            continue
        items.append(
            {
                "source": "sql_recent_chat",
                "owner_user_id": user_id,
                "owner_role": role,
                "memory_type": "short_term_conversation_turn",
                "content": {
                    "turn_id": turn.get("turn_id"),
                    "question": turn.get("question", ""),
                    "answer": turn.get("answer", ""),
                    "created_at": turn.get("created_at"),
                    "allowed": turn.get("allowed"),
                },
            }
        )
    return items


def _is_follow_up(message: str, has_history: bool) -> bool:
    if not has_history:
        return False
    value = re.sub(r"\s+", "", message.strip())
    if not value:
        return False
    return any(pattern.search(value) for pattern in _FOLLOW_UP_PATTERNS)


def _retrieval_query(message: str, turns: list[dict[str, Any]], follow_up: bool) -> str:
    current = message.strip()
    if not follow_up or not turns:
        return current
    previous = turns[-1]
    previous_question = str(previous.get("question") or "").strip()
    parts = [f"当前追问：{current}", f"上一问题：{previous_question}"]
    return "\n".join(parts)


def _relevant_recent_turns(turns: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    """Keep only the latest contiguous topic chain, never the whole chat history."""
    if not turns:
        return []
    selected = [turns[-1]]
    anchor = str(turns[-1].get("question") or "")
    for turn in reversed(turns[:-1]):
        if len(selected) >= limit or not _shares_topic(anchor, str(turn.get("question") or "")):
            break
        selected.append(turn)
    return list(reversed(selected))


def _references_previous_answer(message: str) -> bool:
    value = re.sub(r"\s+", "", message.strip())
    return bool(value) and any(pattern.search(value) for pattern in _ANSWER_REFERENCE_PATTERNS)


def _shares_topic(left: str, right: str) -> bool:
    left_terms = _topic_terms(left)
    right_terms = _topic_terms(right)
    return bool(left_terms and right_terms and left_terms & right_terms)


def _topic_terms(value: str) -> set[str]:
    sequences = re.findall(r"[\u3400-\u9fff]{2,}", value)
    terms: set[str] = set()
    for sequence in sequences:
        for width in (2, 3, 4, 5):
            for index in range(max(0, len(sequence) - width + 1)):
                term = sequence[index : index + width]
                if any(fragment in term for fragment in _TOPIC_STOP_FRAGMENTS):
                    continue
                terms.add(term)
    return terms


def _compact(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _empty_context(message: str) -> dict[str, Any]:
    return {
        "strategy": "disabled",
        "scope": {},
        "recent_turns": [],
        "turn_count": 0,
        "available_turn_count": 0,
        "follow_up_detected": False,
        "answer_reference_detected": False,
        "include_assistant_answers": False,
        "context_mode": "disabled",
        "retrieval_query": message.strip(),
    }
