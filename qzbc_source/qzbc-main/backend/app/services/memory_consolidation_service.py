from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import RoleMemory


MEMORY_SOURCE = "memory_consolidation"
_MAX_CONTENT_LENGTH = 500


class MemoryAction(StrEnum):
    ADD = "add"
    UPDATE = "update"
    UNCHANGED = "unchanged"
    IGNORE = "ignore"
    BLOCK = "block"
    DELETE = "delete"


class MemoryDecision(BaseModel):
    action: MemoryAction
    category: str = ""
    memory_key: str = ""
    content: str = ""
    importance: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str = ""
    retention: str = "none"


class ConsolidationResult(BaseModel):
    decision: MemoryDecision
    memory_id: int | None = None
    external_sync_required: bool = False


@dataclass(frozen=True)
class _PatternRule:
    category: str
    memory_key: str
    roles: frozenset[str]
    pattern: re.Pattern[str]
    importance: float


_PROTECTED_FACTS = re.compile(
    r"(过敏|不良反应|病史|既往史|诊断|确诊|处方|剂量|用量|每天\s*\d|"
    r"血压|血糖|检验|检查结果|怀孕|妊娠|哺乳|身份证|手机号|住址)",
    re.IGNORECASE,
)
_TEMPORARY_CONTEXT = re.compile(r"(今天|明天|后天|这次|本次|临时|一会儿|待会儿|本周|这个月)")
_EXPLICIT_REMEMBER = re.compile(r"^(?:请|麻烦你)?(?:帮我)?(?:记住|记一下|长期记住|以后记得)[：,:，\s]*(.+)$")
_EXPLICIT_FORGET = re.compile(r"^(?:请|麻烦你)?(?:帮我)?(?:忘记|不要再记住)[：,:，\s]*(.+)$")
_STABILITY_CUE = re.compile(r"(以后|今后|一直|长期|通常|总是|偏好|喜欢|希望你|回答时|我的目标|薄弱点|主要关注)")
_LONG_TERM_PREFERENCE_SCOPE = re.compile(
    r"(以后|今后|一直|长期|回答时|我的(?:偏好|学习目标|薄弱点)|我的目标|"
    r"我(?:更)?(?:喜欢|偏好)|主要关注)"
)
_STUDENT_SOURCE_STYLE = re.compile(
    r"(引经据典|经典(?:原文|著作)?|权威(?:书籍|教材|著作)|参考书|延伸阅读|推荐.*(?:书|教材)|引用.*(?:经典|教材|著作))"
)

_STUDENT_SOURCE_STYLE_MEMORY = "学生偏好：复杂中医药知识问题优先依据可追溯的中医经典或权威教材回答；尽量注明书名和可定位篇章，并给出延伸阅读。"

_RULES = (
    _PatternRule(
        "response_preference",
        "response_detail",
        frozenset({"doctor", "student", "patient"}),
        re.compile(r"(?:我(?:更)?(?:喜欢|希望|偏好)|以后|回答时|请)(?:你)?[^。！？]{0,35}(简短|简洁|详细|展开|精炼)"),
        0.88,
    ),
    _PatternRule(
        "response_preference",
        "explanation_level",
        frozenset({"doctor", "student", "patient"}),
        re.compile(r"(?:我(?:更)?(?:喜欢|希望|偏好)|以后|回答时|请)(?:你)?[^。！？]{0,35}(通俗|易懂|专业|教学|学术|少用术语)"),
        0.9,
    ),
    _PatternRule(
        "response_preference",
        "response_format",
        frozenset({"doctor", "student", "patient"}),
        re.compile(r"(?:我(?:更)?(?:喜欢|希望|偏好)|以后|回答时|请)(?:你)?[^。！？]{0,40}(表格|分点|先说结论|先给结论|逐步|步骤)"),
        0.88,
    ),
    _PatternRule(
        "learning_goal",
        "learning_goal",
        frozenset({"student"}),
        re.compile(r"(?:我的)?(?:学习目标|目标是|正在学习|想系统学习|计划学习)[：,:，\s]*(.{2,120})"),
        0.86,
    ),
    _PatternRule(
        "learning_weakness",
        "learning_weakness",
        frozenset({"student"}),
        re.compile(r"(?:我的)?(?:薄弱点|弱项|总是分不清|不太理解|容易弄错)[：,:，\s]*(.{2,120})"),
        0.9,
    ),
    _PatternRule(
        "professional_focus",
        "professional_focus",
        frozenset({"doctor"}),
        re.compile(r"(?:我)?(?:主要从事|专业方向|长期关注|主要关注|擅长)[：,:，\s]*(.{2,120})"),
        0.86,
    ),
)


class MemoryConsolidationAgent:
    """Conservatively promote user-authored chat facts into role-scoped memory.

    SQL chat rows are the immutable conversation archive. This agent only manages
    reusable user preferences/goals. Clinical facts remain in their authoritative
    structured tables and are never inferred into free-text memory.
    """

    def __init__(self, db: Session):
        self.db = db

    def decide(self, *, role: str, user_message: str, explicit: bool = False) -> MemoryDecision:
        _validate_role(role)
        text = _compact(user_message)
        if not text:
            return MemoryDecision(action=MemoryAction.IGNORE, reason="empty_message")

        forget_match = _EXPLICIT_FORGET.match(text)
        if forget_match:
            return MemoryDecision(
                action=MemoryAction.DELETE,
                content=_compact(forget_match.group(1)),
                importance=1,
                confidence=1,
                reason="explicit_forget_instruction",
            )

        remember_match = _EXPLICIT_REMEMBER.match(text)
        candidate = _compact(remember_match.group(1)) if remember_match else text
        is_explicit = explicit or remember_match is not None
        # A one-off imperative such as “请详细讲解两者的区别” is a
        # question, not a durable preference.  Automatic promotion requires a
        # clear persistence scope; explicit "请记住" is always sufficient.
        has_long_term_scope = is_explicit or bool(_LONG_TERM_PREFERENCE_SCOPE.search(candidate))

        # Medical facts take precedence over any incidental preference-like
        # phrase elsewhere in the same message.  A case stem can contain a
        # long-term history and a generic instruction such as “请……教学……”; it
        # must continue through medical reasoning rather than being mistaken
        # for a pure response-style preference.  Structured clinical records
        # remain authoritative and are never promoted to free-text memory.
        if _PROTECTED_FACTS.search(candidate):
            return MemoryDecision(
                action=MemoryAction.BLOCK,
                content=candidate,
                importance=1,
                confidence=1,
                reason="protected_medical_or_identity_fact_use_structured_record",
            )

        if _TEMPORARY_CONTEXT.search(candidate):
            return MemoryDecision(
                action=MemoryAction.IGNORE,
                content=candidate,
                confidence=0.98,
                reason="temporary_context_stays_in_short_term_history",
            )

        # This preference used to fall through to the generic explicit-memory
        # branch and was persisted verbatim (including typos and filler such
        # as “我是一个学生”).  It is a stable, role-specific teaching setting,
        # so retain its meaning in one canonical sentence instead.
        if has_long_term_scope and role == "student" and _STUDENT_SOURCE_STYLE.search(candidate):
            return MemoryDecision(
                action=MemoryAction.ADD,
                category="learning_preference",
                memory_key="teaching_source_style",
                content=_STUDENT_SOURCE_STYLE_MEMORY,
                importance=max(0.93, 0.97 if is_explicit else 0),
                confidence=0.98,
                reason="student_source_style_canonicalized",
                retention="long_term",
            )

        if has_long_term_scope:
            for rule in _RULES:
                match = rule.pattern.search(candidate)
                if role not in rule.roles or not match:
                    continue
                normalized = _compact(match.group(0))
                return MemoryDecision(
                    action=MemoryAction.ADD,
                    category=rule.category,
                    memory_key=rule.memory_key,
                    content=normalized,
                    importance=max(rule.importance, 0.95 if is_explicit else 0),
                    confidence=0.96,
                    reason="explicit_rule_match" if is_explicit else "stable_role_memory_rule_match",
                    retention="long_term",
                )

        if is_explicit and _STABILITY_CUE.search(candidate):
            return MemoryDecision(
                action=MemoryAction.ADD,
                category="user_preference",
                memory_key="explicit_preference",
                content=candidate,
                importance=0.95,
                confidence=0.9,
                reason="explicit_stable_memory_instruction",
                retention="long_term",
            )
        return MemoryDecision(
            action=MemoryAction.IGNORE,
            content=candidate,
            confidence=0.9,
            reason="no_stable_reusable_user_fact",
        )

    def preference_acknowledgement(self, *, owner_role: str, user_message: str) -> str | None:
        """Return a direct acknowledgement for a pure memory instruction.

        Preference instructions are not medical questions.  Routing them
        through knowledge retrieval made generic evidence snippets look like
        answers (for example, a student asking to remember a study preference
        could receive a cough diagnosis workflow).  The actual write still
        happens through :meth:`consolidate_chat` after the chat is saved.
        """
        decision = self.decide(role=owner_role, user_message=user_message)
        if decision.action != MemoryAction.ADD:
            return None
        label = _category_label(decision.category)
        role_action = {
            "student": "后续会按这个学习偏好组织教学回答",
            "doctor": "后续会按这个偏好组织专业辅助回答",
            "patient": "后续会按这个偏好调整科普表达",
        }.get(owner_role, "后续会按这个偏好组织回答")
        return f"已记录为{label}：{decision.content}\n\n{role_action}；它只影响回答方式和学习支持，不会替代知识来源或生成诊疗、处方结论。"

    def consolidate_chat(
        self,
        *,
        owner_user_id: int,
        owner_role: str,
        user_message: str,
        source_chat_id: int,
        trace_id: str | None,
    ) -> ConsolidationResult:
        decision = self.decide(role=owner_role, user_message=user_message)
        if decision.action == MemoryAction.DELETE:
            deleted = self.forget_matching(
                owner_user_id=owner_user_id,
                owner_role=owner_role,
                query=decision.content,
            )
            decision.reason = f"explicit_forget_instruction_deleted_{deleted}"
            return ConsolidationResult(decision=decision)
        if decision.action != MemoryAction.ADD:
            return ConsolidationResult(decision=decision)
        return self._upsert(
            owner_user_id=owner_user_id,
            owner_role=owner_role,
            decision=decision,
            source_mode="automatic_chat_consolidation",
            source_chat_id=source_chat_id,
            trace_id=trace_id,
        )

    def remember_explicit(self, *, owner_user_id: int, owner_role: str, content: str) -> ConsolidationResult:
        decision = self.decide(role=owner_role, user_message=content, explicit=True)
        if decision.action == MemoryAction.BLOCK:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="过敏史、病史、诊断、处方、剂量和身份信息必须写入对应结构化资料，不能保存为自由文本长期记忆。",
            )
        if decision.action != MemoryAction.ADD:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="这段内容属于临时上下文或缺少稳定偏好/目标，未写入长期记忆。",
            )
        return self._upsert(
            owner_user_id=owner_user_id,
            owner_role=owner_role,
            decision=decision,
            source_mode="explicit_user_instruction",
            source_chat_id=None,
            trace_id=None,
        )

    def _upsert(
        self,
        *,
        owner_user_id: int,
        owner_role: str,
        decision: MemoryDecision,
        source_mode: str,
        source_chat_id: int | None,
        trace_id: str | None,
    ) -> ConsolidationResult:
        row = self._active_by_key(owner_user_id, owner_role, decision.memory_key)
        now = datetime.now(timezone.utc)
        fingerprint = _fingerprint(decision.category, decision.memory_key, decision.content)
        if row:
            current = dict(row.content_json or {})
            if current.get("fingerprint") == fingerprint:
                current["source_chat_ids"] = _append_unique(current.get("source_chat_ids"), source_chat_id)
                current["trace_ids"] = _append_unique(current.get("trace_ids"), trace_id)
                row.content_json = current
                row.updated_at = now
                self.db.flush()
                decision.action = MemoryAction.UNCHANGED
                decision.reason = "deduplicated_existing_memory"
                return ConsolidationResult(decision=decision, memory_id=row.id)
            previous_versions = list(current.get("previous_versions") or [])
            previous_versions.append(
                {
                    "content": current.get("content"),
                    "fingerprint": current.get("fingerprint"),
                    "source_chat_ids": current.get("source_chat_ids") or [],
                    "trace_ids": current.get("trace_ids") or [],
                    "revision": current.get("revision", 1),
                }
            )
            current.update(
                {
                    "content": decision.content,
                    "memory_key": decision.memory_key,
                    "category": decision.category,
                    "importance": decision.importance,
                    "confidence": decision.confidence,
                    "retention": decision.retention,
                    "status": "active",
                    "fingerprint": fingerprint,
                    "source_mode": source_mode,
                    # The newest source supports the newest value. Older sources
                    # remain in previous_versions rather than falsely proving it.
                    "source_chat_ids": _append_unique([], source_chat_id),
                    "trace_ids": _append_unique([], trace_id),
                    "previous_versions": previous_versions[-5:],
                    "revision": int(current.get("revision") or 1) + 1,
                }
            )
            row.content_json = dict(current)
            row.updated_at = now
            decision.action = MemoryAction.UPDATE
        else:
            row = RoleMemory(
                owner_user_id=owner_user_id,
                owner_role=owner_role,
                memory_type=decision.category,
                source=MEMORY_SOURCE,
                content_json={
                    "content": decision.content,
                    "memory_key": decision.memory_key,
                    "category": decision.category,
                    "importance": decision.importance,
                    "confidence": decision.confidence,
                    "retention": decision.retention,
                    "status": "active",
                    "fingerprint": fingerprint,
                    "source_mode": source_mode,
                    "source_chat_ids": _append_unique([], source_chat_id),
                    "trace_ids": _append_unique([], trace_id),
                    "revision": 1,
                },
            )
            self.db.add(row)
        self.db.flush()
        return ConsolidationResult(decision=decision, memory_id=row.id, external_sync_required=True)

    def _active_by_key(self, owner_user_id: int, owner_role: str, memory_key: str) -> RoleMemory | None:
        rows = self.db.scalars(
            select(RoleMemory)
            .where(
                RoleMemory.owner_user_id == owner_user_id,
                RoleMemory.owner_role == owner_role,
                RoleMemory.source == MEMORY_SOURCE,
            )
            .order_by(RoleMemory.updated_at.desc(), RoleMemory.id.desc())
        ).all()
        return next(
            (
                row
                for row in rows
                if (row.content_json or {}).get("status", "active") == "active"
                and (row.content_json or {}).get("memory_key") == memory_key
            ),
            None,
        )

    def list_active(self, *, owner_user_id: int, owner_role: str) -> list[dict[str, Any]]:
        _validate_role(owner_role)
        rows = self.db.scalars(
            select(RoleMemory)
            .where(
                RoleMemory.owner_user_id == owner_user_id,
                RoleMemory.owner_role == owner_role,
                RoleMemory.source == MEMORY_SOURCE,
            )
            .order_by(RoleMemory.updated_at.desc(), RoleMemory.id.desc())
        ).all()
        return [_serialize(row) for row in rows if (row.content_json or {}).get("status", "active") == "active"]

    def forget(self, *, owner_user_id: int, owner_role: str, memory_id: int) -> RoleMemory:
        row = self.db.scalar(
            select(RoleMemory).where(
                RoleMemory.id == memory_id,
                RoleMemory.owner_user_id == owner_user_id,
                RoleMemory.owner_role == owner_role,
                RoleMemory.source == MEMORY_SOURCE,
            )
        )
        if not row or (row.content_json or {}).get("status", "active") != "active":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Long-term memory not found")
        content = dict(row.content_json or {})
        content["status"] = "deleted"
        content["deleted_reason"] = "explicit_user_forget"
        row.content_json = content
        row.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        return row

    def forget_matching(self, *, owner_user_id: int, owner_role: str, query: str) -> int:
        terms = {term for term in re.findall(r"[\w\u3400-\u9fff]{2,}", query.lower()) if term not in {"这个", "之前", "记忆", "内容", "偏好"}}
        rows = self.list_active(owner_user_id=owner_user_id, owner_role=owner_role)
        matching_ids = [
            int(item["id"])
            for item in rows
            if not terms or any(term in str(item.get("content") or "").lower() for term in terms)
        ]
        for memory_id in matching_ids:
            self.forget(owner_user_id=owner_user_id, owner_role=owner_role, memory_id=memory_id)
        return len(matching_ids)

    def detach_chat_source(self, *, owner_user_id: int, owner_role: str, source_chat_id: int) -> list[int]:
        """Forget a derived memory only when the deleted chat was its sole provenance."""
        synced_ids: list[int] = []
        rows = self.db.scalars(
            select(RoleMemory).where(
                RoleMemory.owner_user_id == owner_user_id,
                RoleMemory.owner_role == owner_role,
                RoleMemory.source == MEMORY_SOURCE,
            )
        ).all()
        for row in rows:
            content = dict(row.content_json or {})
            source_ids = [int(value) for value in content.get("source_chat_ids") or [] if str(value).isdigit()]
            if source_chat_id not in source_ids:
                continue
            remaining = [value for value in source_ids if value != source_chat_id]
            content["source_chat_ids"] = remaining
            if not remaining and content.get("source_mode") == "automatic_chat_consolidation":
                content["status"] = "deleted"
                content["deleted_reason"] = "source_chat_deleted"
            row.content_json = content
            row.updated_at = datetime.now(timezone.utc)
            synced_ids.append(row.id)
        self.db.flush()
        return synced_ids


def serialize_memory(row: RoleMemory) -> dict[str, Any]:
    return _serialize(row)


def sync_external_long_term_memory(*, role: str, user_id: int, memory_id: int) -> None:
    """Best-effort Mem0 mirror. SQL remains authoritative."""
    try:
        from app.core.config import get_settings

        if get_settings().memory_backend != "mem0":
            return
        from app.db.session import SessionLocal
        from app.mcp.adapters.mem0_real import sync_long_term_memory

        with SessionLocal() as db:
            row = db.get(RoleMemory, memory_id)
            if not row or row.owner_role != role or row.owner_user_id != user_id:
                return
            sync_long_term_memory(role=role, user_id=user_id, memory_id=memory_id, content_json=row.content_json)
    except Exception:
        pass


def _serialize(row: RoleMemory) -> dict[str, Any]:
    content = row.content_json or {}
    category = str(content.get("category") or row.memory_type)
    normalized_content = canonicalize_memory_content(
        role=row.owner_role,
        category=category,
        memory_key=str(content.get("memory_key") or ""),
        content=str(content.get("content") or ""),
    )
    return {
        "id": row.id,
        "category": category,
        "category_label": _category_label(category),
        "memory_key": content.get("memory_key") or "",
        "content": normalized_content,
        "importance": content.get("importance"),
        "confidence": content.get("confidence"),
        "source_mode": content.get("source_mode"),
        "revision": content.get("revision", 1),
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def _category_label(category: str) -> str:
    return {
        "response_preference": "回答偏好",
        "user_preference": "个人偏好",
        "learning_goal": "学习目标",
        "learning_weakness": "学习薄弱点",
        "learning_preference": "学习偏好",
        "professional_focus": "专业关注方向",
    }.get(category, "长期记忆")


def canonicalize_memory_content(*, role: str, category: str, memory_key: str, content: str) -> str:
    """Normalize legacy display/retrieval text without changing medical facts.

    This intentionally handles only a narrowly-defined, non-medical learning
    preference.  Structured medical records must never be rewritten here.
    """
    value = _compact(content)
    if role == "student" and (memory_key == "teaching_source_style" or (category in {"user_preference", "learning_preference"} and _STUDENT_SOURCE_STYLE.search(value))):
        return _STUDENT_SOURCE_STYLE_MEMORY
    return value


def _append_unique(values: Any, value: Any) -> list[Any]:
    result = list(values) if isinstance(values, list) else []
    if value is not None and value not in result:
        result.append(value)
    return result[-10:]


def _fingerprint(category: str, memory_key: str, content: str) -> str:
    normalized = re.sub(r"\s+", "", content).lower()
    return hashlib.sha256(f"{category}|{memory_key}|{normalized}".encode("utf-8")).hexdigest()


def _compact(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;：:")
    return text[:_MAX_CONTENT_LENGTH]


def _validate_role(role: str) -> None:
    if role not in {"doctor", "student", "patient"}:
        raise ValueError(f"unsupported memory role: {role}")
