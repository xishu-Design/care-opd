"""Typed, dependency-free schemas for the CARE-OPD research environment.

The environment models an evidence-recovery workflow for *synthetic* TCM
follow-up safety review.  It is an algorithmic benchmark, not a diagnostic
system and none of the records produced by this package are clinical advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class EvidenceSource(StrEnum):
    MEM0 = "search_mem0"
    GRAPHITI = "search_graphiti"
    LIGHTRAG = "search_lightrag"
    DOCTOR = "ask_doctor"


class ActionKind(StrEnum):
    SEARCH_MEM0 = "search_mem0"
    SEARCH_GRAPHITI = "search_graphiti"
    SEARCH_LIGHTRAG = "search_lightrag"
    ASK_DOCTOR = "ask_doctor"
    DECIDE = "decide"
    HUMAN_REVIEW = "human_review"

    @property
    def is_tool(self) -> bool:
        return self in {
            ActionKind.SEARCH_MEM0,
            ActionKind.SEARCH_GRAPHITI,
            ActionKind.SEARCH_LIGHTRAG,
            ActionKind.ASK_DOCTOR,
        }


class Decision(StrEnum):
    CLEAR = "clear"
    BLOCK = "block"
    REVIEW = "review"


class ObservationStatus(StrEnum):
    FOUND = "found"
    NO_MATCH = "no_match"
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"
    NEEDS_PREREQUISITE = "needs_prerequisite"
    INVALID_QUERY = "invalid_query"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    BUDGET_EXHAUSTED = "budget_exhausted"


class FailureType(StrEnum):
    NONE = "none"
    PARSE_ERROR = "parse_error"
    INVALID_ACTION = "invalid_action"
    WRONG_SOURCE = "wrong_source"
    QUERY_UNDERSPECIFIED = "query_underspecified"
    REPEATED_NO_MATCH = "repeated_no_match"
    IGNORED_UNAVAILABLE = "ignored_unavailable"
    STALE_EVIDENCE_USED = "stale_evidence_used"
    CONFLICT_IGNORED = "conflict_ignored"
    MISSING_PREREQUISITE = "missing_prerequisite"
    PREMATURE_DECISION = "premature_decision"
    UNSUPPORTED_DECISION = "unsupported_decision"
    UNSAFE_CLEAR = "unsafe_clear"
    SAFE_FALSE_BLOCK = "safe_false_block"
    UNNECESSARY_ESCALATION = "unnecessary_escalation"
    CYCLE = "cycle"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    OTHER = "other"


class RiskFamily(StrEnum):
    SPECIAL_STATE = "special_state"
    INCOMPATIBLE_PAIR = "incompatible_pair"
    ALLERGY = "allergy"
    CONCURRENT_MEDICATION = "concurrent_medication"
    CLEAR_CONTROL = "clear_control"
    MULTI_RISK = "multi_risk"


class Perturbation(StrEnum):
    CLEAN = "clean"
    SOURCE_OUTAGE = "source_outage"
    SYNONYM_SHIFT = "synonym_shift"
    STALE_RECORD = "stale_record"
    CONFLICTING_RECORD = "conflicting_record"
    ROUTE_SWAP = "route_swap"
    PREREQUISITE = "prerequisite"
    DISTRACTOR_DENSE = "distractor_dense"


@dataclass(frozen=True)
class PrescriptionItem:
    name: str
    dose_g: float


@dataclass(frozen=True)
class PublicCase:
    """Information visible to the policy before tool use."""

    patient_id: str
    visit_id: str
    age_band: str
    chief_concern: str
    prescription: Tuple[PrescriptionItem, ...]
    declared_allergies: Tuple[str, ...] = ()
    special_state: str = "unknown"
    concurrent_medication: str = "unknown"
    task: str = (
        "核验本次合成复诊记录的证据充分性，并输出 clear/block/review；"
        "不得把本数据用于真实诊疗。"
    )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source: EvidenceSource
    content: str
    fact_key: str
    fact_value: str
    timestamp: int
    valid_from: int
    valid_until: int
    reliability: float
    required_terms: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()
    stale: bool = False
    contradictory_to: Optional[str] = None
    is_distractor: bool = False

    def active_at(self, visit_time: int) -> bool:
        return self.valid_from <= visit_time <= self.valid_until and not self.stale


@dataclass(frozen=True)
class HiddenWorld:
    """Latent benchmark state. Never serialize this into a policy prompt."""

    world_id: str
    visit_time: int
    risk_family: RiskFamily
    true_decision: Decision
    rule_id: str
    patient_fact: str
    rule_fact: str
    required_fact_keys: Tuple[str, ...]
    evidence_by_source: Mapping[str, Tuple[Evidence, ...]]
    unavailable_sources: Tuple[str, ...] = ()
    prerequisite_by_source: Mapping[str, str] = field(default_factory=dict)
    route_layout: str = "canonical"
    query_style: str = "canonical"
    perturbation: Perturbation = Perturbation.CLEAN
    counterfactual_pair_id: str = ""


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    group_id: str
    split: Split
    seed: int
    composition_key: str
    public: PublicCase
    hidden: HiddenWorld
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def policy_view(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "public": to_jsonable(self.public),
            "remaining_budget": int(self.metadata.get("max_steps", 6)),
            "allowed_actions": [k.value for k in ActionKind],
            "output_contract": {
                "kind": "one allowed action",
                "query": (
                    "required for search/doctor actions; empty otherwise"
                ),
                "evidence_ids": (
                    "required when deciding; empty for all other kinds"
                ),
                "decision": (
                    "clear|block|review when kind=decide; null otherwise"
                ),
                "rationale": "brief evidence-grounded explanation",
            },
        }


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    query: str = ""
    evidence_ids: Tuple[str, ...] = ()
    decision: Optional[Decision] = None
    rationale: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Action":
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown action fields: {unknown}")
        try:
            kind = ActionKind(str(value["kind"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("kind must be a supported action") from exc
        decision_value = value.get("decision")
        decision = None if decision_value in (None, "") else Decision(str(decision_value))
        evidence_ids = value.get("evidence_ids", ())
        if not isinstance(evidence_ids, (list, tuple)):
            raise ValueError("evidence_ids must be an array")
        return cls(
            kind=kind,
            query=str(value.get("query", "") or ""),
            evidence_ids=tuple(str(x) for x in evidence_ids),
            decision=decision,
            rationale=str(value.get("rationale", "") or ""),
        )

    @classmethod
    def from_json(cls, text: str) -> "Action":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("action JSON must be an object")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class Observation:
    status: ObservationStatus
    source: Optional[EvidenceSource]
    message: str
    evidence: Tuple[Evidence, ...] = ()
    failure_type: FailureType = FailureType.NONE
    recoverable: bool = True

    def policy_view(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "source": None if self.source is None else self.source.value,
            "message": self.message,
            "evidence": [
                {
                    "evidence_id": e.evidence_id,
                    "content": e.content,
                    "timestamp": e.timestamp,
                    "reliability": e.reliability,
                    "stale": e.stale,
                    "contradictory_to": e.contradictory_to,
                }
                for e in self.evidence
            ],
            "failure_type": self.failure_type.value,
            "recoverable": self.recoverable,
        }


@dataclass(frozen=True)
class StepRecord:
    index: int
    action: Action
    observation: Observation
    reward: float
    done: bool
    oracle_action: Optional[Action] = None


@dataclass
class EpisodeState:
    case: CaseSpec
    steps: List[StepRecord] = field(default_factory=list)
    acquired_evidence: Dict[str, Evidence] = field(default_factory=dict)
    completed_prerequisites: List[str] = field(default_factory=list)
    done: bool = False
    resolved: bool = False
    escalated: bool = False
    terminal_decision: Optional[Decision] = None

    @property
    def remaining_budget(self) -> int:
        max_steps = int(self.case.metadata.get("max_steps", 6))
        return max(0, max_steps - len(self.steps))

    @property
    def seen_statuses(self) -> Tuple[str, ...]:
        return tuple(s.observation.status.value for s in self.steps)

    def policy_view(self) -> Dict[str, Any]:
        return {
            **self.case.policy_view(),
            "remaining_budget": self.remaining_budget,
            "history": [
                {
                    "step": step.index,
                    "action": to_jsonable(step.action),
                    "observation": step.observation.policy_view(),
                }
                for step in self.steps
            ],
            "acquired_evidence_ids": sorted(self.acquired_evidence),
        }


@dataclass(frozen=True)
class ActionAssessment:
    accepted: bool
    failure_type: FailureType
    status: ObservationStatus
    oracle_aligned: bool
    resolves_subgoal: bool
    risk_severity: float
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairCertificate:
    case_id: str
    state_hash: str
    rejected_action: Action
    oracle_action: Action
    failure_type: FailureType
    minimal_component_sets: Tuple[Tuple[str, ...], ...]
    selected_components: Tuple[str, ...]
    intervened_action: Action
    before: ActionAssessment
    after: ActionAssessment
    verified_minimal: bool
    certificate_hash: str


ACTION_COMPONENTS: Tuple[str, ...] = (
    "kind",
    "query",
    "evidence_ids",
    "decision",
    "rationale",
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        converted = [to_jsonable(v) for v in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def jsonl_dump(rows: Iterable[Any], path: str) -> int:
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def jsonl_iter(path: str) -> Iterator[Dict[str, Any]]:
    """Stream JSON objects from a JSONL file with line-numbered errors."""

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def jsonl_load(path: str) -> List[Dict[str, Any]]:
    return list(jsonl_iter(path))


def evidence_from_mapping(value: Mapping[str, Any]) -> Evidence:
    return Evidence(
        evidence_id=str(value["evidence_id"]),
        source=EvidenceSource(str(value["source"])),
        content=str(value["content"]),
        fact_key=str(value["fact_key"]),
        fact_value=str(value["fact_value"]),
        timestamp=int(value["timestamp"]),
        valid_from=int(value["valid_from"]),
        valid_until=int(value["valid_until"]),
        reliability=float(value["reliability"]),
        required_terms=tuple(str(x) for x in value["required_terms"]),
        aliases=tuple(str(x) for x in value.get("aliases", ())),
        stale=bool(value.get("stale", False)),
        contradictory_to=(
            None
            if value.get("contradictory_to") in (None, "")
            else str(value["contradictory_to"])
        ),
        is_distractor=bool(value.get("is_distractor", False)),
    )


def case_from_mapping(value: Mapping[str, Any]) -> CaseSpec:
    public_value = value["public"]
    hidden_value = value["hidden"]
    public = PublicCase(
        patient_id=str(public_value["patient_id"]),
        visit_id=str(public_value["visit_id"]),
        age_band=str(public_value["age_band"]),
        chief_concern=str(public_value["chief_concern"]),
        prescription=tuple(
            PrescriptionItem(name=str(item["name"]), dose_g=float(item["dose_g"]))
            for item in public_value["prescription"]
        ),
        declared_allergies=tuple(
            str(x) for x in public_value.get("declared_allergies", ())
        ),
        special_state=str(public_value.get("special_state", "unknown")),
        concurrent_medication=str(
            public_value.get("concurrent_medication", "unknown")
        ),
        task=str(public_value.get("task", "")),
    )
    evidence_by_source = {
        str(source): tuple(evidence_from_mapping(item) for item in entries)
        for source, entries in hidden_value["evidence_by_source"].items()
    }
    hidden = HiddenWorld(
        world_id=str(hidden_value["world_id"]),
        visit_time=int(hidden_value["visit_time"]),
        risk_family=RiskFamily(str(hidden_value["risk_family"])),
        true_decision=Decision(str(hidden_value["true_decision"])),
        rule_id=str(hidden_value["rule_id"]),
        patient_fact=str(hidden_value["patient_fact"]),
        rule_fact=str(hidden_value["rule_fact"]),
        required_fact_keys=tuple(str(x) for x in hidden_value["required_fact_keys"]),
        evidence_by_source=evidence_by_source,
        unavailable_sources=tuple(
            str(x) for x in hidden_value.get("unavailable_sources", ())
        ),
        prerequisite_by_source={
            str(k): str(v)
            for k, v in hidden_value.get("prerequisite_by_source", {}).items()
        },
        route_layout=str(hidden_value.get("route_layout", "canonical")),
        query_style=str(hidden_value.get("query_style", "canonical")),
        perturbation=Perturbation(
            str(hidden_value.get("perturbation", Perturbation.CLEAN.value))
        ),
        counterfactual_pair_id=str(
            hidden_value.get("counterfactual_pair_id", "")
        ),
    )
    return CaseSpec(
        case_id=str(value["case_id"]),
        group_id=str(value["group_id"]),
        split=Split(str(value["split"])),
        seed=int(value["seed"]),
        composition_key=str(value["composition_key"]),
        public=public,
        hidden=hidden,
        metadata=dict(value.get("metadata", {})),
    )
