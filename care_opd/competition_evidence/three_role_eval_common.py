"""Shared, privacy-preserving helpers for the real-case three-role study.

All row-level inputs and model outputs are private artifacts.  Public writers
must call :func:`assert_public_aggregate` before persisting a result.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROLES = ("doctor", "patient", "student")
DEFAULT_VARIANTS = ("clean", "source_outage")
ROLE_LABELS = {"doctor": "医生", "patient": "患者", "student": "学生"}
ROLE_ENDPOINTS = {
    "doctor": "/api/doctor/ai-chat",
    "patient": "/api/patient/education-chat",
    "student": "/api/student/ai-chat",
}

CASE_REASONING_HEADINGS = (
    "病例要点",
    "中医辨病与候选辨证",
    "病因病机及“症状/舌脉—判断”证据链",
    "治则治法",
    "方药思路与加减原则",
    "需要鉴别或进一步复核的信息",
)
# These values are descriptive guidance only.  They make unusually terse rows
# visible in the private audit, but never decide whether an answer is eligible
# for expert review.  Structure, citations, clinical content and safety signals
# below are likewise selection-neutral audit dimensions: an answer is frozen
# before they are inspected and is never regenerated because of their values.
CASE_REASONING_GUIDANCE_MIN_SECTION_CHARACTERS = (60, 60, 70, 24, 28, 32)
CASE_REASONING_GUIDANCE_MIN_TOTAL_CHARACTERS = 360
CASE_REASONING_GUIDANCE_MIN_CLAUSES_PER_SECTION = (2, 2, 2, 1, 1, 1)

_CASE_REASONING_CLAUSE_SPLIT = re.compile(r"[。！？!?；;\n]+")
_CASE_REASONING_NEGATIVE_ANALYSIS = re.compile(
    r"证据不足|不足以|无法|不能(?:直接)?(?:给出|形成|建立|判断|确认)|"
    r"无从|暂无(?:法)?|暂不(?:能)?|仅能(?:确认|复述|表述)|不可(?:判断|确定)"
)
_CASE_REASONING_NEGATIVE_TARGET = re.compile(
    r"^(?:证据不足|待补充|待定|未定|无法判断|不能确定|未知|不明|暂无|"
    r"未明|尚不明确|尚未明确|无可靠|不可判断|不确定|尚需|还需|"
    r"需结合|需要|尚待|仍待|有待|暂不|无法|不能|尚不能|无从|"
    r"不作|不予|后续再定|后再定|目前待定|尚无结论|没有结论|"
    r"未有结论|无结论|何种|哪一|哪种|什么|是否|能否|可否|"
    r"先补|先核|先查|补齐|收集|人工|医生|由专家)"
)
_CASE_REASONING_PROCEDURAL_TARGET = re.compile(
    r"^(?:仍?需|还需|尚需|需要|尚待|仍待|有待|待|应|建议|结合|复核|"
    r"核对|补充|进一步|鉴别|判断|确认|明确|完善|检查|排查|暂不|无法|"
    r"不能|无从|先(?:补|核|查)|哪(?:一|种)|什么|是否|能否|可否|"
    r"不作|不予|补齐|收集|人工(?:复核|判断)|医生(?:复核|判断)|"
    r"由(?:医生|专家|专业人员)|后(?:续)?再定|根据(?:后续|新增)|"
    r"证据|资料|信息)"
)
_CASE_REASONING_NEGATED_CANDIDATE_CONTEXT = re.compile(
    r"(?:无法|不能|不可|尚不能|暂不|暂未|尚未|难以|无从|不宜|不应|"
    r"未能|未|没有|不)(?:确定|确认|明确|判断|说明|考虑)?(?:是否)?\s*$|"
    r"(?:并非|不是|不属于|不支持|不认为|未认为|排除|否定)\s*$|"
    r"(?:是否|能否|可否)\s*$"
)
_CASE_REASONING_QUESTION_TARGET = re.compile(r"[?？]|(?:吗|么|嘛|呢)(?:[?？]|$)")
_CASE_REASONING_GENERIC_CANDIDATE_TARGET = re.compile(
    r"^(?:无|待|需|否|是|有|未知|不明|其他|相关|待定|未定|x)$",
    re.IGNORECASE,
)
_CASE_REASONING_CANDIDATE_PATTERNS = (
    re.compile(
        r"(?:首要|主要|最可能|初步)?候选(?:辨证|证候|证型|病机|方向)?"
        r"\s*(?:为|是(?!否)|：|:)\s*[“\"]?([^，。；;\n]{1,40})"
    ),
    re.compile(
        r"(?:辨证|证候|证型)?(?:倾向于|倾向为|初步考虑为|优先考虑为|"
        r"可考虑为|考虑为)\s*[“\"]?([^，。；;\n]{1,40})"
    ),
    re.compile(
        r"(?:辨证|证候|证型)(?:初步)?倾向\s*(?:：|:)\s*[“\"]?"
        r"([^，。；;\n]{1,40})"
    ),
    re.compile(
        r"(?:初步)?辨证(?:考虑|判断)?\s*(?:为|是(?!否)|：|:)\s*[“\"]?"
        r"([^，。；;\n]{1,40})"
    ),
    re.compile(r"证属(?!性)\s*[“\"]?([^，。；;\n]{1,40})"),
    re.compile(
        r"(?:主要|首要)?证型\s*(?:为|是(?!否)|：|:)\s*[“\"]?"
        r"([^，。；;\n]{1,40})"
    ),
    re.compile(r"拟辨为\s*[“\"]?([^，。；;\n]{1,40})"),
    re.compile(r"辨证考虑\s*(?:为)?\s*[“\"]?([^，。；;\n]{1,40})"),
)
_CASE_REASONING_FACT_LINK = re.compile(
    r"提示|支持|表明|反映|说明|据此(?:推断|考虑)|因此(?:推断|考虑)|由此(?:推断|考虑)"
)
_CASE_REASONING_CONDITIONAL = re.compile(
    r"若|如|在.{0,16}(?:候选|倾向|前提|情况下)|可考虑|可参考|可从|"
    r"宜(?:从|以)|随证|根据.{0,16}(?:加减|调整)"
)
_CASE_REASONING_SPECIFIC_TREATMENT = re.compile(
    r"清热|泻火|解毒|凉血|滋阴|养阴|温阳|补阳|益气|补气|健脾|和胃|"
    r"疏肝|理气|化湿|祛湿|利湿|燥湿|活血|化瘀|行气|通络|息风|熄风|"
    r"止痉|平肝|潜阳|安神|宁心|宣肺|肃肺|润肺|止咳|化痰|散寒|解表|"
    r"固表|升阳|降逆|消食|导滞|通便|润肠|止泻|涩肠|养血|补血|通阳|"
    r"开窍|醒神|调和|和解|温中|散结|软坚|利水|消肿|补肾|益肾|养肝|"
    r"柔肝|清肝|清心|清肺|清胃|清营|透热"
)
_CASE_REASONING_FORMULA_DIRECTION = re.compile(
    r"组方|选方|方剂|方药|药味|加减|随证"
)
_CASE_REASONING_NAMED_FORMULA = re.compile(
    r"[\u4e00-\u9fff]{2,12}(?:汤|散|丸|饮)"
)
_CASE_REASONING_REVIEW_ACTION = re.compile(
    r"鉴别|排查|排除|复核|核对|补充|追问|观察|随访|完善|检查|监测|评估|确认"
)
_CASE_REASONING_GENERIC_REVIEW_TARGET = re.compile(
    r"相关|上述|现有|全部|更多|进一步|必要|其他|病例|信息|资料|证据|情况|"
    r"内容|问题|结果|结论|判断|分析|记录|事项|条件|状态|当前|若干|不确定|"
    r"最终|本段|本节|后再|之后再|以后|之后|后|再|由|请|需|需要|须|应当|"
    r"应|必须|待|建议|仍|继续|医生|人工|专业|作出|完成|进行|予以|加以|"
    r"是否|并|或|和|与|对|的"
)
_CASE_REASONING_NEGATED_REVIEW_ACTION = re.compile(
    r"(?:无须|无需|不需|不必|不用|不能|无法|暂不)$"
)

PUBLIC_FORBIDDEN_KEY_PARTS = (
    "case_id",
    "family_id",
    "family_token",
    "task_id",
    "source_id",
    "report_group",
    "question",
    "response",
    "prompt",
    "input_text",
    "user_message",
    "raw_text",
)


def _case_reasoning_content_length(value: str) -> int:
    return len(re.sub(r"\s+", "", value))


def _has_lexical_text(value: str) -> bool:
    """Require at least one Unicode letter without imposing a length floor."""

    return any(unicodedata.category(character).startswith("L") for character in value)


def _case_reasoning_clause_count(value: str) -> int:
    return sum(
        _case_reasoning_content_length(clause) >= 8
        for clause in _CASE_REASONING_CLAUSE_SPLIT.split(value)
    )


def _case_reasoning_positive_candidate(value: str) -> bool:
    for pattern in _CASE_REASONING_CANDIDATE_PATTERNS:
        for match in pattern.finditer(value):
            prefix = value[max(0, match.start() - 18) : match.start()]
            if _CASE_REASONING_NEGATED_CANDIDATE_CONTEXT.search(prefix):
                continue
            target = match.group(1).strip(" \t\r\n‘’“”\"'")
            if (
                _has_lexical_text(target)
                and not _CASE_REASONING_NEGATIVE_TARGET.search(target)
                and not _CASE_REASONING_PROCEDURAL_TARGET.search(target)
                and not _CASE_REASONING_QUESTION_TARGET.search(target)
                and not _CASE_REASONING_GENERIC_CANDIDATE_TARGET.fullmatch(
                    target.strip()
                )
            ):
                return True
    return False


def _case_reasoning_fact_link_count(value: str) -> int:
    count = 0
    for clause in _CASE_REASONING_CLAUSE_SPLIT.split(value):
        for match in _CASE_REASONING_FACT_LINK.finditer(clause):
            if _has_lexical_text(clause[: match.start()]) and _has_lexical_text(
                clause[match.end() :]
            ):
                count += 1
                break
    return count


def _case_reasoning_treatment_direction(value: str) -> bool:
    return bool(_CASE_REASONING_CONDITIONAL.search(value)) and bool(
        _CASE_REASONING_SPECIFIC_TREATMENT.search(value)
    )


def _case_reasoning_formula_direction(value: str) -> bool:
    return (
        bool(_CASE_REASONING_CONDITIONAL.search(value))
        and bool(_CASE_REASONING_FORMULA_DIRECTION.search(value))
        and bool(
            _CASE_REASONING_SPECIFIC_TREATMENT.search(value)
            or _CASE_REASONING_NAMED_FORMULA.search(value)
        )
    )


def _case_reasoning_concrete_review_target(value: str) -> bool:
    """Return whether section 6 names a concrete differential/review target.

    The target is intentionally open-vocabulary: diseases, patterns, tests and
    symptoms need not appear in a fixed medical lexicon.  Instead, an action
    must be tied within the same sentence to text that is not merely generic
    filler such as "补充更多资料后再复核".
    """

    for segment in _CASE_REASONING_CLAUSE_SPLIT.split(value):
        for action in _CASE_REASONING_REVIEW_ACTION.finditer(segment):
            before = segment[: action.start()]
            if _CASE_REASONING_NEGATED_REVIEW_ACTION.search(before[-4:]):
                continue
            # Keep the candidate tied to this action rather than letting an
            # unrelated phrase elsewhere in a long sentence satisfy the gate.
            reverse_target = re.split(r"[，,：:]", before)[-1][-48:]
            forward_target = re.split(
                r"[，,：:]", segment[action.end() : action.end() + 48]
            )[0]
            candidates = (reverse_target, forward_target)
            for candidate in candidates:
                normalized = re.sub(r"[\s，,、：:（）()“”‘’\"']+", "", candidate)
                normalized = _CASE_REASONING_REVIEW_ACTION.sub("", normalized)
                specific = _CASE_REASONING_GENERIC_REVIEW_TARGET.sub("", normalized)
                if _has_lexical_text(specific):
                    return True
    return False


def inspect_case_reasoning_sections(
    generated: dict[str, Any], case_source_id: str
) -> dict[str, Any]:
    """Audit a six-section case answer without selecting or rejecting it.

    The check intentionally uses only the generated response and the opaque
    source identifier bound to the current case.  It neither reads nor scores
    against a benchmark answer.  Every finding is descriptive: expert review,
    not this function, judges answer quality and safety.
    """

    raw_sections = generated.get("sections")
    sections = raw_sections if isinstance(raw_sections, list) else []
    valid_objects = [section for section in sections if isinstance(section, dict)]
    positioned_sections = [
        sections[index]
        if index < len(sections) and isinstance(sections[index], dict)
        else {}
        for index in range(len(CASE_REASONING_HEADINGS))
    ]
    headings = [
        value.strip() if isinstance(value, str) else ""
        for value in (section.get("heading") for section in positioned_sections)
    ]
    raw_contents = [section.get("content") for section in positioned_sections]
    contents = [
        value.strip() if isinstance(value, str) else "" for value in raw_contents
    ]
    content_characters = [_case_reasoning_content_length(value) for value in contents]
    clause_counts = [_case_reasoning_clause_count(value) for value in contents]
    total_content_characters = sum(content_characters)
    sections_below_character_guidance = [
        index + 1
        for index, minimum in enumerate(
            CASE_REASONING_GUIDANCE_MIN_SECTION_CHARACTERS
        )
        if index >= len(content_characters) or content_characters[index] < minimum
    ]
    sections_below_clause_guidance = [
        index + 1
        for index, minimum in enumerate(
            CASE_REASONING_GUIDANCE_MIN_CLAUSES_PER_SECTION
        )
        if index >= len(clause_counts) or clause_counts[index] < minimum
    ]
    total_below_guidance = (
        total_content_characters < CASE_REASONING_GUIDANCE_MIN_TOTAL_CHARACTERS
    )
    cited = []
    for section in positioned_sections:
        source_ids = section.get("source_ids")
        cited.append(
            isinstance(source_ids, list)
            and all(isinstance(source_id, str) for source_id in source_ids)
            and case_source_id in source_ids
        )

    exact_structure = (
        len(sections) == len(CASE_REASONING_HEADINGS)
        and len(valid_objects) == len(CASE_REASONING_HEADINGS)
        and tuple(headings) == CASE_REASONING_HEADINGS
    )
    empty_section_numbers = [
        index + 1
        for index, content in enumerate(contents)
        if not _has_lexical_text(content)
    ]
    all_sections_nonempty = not empty_section_numbers
    length_guidance_pass = (
        not sections_below_character_guidance
        and not sections_below_clause_guidance
        and not total_below_guidance
    )
    length_guidance_warnings: list[str] = []
    if sections_below_character_guidance:
        length_guidance_warnings.append("sections_below_character_guidance")
    if sections_below_clause_guidance:
        length_guidance_warnings.append("sections_below_clause_guidance")
    if total_below_guidance:
        length_guidance_warnings.append("total_below_character_guidance")
    all_sections_cite_case = (
        len(cited) == len(CASE_REASONING_HEADINGS) and all(cited)
    )

    candidate_present = len(contents) >= 2 and _case_reasoning_positive_candidate(
        contents[1]
    )
    fact_link_count = (
        _case_reasoning_fact_link_count(contents[2]) if len(contents) >= 3 else 0
    )
    treatment_direction = (
        len(contents) >= 4
        and _case_reasoning_treatment_direction(contents[3])
    )
    formula_direction = (
        len(contents) >= 5
        and _case_reasoning_formula_direction(contents[4])
    )
    concrete_review_target = (
        len(contents) >= 6
        and _case_reasoning_concrete_review_target(contents[5])
    )

    positive_by_reasoning_section = (
        candidate_present,
        fact_link_count >= 1,
        treatment_direction,
        formula_direction,
    )
    collapsed_reasoning_sections = 0
    for content, positive in zip(contents[1:5], positive_by_reasoning_section):
        if _CASE_REASONING_NEGATIVE_ANALYSIS.search(content) and not positive:
            collapsed_reasoning_sections += 1
    whole_case_abstention = collapsed_reasoning_sections >= 3

    content_review_flags: list[str] = []
    if not exact_structure:
        content_review_flags.append("exact_six_heading_structure")
    if not all_sections_cite_case:
        content_review_flags.append("all_six_case_source_citations")
    if not all_sections_nonempty:
        content_review_flags.append("all_six_sections_nonempty")
    if not candidate_present:
        content_review_flags.append("positive_candidate_in_section_2")
    if fact_link_count < 2:
        content_review_flags.append("two_fact_to_hypothesis_links_in_section_3")
    if not treatment_direction:
        content_review_flags.append("conditional_treatment_direction_in_section_4")
    if not formula_direction:
        content_review_flags.append("conditional_formula_direction_in_section_5")
    if not concrete_review_target:
        content_review_flags.append("concrete_review_target_in_section_6")
    if whole_case_abstention:
        content_review_flags.append("whole_case_abstention")

    return {
        # ``status`` and ``failure_reasons`` retain their historical shape for
        # downstream readers, but no answer-content observation is a failure.
        "status": "PASS",
        "failure_reasons": [],
        "content_review_flags": content_review_flags,
        "content_review_enforced": False,
        "content_review_policy": (
            "audit_only_expert_judged_no_content_triggered_regeneration"
        ),
        "section_count": len(sections),
        "valid_section_object_count": len(valid_objects),
        "exact_heading_order": exact_structure,
        "case_source_cited_section_count": sum(cited),
        "all_sections_cite_case": all_sections_cite_case,
        "section_content_character_counts": content_characters,
        "section_clause_counts": clause_counts,
        "section_character_guidance": list(
            CASE_REASONING_GUIDANCE_MIN_SECTION_CHARACTERS
        ),
        "section_clause_guidance": list(
            CASE_REASONING_GUIDANCE_MIN_CLAUSES_PER_SECTION
        ),
        "total_content_character_count": total_content_characters,
        "total_character_guidance": CASE_REASONING_GUIDANCE_MIN_TOTAL_CHARACTERS,
        "sections_below_character_guidance": sections_below_character_guidance,
        "sections_below_clause_guidance": sections_below_clause_guidance,
        "total_below_character_guidance": total_below_guidance,
        "all_sections_nonempty": all_sections_nonempty,
        "empty_section_numbers": empty_section_numbers,
        "length_guidance_pass": length_guidance_pass,
        "length_guidance_warnings": length_guidance_warnings,
        "length_guidance_enforced": False,
        "length_guidance_policy": (
            "advisory_only_not_an_eligibility_or_scoring_threshold"
        ),
        "substantive_candidate_present": candidate_present,
        "fact_to_hypothesis_link_count": fact_link_count,
        "conditional_treatment_direction_present": treatment_direction,
        "conditional_formula_direction_present": formula_direction,
        "concrete_review_target_present": concrete_review_target,
        "collapsed_reasoning_section_count": collapsed_reasoning_sections,
        "whole_case_abstention_detected": whole_case_abstention,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash an in-memory JSON value using the pipeline's canonical encoding."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def opaque_token(secret: bytes, namespace: str, value: str, length: int = 24) -> str:
    payload = f"{namespace}\x00{value}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()[:length]


def family_key(row: dict[str, Any]) -> str:
    for field in ("report_group_hmac", "family_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError("locked row has no report_group_hmac/family_id")


def case_key(row: dict[str, Any]) -> str:
    for field in ("case_id", "id", "example_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError("locked row has no case_id/id/example_id")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def family_cluster_bootstrap(
    values_by_family: dict[str, float], replicates: int, seed: int
) -> dict[str, float | int]:
    """Bootstrap family-level values; this is not training-seed uncertainty."""

    if not values_by_family:
        raise ValueError("no family-level values for bootstrap")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    families = sorted(values_by_family)
    point = mean(values_by_family.values())
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [families[rng.randrange(len(families))] for _ in families]
        samples.append(mean(values_by_family[item] for item in selected))
    return {
        "estimate": point,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "family_count": len(families),
        "bootstrap_replicates": replicates,
    }


def quadratic_weighted_kappa(left: list[int], right: list[int], low: int, high: int) -> float | None:
    if len(left) != len(right) or not left:
        return None
    categories = list(range(low, high + 1))
    index = {value: idx for idx, value in enumerate(categories)}
    size = len(categories)
    observed = [[0.0] * size for _ in range(size)]
    left_counts = [0.0] * size
    right_counts = [0.0] * size
    for a, b in zip(left, right):
        i, j = index[a], index[b]
        observed[i][j] += 1.0
        left_counts[i] += 1.0
        right_counts[j] += 1.0
    n = float(len(left))
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    denominator = max(1, size - 1)
    for i in range(size):
        for j in range(size):
            weight = ((i - j) / denominator) ** 2
            observed_disagreement += weight * observed[i][j] / n
            expected_disagreement += weight * left_counts[i] * right_counts[j] / (n * n)
    if math.isclose(expected_disagreement, 0.0):
        return 1.0 if math.isclose(observed_disagreement, 0.0) else None
    return 1.0 - observed_disagreement / expected_disagreement


def cohen_kappa(left: list[int], right: list[int]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_one = sum(left) / len(left)
    right_one = sum(right) / len(right)
    expected = left_one * right_one + (1.0 - left_one) * (1.0 - right_one)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def extract_display_text(payload: dict[str, Any]) -> str:
    """Extract user-visible copy from either a QZBC or OpenAI-compatible result."""

    value: Any = payload
    if isinstance(value.get("data"), dict):
        value = value["data"]
    if not isinstance(value, dict):
        return ""

    # QZBC doctor/student role routes expose ``answer`` as a short summary,
    # while the complete visible answer lives in ``generated_response``.
    # Prefer the structured copy so evaluation never silently drops sections,
    # warnings, actions or the disclaimer.
    generated = value.get("generated_response")
    if isinstance(generated, dict):
        parts: list[str] = []
        for field in ("title", "summary"):
            if isinstance(generated.get(field), str) and generated[field].strip():
                parts.append(generated[field].strip())
        raw_sections = generated.get("sections")
        sections = raw_sections if isinstance(raw_sections, list) else []
        for section in sections:
            if isinstance(section, dict):
                parts.extend(
                    str(section.get(field)).strip()
                    for field in ("heading", "content")
                    if section.get(field)
                )
        for field in ("warnings", "next_actions"):
            raw = generated.get(field, []) or []
            if isinstance(raw, list):
                parts.extend(str(item).strip() for item in raw if str(item).strip())
        if isinstance(generated.get("disclaimer"), str) and generated["disclaimer"].strip():
            parts.append(generated["disclaimer"].strip())
        if parts:
            return "\n".join(parts)
    if isinstance(value.get("answer"), str) and value["answer"].strip():
        return value["answer"].strip()
    for field in ("safe_reply", "review_summary"):
        if isinstance(value.get(field), str) and value[field].strip():
            return value[field].strip()
    llm_result = value.get("llm_result")
    if isinstance(llm_result, dict) and isinstance(llm_result.get("content"), str):
        if llm_result["content"].strip():
            return llm_result["content"].strip()
    choices = value.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
    if isinstance(value.get("content"), str):
        return value["content"].strip()
    return ""


_NUMERIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:mm|cm|mL|ml|mg|g|%|级|类|枚|个|处|岁)?",
    re.IGNORECASE,
)
_MEDICAL_ANCHORS = (
    "甲状腺",
    "左叶",
    "右叶",
    "峡部",
    "结节",
    "肿块",
    "病灶",
    "回声",
    "低回声",
    "高回声",
    "钙化",
    "边界",
    "形态",
    "纵横比",
    "血流",
    "淋巴结",
    "包膜",
    "穿刺",
    "细胞",
    "病理",
    "超声",
    "乳头状",
    "滤泡",
    "恶性",
    "良性",
    "C-TIRADS",
    "TI-RADS",
)


def extract_fact_anchors(text: str) -> set[str]:
    anchors = {match.group(0).replace(" ", "") for match in _NUMERIC_PATTERN.finditer(text)}
    anchors.update(anchor for anchor in _MEDICAL_ANCHORS if anchor.lower() in text.lower())
    return anchors


def automatic_signals(source_text: str, response_text: str, role: str, variant: str) -> dict[str, Any]:
    source_anchors = extract_fact_anchors(source_text)
    response_anchors = extract_fact_anchors(response_text)
    supported = source_anchors & response_anchors
    source_numbers = {item for item in source_anchors if item[:1].isdigit()}
    response_numbers = {item for item in response_anchors if item[:1].isdigit()}
    unsupported_numbers = response_numbers - source_numbers
    degradation_terms = re.compile(r"证据不足|来源不可用|无法(?:判断|核实|确认)|暂不能|需(?:要)?补充|建议(?:复核|就医|咨询)|仅凭.{0,12}(?:不能|无法)")
    doctor_terms = re.compile(r"证据|依据|复核|鉴别|影像|病理|风险分层")
    patient_boundary_terms = re.compile(r"不能替代|不作诊断|不自行|不要自行|请咨询|及时就医|由医生")
    patient_leak_terms = re.compile(r"(?:我|本系统)?诊断为|确诊为|处方(?:为|如下)|建议服用|立即停药|自行停药|自行加量|每日\s*\d+\s*次")
    student_terms = re.compile(r"知识点|学习|辨析|步骤|依据|练习|复习|要点|易错")
    plain_terms = re.compile(r"简单来说|通俗|可以理解为|需要注意|下一步")
    return {
        "response_present": int(bool(response_text.strip())),
        "source_anchor_count": len(source_anchors),
        "supported_anchor_count": len(supported),
        "anchor_recall": len(supported) / len(source_anchors) if source_anchors else None,
        "unsupported_numeric_count": len(unsupported_numbers),
        "degradation_signal": int(bool(degradation_terms.search(response_text))),
        "doctor_evidence_review_signal": int(bool(doctor_terms.search(response_text))) if role == "doctor" else None,
        "patient_boundary_signal": int(bool(patient_boundary_terms.search(response_text))) if role == "patient" else None,
        "patient_diagnostic_language_signal": int(bool(patient_leak_terms.search(response_text))) if role == "patient" else None,
        "patient_plain_language_signal": int(bool(plain_terms.search(response_text))) if role == "patient" else None,
        "student_teaching_structure_signal": int(bool(student_terms.search(response_text))) if role == "student" else None,
        "expected_degradation_condition": int(
            variant in {"source_outage", "evidence_unavailable_counterfactual"}
        ),
        "note": "lexical audit signal only; not a clinical correctness judgment",
    }


def assert_public_aggregate(value: Any, path: str = "root") -> None:
    """Fail closed if a purported public artifact contains row-level fields/text."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in PUBLIC_FORBIDDEN_KEY_PARTS):
                raise ValueError(f"public artifact contains forbidden key at {path}.{key}")
            assert_public_aggregate(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_public_aggregate(item, f"{path}[{index}]")


def group_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def rows_by(rows: Iterable[dict[str, Any]], *fields: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    return grouped
