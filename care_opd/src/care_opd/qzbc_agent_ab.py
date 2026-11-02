"""Paired smoke test through two unchanged qzbc Agent backends.

This module deliberately requires two backend processes.  qzbc caches its
settings at process start, so a valid end-to-end comparison cannot switch
``OLLAMA_MODEL`` inside one running Agent.  The two processes must use the
same qzbc source and equivalent freshly seeded data, while keeping their
SQLite/Graphiti/Mem0 writable stores isolated.

The scientific model-quality result is produced by :mod:`qzbc_native_eval`.
This runner is a deployment integration check: it proves that requests pass
through qzbc's real graphs and that each arm actually used the requested
Ollama model without a template fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "doctor_evidence_boundary",
        "role": "doctor",
        "path": "/api/doctor/ai-chat",
        "message": (
            "请结合系统当前可见证据，总结处方复核时最需要确认的风险；"
            "证据不足或相互冲突之处必须明确标注，不要补写患者事实。"
        ),
    },
    {
        "id": "doctor_followup_checklist",
        "role": "doctor",
        "path": "/api/doctor/ai-chat",
        "message": (
            "给出复诊时核查症状变化、用药反应和风险信息的证据化清单，"
            "每条结论都要受当前 Evidence Packet 约束。"
        ),
    },
    {
        "id": "student_differentiation",
        "role": "student",
        "path": "/api/student/ai-chat",
        "message": (
            "解释辨证分析中如何区分气虚与血虚，并说明材料不足时应怎样"
            "保留不确定性；不要把教学知识写成对具体患者的诊断。"
        ),
    },
    {
        "id": "student_risk_reasoning",
        "role": "student",
        "path": "/api/student/ai-chat",
        "message": (
            "作为教学练习，分析含桃仁方剂时应先核对哪些风险证据，"
            "并指出检索失败后不能做出的结论。"
        ),
    },
    {
        "id": "patient_adverse_event",
        "role": "patient",
        "path": "/api/patient/education-chat",
        "message": (
            "我正在按医嘱服用中药，如果出现不适，应该怎样安全记录并"
            "联系医生？请只做一般科普，不要替我调整药量。"
        ),
    },
    {
        "id": "patient_no_self_adjustment",
        "role": "patient",
        "path": "/api/patient/education-chat",
        "message": "请用通俗语言解释为什么不能自行增加、减少或停用中药剂量。",
    },
)

ROLE_USERS = {
    "doctor": "doctor_demo",
    "student": "student_demo",
    "patient": "patient_demo",
}

GENERATED_FIELDS = (
    "title",
    "summary",
    "sections",
    "warnings",
    "next_actions",
    "disclaimer",
)

COMMON_GRAPH_NODES = {
    "AuthNode",
    "RouterNode",
    "ConversationContextNode",
    "MemoryRouterNode",
    "MCPToolNode",
    "EvidencePacketNode",
    "LLMGenerateNode",
    "SafetyPostNode",
    "ResponseNode",
    "TraceNode",
}


@dataclass(frozen=True)
class Arm:
    name: str
    url: str
    expected_model: str


class QZBCHTTPError(RuntimeError):
    pass


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Mapping[str, Any]] = None,
    token: Optional[str] = None,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise QZBCHTTPError(
            f"{method} {base_url}{path} returned HTTP {exc.code}: {detail[:1000]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise QZBCHTTPError(f"cannot reach {base_url}{path}: {exc.reason}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QZBCHTTPError(
            f"{method} {base_url}{path} returned non-JSON: {raw[:1000]}"
        ) from exc
    if not isinstance(value, dict):
        raise QZBCHTTPError(f"{method} {base_url}{path} returned a non-object")
    return value


def _unwrap(response: Mapping[str, Any], *, operation: str) -> Mapping[str, Any]:
    if response.get("success") is not True:
        raise QZBCHTTPError(f"{operation} failed: {json.dumps(response, ensure_ascii=False)}")
    data = response.get("data")
    if not isinstance(data, Mapping):
        raise QZBCHTTPError(f"{operation} returned no data object")
    return data


def _login(arm: Arm, role: str, password: str, timeout: float) -> str:
    response = _request_json(
        arm.url,
        "/api/auth/login",
        method="POST",
        payload={
            "username": ROLE_USERS[role],
            "password": password,
            "entry_role": role,
        },
        timeout=timeout,
    )
    data = _unwrap(response, operation=f"{arm.name}/{role} login")
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise QZBCHTTPError(f"{arm.name}/{role} login returned no access token")
    return token


def _reset_demo(arm: Arm, password: str, timeout: float) -> None:
    token = _login(arm, "doctor", password, timeout)
    response = _request_json(
        arm.url,
        "/api/demo/reset",
        method="POST",
        payload={"reset_scope": "demo_only", "reseed": True},
        token=token,
        timeout=timeout,
    )
    _unwrap(response, operation=f"{arm.name} demo reset")


def _health(arm: Arm, timeout: float) -> Mapping[str, Any]:
    value = _request_json(arm.url, "/health", timeout=timeout)
    if value.get("status") != "ok":
        raise QZBCHTTPError(f"{arm.name} health check failed: {value}")
    return value


def _generated_response_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or any(field not in value for field in GENERATED_FIELDS):
        return False
    if not all(isinstance(value[field], str) for field in ("title", "summary", "disclaimer")):
        return False
    if not all(isinstance(value[field], list) for field in ("sections", "warnings", "next_actions")):
        return False
    for section in value["sections"]:
        if not isinstance(section, Mapping):
            return False
        if not isinstance(section.get("heading"), str) or not isinstance(
            section.get("content"), str
        ):
            return False
        source_ids = section.get("source_ids")
        if not isinstance(source_ids, list) or not all(
            isinstance(item, str) for item in source_ids
        ):
            return False
    return all(isinstance(item, str) for field in ("warnings", "next_actions") for item in value[field])


def _source_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source_id", "source"} and isinstance(item, (str, int)):
                found.add(str(item))
            else:
                found.update(_source_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_source_ids(item))
    return found


def _emitted_source_ids(generated: Any) -> set[str]:
    if not isinstance(generated, Mapping):
        return set()
    emitted: set[str] = set()
    sections = generated.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, Mapping) and isinstance(section.get("source_ids"), list):
                emitted.update(str(item) for item in section["source_ids"])
    return emitted


def _run_case(
    arm: Arm,
    case: Mapping[str, str],
    token: str,
    *,
    expected_provider: str,
    timeout: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    response = _request_json(
        arm.url,
        case["path"],
        method="POST",
        payload={"message": case["message"]},
        token=token,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    data = dict(_unwrap(response, operation=f"{arm.name}/{case['id']}"))
    llm_result = data.get("llm_result")
    llm = llm_result if isinstance(llm_result, Mapping) else {}
    generated = data.get("generated_response")
    path = data.get("graph_node_path")
    graph_path = [str(item) for item in path] if isinstance(path, list) else []
    required_nodes = set(COMMON_GRAPH_NODES)
    if case["role"] == "patient":
        required_nodes.add("SafetyPreNode")
    observed_model = str(llm.get("model") or "")
    observed_provider = str(llm.get("provider") or "")
    raw_metadata = llm.get("metadata")
    llm_metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    allowed_sources = _source_ids(data.get("evidence_packet"))
    emitted_sources = _emitted_source_ids(generated)
    checks = {
        "full_graph_path": required_nodes.issubset(set(graph_path)),
        "generated_response_schema": _generated_response_valid(generated),
        "expected_model": observed_model == arm.expected_model,
        "expected_provider": observed_provider == expected_provider,
        "model_metadata_present": isinstance(raw_metadata, Mapping),
        "model_no_fallback": not bool(llm_metadata.get("fallback_used")),
        "model_not_degraded": not bool(llm_metadata.get("degraded")),
        # Tool/retrieval fallback is kept separate. A freshly isolated
        # Graphiti store can legitimately be empty while the LLM deployment is
        # healthy; folding that into model provenance would create a false
        # failure.
        "agent_no_fallback": not bool(data.get("fallback_used")),
        "agent_not_degraded": not bool(data.get("degraded")),
        "citation_ids_grounded": emitted_sources.issubset(allowed_sources),
    }
    # Citation grounding is reported separately because qzbc may intentionally
    # expose canonical policy IDs outside retrieved_evidence.  The deployment
    # integrity gate concerns the actual model, graph path, schema and fallback.
    integrity_keys = (
        "full_graph_path",
        "generated_response_schema",
        "expected_model",
        "expected_provider",
        "model_metadata_present",
        "model_no_fallback",
        "model_not_degraded",
    )
    return {
        "case_id": case["id"],
        "role": case["role"],
        "path": case["path"],
        "message": case["message"],
        "arm": arm.name,
        "expected_model": arm.expected_model,
        "observed_model": observed_model,
        "observed_provider": observed_provider,
        "latency_seconds": round(elapsed, 4),
        "integrity_pass": all(checks[key] for key in integrity_keys),
        "checks": checks,
        "emitted_source_ids": sorted(emitted_sources),
        "available_source_ids": sorted(allowed_sources),
        "response": data,
    }


def _read_cases(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return [dict(item) for item in DEFAULT_CASES]
    cases: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: case must be an object")
            missing = [
                key for key in ("id", "role", "path", "message") if not value.get(key)
            ]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {missing}")
            if value["role"] not in ROLE_USERS:
                raise ValueError(f"{path}:{line_number}: unsupported role {value['role']!r}")
            cases.append({key: str(value[key]) for key in ("id", "role", "path", "message")})
    if not cases:
        raise ValueError(f"{path} has no cases")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    return cases


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# qzbc full-Agent paired deployment smoke test",
        "",
        "| Case | Role | Base integrity | Trained integrity | Base model | Trained model | Base latency | Trained latency |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    pairs = report["pairs"]
    for pair in pairs:
        base = pair["base"]
        trained = pair["trained"]
        lines.append(
            "| {case} | {role} | {bp} | {tp} | {bm} | {tm} | {bl:.2f}s | {tl:.2f}s |".format(
                case=pair["case_id"],
                role=pair["role"],
                bp="PASS" if base["integrity_pass"] else "FAIL",
                tp="PASS" if trained["integrity_pass"] else "FAIL",
                bm=base["observed_model"] or "missing",
                tm=trained["observed_model"] or "missing",
                bl=base["latency_seconds"],
                tl=trained["latency_seconds"],
            )
        )
    summary = report["summary"]
    lines.extend(
        [
            "",
            "## Integrity gate",
            "",
            f"- Verdict: **{summary['verdict']}**",
            f"- Base: `{summary['base_integrity_pass']}/{summary['cases']}`",
            f"- Trained: `{summary['trained_integrity_pass']}/{summary['cases']}`",
            f"- Paired pass: `{summary['paired_integrity_pass']}/{summary['cases']}`",
            (
                "- Agent non-degraded (base/trained): "
                f"`{summary['base_agent_non_degraded']}/"
                f"{summary['trained_agent_non_degraded']}`"
            ),
            "",
            "This check executes qzbc's real Agent Graph and verifies model provenance.",
            "It is a deployment smoke test, not a clinical-effect or standalone model-quality result.",
            "",
        ]
    )
    return "\n".join(lines)


def compare(
    base: Arm,
    trained: Arm,
    *,
    cases: Sequence[Mapping[str, str]],
    password: str,
    expected_provider: str,
    timeout: float,
    reset_demo: bool,
) -> Dict[str, Any]:
    if base.url.rstrip("/") == trained.url.rstrip("/"):
        raise ValueError(
            "full-Agent A/B requires two backend URLs because qzbc caches OLLAMA_MODEL"
        )
    health = {
        base.name: dict(_health(base, timeout)),
        trained.name: dict(_health(trained, timeout)),
    }
    if reset_demo:
        _reset_demo(base, password, timeout)
        _reset_demo(trained, password, timeout)
    tokens = {
        arm.name: {
            role: _login(arm, role, password, timeout) for role in ROLE_USERS
        }
        for arm in (base, trained)
    }
    pairs: List[Dict[str, Any]] = []
    for case in cases:
        base_result = _run_case(
            base,
            case,
            tokens[base.name][case["role"]],
            expected_provider=expected_provider,
            timeout=timeout,
        )
        trained_result = _run_case(
            trained,
            case,
            tokens[trained.name][case["role"]],
            expected_provider=expected_provider,
            timeout=timeout,
        )
        pairs.append(
            {
                "case_id": case["id"],
                "role": case["role"],
                "base": base_result,
                "trained": trained_result,
            }
        )
    base_passes = sum(bool(pair["base"]["integrity_pass"]) for pair in pairs)
    trained_passes = sum(bool(pair["trained"]["integrity_pass"]) for pair in pairs)
    paired_passes = sum(
        bool(pair["base"]["integrity_pass"] and pair["trained"]["integrity_pass"])
        for pair in pairs
    )
    base_agent_clean = sum(
        bool(
            pair["base"]["checks"]["agent_no_fallback"]
            and pair["base"]["checks"]["agent_not_degraded"]
        )
        for pair in pairs
    )
    trained_agent_clean = sum(
        bool(
            pair["trained"]["checks"]["agent_no_fallback"]
            and pair["trained"]["checks"]["agent_not_degraded"]
        )
        for pair in pairs
    )
    all_pass = paired_passes == len(pairs)
    return {
        "schema_version": "care-opd-qzbc-agent-ab-v1",
        "comparison_scope": (
            "unchanged qzbc RoleAIChatGraph/PatientEducationGraph deployment smoke"
        ),
        "base": {"url": base.url, "expected_model": base.expected_model},
        "trained": {"url": trained.url, "expected_model": trained.expected_model},
        "expected_provider": expected_provider,
        "reset_demo": reset_demo,
        "health": health,
        "pairs": pairs,
        "summary": {
            "verdict": "PASS" if all_pass else "FAIL",
            "cases": len(pairs),
            "base_integrity_pass": base_passes,
            "trained_integrity_pass": trained_passes,
            "paired_integrity_pass": paired_passes,
            "base_agent_non_degraded": base_agent_clean,
            "trained_agent_non_degraded": trained_agent_clean,
            "base_mean_latency_seconds": round(
                statistics.fmean(pair["base"]["latency_seconds"] for pair in pairs), 4
            ),
            "trained_mean_latency_seconds": round(
                statistics.fmean(pair["trained"]["latency_seconds"] for pair in pairs),
                4,
            ),
        },
        "limitations": [
            "This is not the held-out model-quality evaluation.",
            "The built-in demo users and prompts are synthetic and are not clinical advice.",
            "Both backends must use equivalent initial data and isolated writable stores.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired requests through two unchanged qzbc Agent backends"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--trained-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--password", default=os.getenv("QZBC_PASSWORD", "password123"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--reset-demo", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    report = compare(
        Arm("base", args.base_url.rstrip("/"), args.base_model),
        Arm("trained", args.trained_url.rstrip("/"), args.trained_model),
        cases=_read_cases(args.cases),
        password=args.password,
        expected_provider=args.provider,
        timeout=args.timeout,
        reset_demo=args.reset_demo,
    )
    _atomic_json(args.output, report)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(f"Full outputs: {args.output}")
    return 0 if report["summary"]["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
