#!/usr/bin/env python3
"""Run source-grounded multi-agent annotation on private thyroid reports.

The input queue is produced by :mod:`prepare_external_test`.  Real report text
is sent only to an explicitly configured Ollama endpoint.  Non-loopback hosts
are rejected unless the operator acknowledges that transfer with
``--allow-nonloopback``.

Three roles are used:

* blind extractor: sees source text and the allowed field paths, not spreadsheet
  candidate values;
* candidate auditor: sees the source text and spreadsheet candidates;
* adjudicator: sees both outputs and may accept only source-grounded fields.

The deterministic merge, not the language model, decides whether a field is
``exact_verified``.  Any disagreement or unlocatable quote is routed to human
review.  This script is an annotation aid, not a clinical decision system.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "care_opd.real_thyroid_agent_annotation.v2"
OLLAMA_GENERATION_OPTIONS = {
    "think": False,
    "temperature": 0,
    "num_ctx": 16384,
    "num_predict": 4096,
}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "entity_id",
        "field_path",
        "value",
        "unit",
        "support_status",
        "evidence_quote",
        "reason_code",
    ],
    "properties": {
        "entity_id": {"type": "string"},
        "field_path": {"type": "string"},
        "value": {"type": "string"},
        "unit": {"type": "string"},
        "support_status": {
            "enum": [
                "exact_supported",
                "alias_supported",
                "contradicted",
                "not_found",
                "entity_ambiguous",
                "unresolved",
            ]
        },
        "evidence_quote": {"type": "string"},
        "reason_code": {"type": "string"},
    },
}

ANNOTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fields", "case_flags"],
    "properties": {
        "fields": {"type": "array", "items": FIELD_SCHEMA},
        "case_flags": {"type": "array", "items": {"type": "string"}},
    },
}

ADJUDICATED_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "entity_id",
        "field_path",
        "decision",
        "selected_quote",
        "normalized_value",
        "unit",
        "reason_code",
    ],
    "properties": {
        "entity_id": {"type": "string"},
        "field_path": {"type": "string"},
        "decision": {
            "enum": [
                "accept_sheet_candidate",
                "accept_extractor_candidate",
                "unresolved",
            ]
        },
        "selected_quote": {"type": "string"},
        "normalized_value": {"type": "string"},
        "unit": {"type": "string"},
        "reason_code": {"type": "string"},
    },
}

ADJUDICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fields", "case_flags"],
    "properties": {
        "fields": {"type": "array", "items": ADJUDICATED_FIELD_SCHEMA},
        "case_flags": {"type": "array", "items": {"type": "string"}},
    },
}

LOCATOR_SYSTEM = """你是离线医学文本证据标注器，不提供诊断，不使用医学常识补全。
你看不到表格候选值。只抽取 source_text 中直接出现、可由连续原文短语支持的允许字段。
无法定位、病灶归属不唯一、需要推理或原文没有报告时，省略该字段或标记 unresolved。
禁止补充患者属性、日期、跨报告关联、TI-RADS、良恶性、治疗、剂量或预后。
evidence_quote 必须逐字复制 source_text 中的最短充分连续片段；不要输出解释性散文。"""

AUDITOR_SYSTEM = """你是离线结构化候选审计器。逐项比较 spreadsheet candidate 与 source_text。
只有原文可直接支持时才输出 exact_supported；仅存在无歧义同义表达时输出 alias_supported。
空候选不是阴性。不得依靠医学常识、新增诊断、合并冰冻/常规/补充诊断或建立跨报告关系。
每个 supported 项必须给逐字来自 source_text 的最短 evidence_quote；不确定就输出 unresolved。
必须返回输入中的每一个 candidate field，且不要输出解释性散文。"""

ADJUDICATOR_SYSTEM = """你是离线分歧裁决器，不是临床医生，也无权生成新金标。
只比较 blind extractor、candidate auditor、spreadsheet candidate 与 source_text。
只有字段、实体、值和逐字证据一致时才 accept_sheet_candidate。
原表为空或需修正原表的 agent 新发现最多 accept_extractor_candidate，不能进入严格主金标。
任何冲突、歧义、推理或无法逐字定位都必须 unresolved。不要输出解释性散文。"""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any], *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        if private:
            try:
                path.chmod(0o600)
            except OSError:
                pass
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
        if private:
            try:
                path.chmod(0o600)
            except OSError:
                pass
    finally:
        temporary.unlink(missing_ok=True)


def _validate_endpoint(base_url: str, allow_nonloopback: bool) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--base-url must be an http(s) URL")
    if parsed.hostname.lower() not in LOOPBACK_HOSTS and not allow_nonloopback:
        raise PermissionError(
            "Real report transfer to a non-loopback endpoint is blocked. "
            "Use an approved private endpoint and pass --allow-nonloopback explicitly."
        )


def _ollama_chat(
    *,
    base_url: str,
    model: str,
    system: str,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    timeout: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ],
        "stream": False,
        "think": OLLAMA_GENERATION_OPTIONS["think"],
        "keep_alive": "30m",
        "format": dict(schema),
        "options": {
            "temperature": OLLAMA_GENERATION_OPTIONS["temperature"],
            "seed": seed,
            "num_ctx": OLLAMA_GENERATION_OPTIONS["num_ctx"],
            "num_predict": OLLAMA_GENERATION_OPTIONS["num_predict"],
        },
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {detail[:500]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama request failed: {error.reason}") from error
    message = response_body.get("message") if isinstance(response_body, Mapping) else None
    content = str(message.get("content") or "") if isinstance(message, Mapping) else ""
    if not content:
        raise RuntimeError("Ollama returned empty content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("Agent output must be a JSON object")
    usage = {
        "input_tokens": int(response_body.get("prompt_eval_count") or 0),
        "output_tokens": int(response_body.get("eval_count") or 0),
    }
    return parsed, usage


def _clean_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return "".join(character for character in normalized if character.isalnum())


def _field_map(value: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in value.get("fields", []) if isinstance(value.get("fields"), list) else []:
        if not isinstance(item, Mapping):
            continue
        key = (str(item.get("entity_id") or ""), str(item.get("field_path") or ""))
        if all(key) and key not in result:
            result[key] = item
    return result


def _quote_span(source: str, quote: str) -> tuple[int, int] | None:
    if not quote:
        return None
    first = source.find(quote)
    if first < 0 or source.find(quote, first + 1) >= 0:
        return None
    return first, first + len(quote)


def _candidate_map(task: Mapping[str, Any]) -> dict[tuple[str, str], str | None]:
    result: dict[tuple[str, str], str | None] = {}
    for item in task.get("candidate_fields", []):
        if not isinstance(item, Mapping):
            continue
        key = (str(item.get("entity_id") or ""), str(item.get("field_path") or ""))
        candidate = item.get("raw_value")
        result[key] = None if candidate is None else str(candidate)
    return result


def _merge_case(
    task: Mapping[str, Any],
    locator: Mapping[str, Any],
    auditor: Mapping[str, Any],
    adjudicator: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = str(task["source_text_private"])
    candidates = _candidate_map(task)
    locator_fields = _field_map(locator)
    auditor_fields = _field_map(auditor)
    adjudicator_fields = _field_map(adjudicator)
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    all_keys = sorted(
        set(candidates) | set(locator_fields) | set(auditor_fields) | set(adjudicator_fields)
    )
    for key in all_keys:
        candidate = candidates.get(key)
        blind = locator_fields.get(key, {})
        audit = auditor_fields.get(key, {})
        judge = adjudicator_fields.get(key, {})
        quote = str(judge.get("selected_quote") or audit.get("evidence_quote") or "")
        span = _quote_span(source, quote)
        exact_audit = audit.get("support_status") == "exact_supported"
        blind_supported = blind.get("support_status") in {
            "exact_supported",
            "alias_supported",
        }
        blind_value = str(blind.get("value") or "")
        audit_value = str(audit.get("value") or "")
        candidate_key = _clean_key(candidate or "")
        exact_value_agreement = (
            bool(candidate_key)
            and candidate_key == _clean_key(blind_value)
            and candidate_key == _clean_key(audit_value)
        )
        strict = (
            candidate is not None
            and exact_audit
            and blind_supported
            and exact_value_agreement
            and judge.get("decision") == "accept_sheet_candidate"
            and span is not None
        )
        record = {
            "case_id": task["case_id"],
            "report_group_hmac": task["report_group_hmac"],
            "modality": task["modality"],
            "entity_id": key[0],
            "field_path": key[1],
            "candidate_value_private": candidate,
            "normalized_value": str(judge.get("normalized_value") or audit_value or candidate or ""),
            "unit": str(judge.get("unit") or audit.get("unit") or ""),
            "reference_status": "exact_verified" if strict else "human_review_required",
            "support_status": str(audit.get("support_status") or "missing_agent_output"),
            "evidence_spans": [],
            "reason_codes": [
                str(blind.get("reason_code") or "BLIND_OUTPUT_MISSING"),
                str(audit.get("reason_code") or "AUDIT_OUTPUT_MISSING"),
                str(judge.get("reason_code") or "ADJUDICATION_OUTPUT_MISSING"),
            ],
        }
        if span is not None:
            start, end = span
            record["evidence_spans"] = [
                {
                    "start": start,
                    "end": end,
                    "text_sha256": _sha256_text(source[start:end]),
                }
            ]
        (accepted if strict else review).append(record)
    return accepted, review


def _role_record(model: str, system: str, output: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(output, ensure_ascii=False, sort_keys=True)
    return {
        "model_id": model,
        "prompt_sha256": _sha256_text(system),
        "output_sha256": _sha256_text(serialized),
    }


def annotate(
    queue_path: Path,
    output_dir: Path,
    *,
    base_url: str,
    locator_model: str,
    auditor_model: str,
    adjudicator_model: str,
    timeout: float,
    seed: int,
    allow_nonloopback: bool,
    acknowledge_private_clinical_data: bool,
    max_cases: int,
) -> dict[str, Any]:
    if not acknowledge_private_clinical_data:
        raise PermissionError(
            "Agent execution requires --acknowledge-private-clinical-data."
        )
    _validate_endpoint(base_url, allow_nonloopback)
    tasks = _read_jsonl(queue_path)
    if max_cases >= 0:
        tasks = tasks[:max_cases]
    task_ids = [str(task["case_id"]) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Agent queue contains duplicate case_id values")
    parsed_endpoint = urllib.parse.urlparse(base_url)
    run_contract = {
        "protocol_version": PROTOCOL_VERSION,
        "queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
        "case_ids_sha256": _sha256_text("\n".join(task_ids)),
        "case_count": len(tasks),
        "models": {
            "evidence_locator": locator_model,
            "semantic_reviewer": auditor_model,
            "adjudicator": adjudicator_model,
        },
        "prompt_sha256": {
            "evidence_locator": _sha256_text(LOCATOR_SYSTEM),
            "semantic_reviewer": _sha256_text(AUDITOR_SYSTEM),
            "adjudicator": _sha256_text(ADJUDICATOR_SYSTEM),
        },
        "seed": seed,
        "generation": dict(OLLAMA_GENERATION_OPTIONS),
        "endpoint_class": (
            "loopback" if str(parsed_endpoint.hostname).lower() in LOOPBACK_HOSTS else "nonloopback"
        ),
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing_contract = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_contract != run_contract:
            raise ValueError("Output directory is bound to a different agent run contract")
    else:
        _atomic_json(manifest_path, run_contract, private=True)
    case_dir = output_dir / "case_results.private"
    case_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    started = time.time()
    for index, task in enumerate(tasks, 1):
        case_id = str(task["case_id"])
        destination = case_dir / f"{case_id}.json"
        if destination.is_file():
            continue
        source = str(task["source_text_private"])
        if str(task.get("source_record_sha256") or "") != _sha256_text(source):
            raise ValueError(f"Queue source hash mismatch for case {case_id}")
        locator_payload = {
            "protocol": PROTOCOL_VERSION,
            "case_id": case_id,
            "modality": task["modality"],
            "source_text": source,
            "allowed_entities_and_fields": task["candidate_entities"],
            "field_template_paths": task["entity_review"]["field_template_paths"],
        }
        auditor_payload = {
            "protocol": PROTOCOL_VERSION,
            "case_id": case_id,
            "modality": task["modality"],
            "source_text": source,
            "spreadsheet_candidates": task["candidate_fields"],
        }
        locator, locator_usage = _ollama_chat(
            base_url=base_url,
            model=locator_model,
            system=LOCATOR_SYSTEM,
            payload=locator_payload,
            schema=ANNOTATION_SCHEMA,
            timeout=timeout,
            seed=seed,
        )
        auditor, auditor_usage = _ollama_chat(
            base_url=base_url,
            model=auditor_model,
            system=AUDITOR_SYSTEM,
            payload=auditor_payload,
            schema=ANNOTATION_SCHEMA,
            timeout=timeout,
            seed=seed + 1,
        )
        adjudicator_payload = {
            "protocol": PROTOCOL_VERSION,
            "case_id": case_id,
            "modality": task["modality"],
            "source_text": source,
            "spreadsheet_candidates": task["candidate_fields"],
            "blind_extractor": locator,
            "candidate_auditor": auditor,
        }
        adjudicator, adjudicator_usage = _ollama_chat(
            base_url=base_url,
            model=adjudicator_model,
            system=ADJUDICATOR_SYSTEM,
            payload=adjudicator_payload,
            schema=ADJUDICATION_SCHEMA,
            timeout=timeout,
            seed=seed + 2,
        )
        accepted, review = _merge_case(task, locator, auditor, adjudicator)
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "case_id": case_id,
            "report_group_hmac": task["report_group_hmac"],
            "source_record_sha256": _sha256_text(source),
            "roles": {
                "evidence_locator": _role_record(locator_model, LOCATOR_SYSTEM, locator),
                "semantic_reviewer": _role_record(auditor_model, AUDITOR_SYSTEM, auditor),
                "adjudicator": _role_record(adjudicator_model, ADJUDICATOR_SYSTEM, adjudicator),
            },
            "usage": {
                "evidence_locator": locator_usage,
                "semantic_reviewer": auditor_usage,
                "adjudicator": adjudicator_usage,
            },
            "raw_agent_outputs_private": {
                "evidence_locator": locator,
                "semantic_reviewer": auditor,
                "adjudicator": adjudicator,
            },
            "exact_verified_fields_private": accepted,
            "human_review_fields_private": review,
        }
        _atomic_json(destination, result, private=True)
        print(f"AGENT_ANNOTATION_PROGRESS={index}/{len(tasks)} case={case_id}", flush=True)

    results = []
    for task in tasks:
        case_id = str(task["case_id"])
        path = case_dir / f"{case_id}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("case_id") != case_id:
            raise ValueError(f"Stale case result identity mismatch: {path}")
        expected_source_sha = str(task.get("source_record_sha256") or "")
        if result.get("source_record_sha256") != expected_source_sha:
            raise ValueError(f"Stale case result source mismatch: {path}")
        results.append(result)
    accepted_rows = [
        field
        for result in results
        for field in result.get("exact_verified_fields_private", [])
    ]
    review_rows = [
        field
        for result in results
        for field in result.get("human_review_fields_private", [])
    ]
    _atomic_jsonl(
        output_dir / "exact_verified_fields.private.jsonl", accepted_rows, private=True
    )
    _atomic_jsonl(
        output_dir / "human_review_queue.private.jsonl", review_rows, private=True
    )
    modality_counts = Counter(str(task.get("modality")) for task in tasks)
    summary = {
        "schema_version": f"{PROTOCOL_VERSION}.summary",
        "status": "COMPLETE" if len(results) == len(tasks) else "PARTIAL",
        "queue_sha256": run_contract["queue_sha256"],
        "cases_requested": len(tasks),
        "cases_completed": len(results),
        "modality_counts": dict(sorted(modality_counts.items())),
        "exact_verified_fields": len(accepted_rows),
        "human_review_fields": len(review_rows),
        "strict_field_coverage": (
            len(accepted_rows) / max(len(accepted_rows) + len(review_rows), 1)
        ),
        "same_model_all_roles": len({locator_model, auditor_model, adjudicator_model}) == 1,
        "elapsed_seconds": round(time.time() - started, 3),
        "privacy": {
            "case_text_in_summary": False,
            "raw_outputs_private": True,
            "nonloopback_explicitly_allowed": allow_nonloopback,
        },
        "claim_boundary": (
            "Agent-consistent source annotation, not clinical expert gold or patient-level validation."
        ),
    }
    _atomic_json(output_dir / "agent_annotation_summary.json", summary, private=False)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run private source-grounded thyroid annotation with local Ollama agents."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--locator-model", required=True)
    parser.add_argument("--auditor-model", required=True)
    parser.add_argument("--adjudicator-model", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-cases", type=int, default=-1)
    parser.add_argument("--allow-nonloopback", action="store_true")
    parser.add_argument("--acknowledge-private-clinical-data", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = annotate(
        args.queue,
        args.output_dir,
        base_url=args.base_url,
        locator_model=args.locator_model,
        auditor_model=args.auditor_model,
        adjudicator_model=args.adjudicator_model,
        timeout=args.timeout,
        seed=args.seed,
        allow_nonloopback=args.allow_nonloopback,
        acknowledge_private_clinical_data=args.acknowledge_private_clinical_data,
        max_cases=args.max_cases,
    )
    print(
        "THYROID_AGENT_ANNOTATION="
        f"{summary['status']} cases={summary['cases_completed']} "
        f"exact={summary['exact_verified_fields']} review={summary['human_review_fields']}"
    )


if __name__ == "__main__":
    main()
