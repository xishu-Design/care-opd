"""Paired held-out evaluation at qzbc's native Ollama model boundary.

The input is the compiled ``qzbc_on_policy_requests.jsonl`` file.  Both model
arms receive the exact same ``student_messages`` and the same structured
``GeneratedResponse`` format/options used by qzbc's non-streaming provider.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .qzbc_bridge import (
    _exact_mcnemar,
    _ollama_model_name,
    check_ollama_models,
)
from .qzbc_rollout import ResponseAssessment, assess_response


# Semantically identical to qzbc ``GeneratedResponse.model_json_schema()``.
GENERATED_RESPONSE_FORMAT: Mapping[str, Any] = {
    "$defs": {
        "GeneratedSection": {
            "properties": {
                "heading": {"default": "", "title": "Heading", "type": "string"},
                "content": {"default": "", "title": "Content", "type": "string"},
                "source_ids": {
                    "items": {"type": "string"},
                    "title": "Source Ids",
                    "type": "array",
                },
            },
            "title": "GeneratedSection",
            "type": "object",
        }
    },
    "properties": {
        "title": {"default": "", "title": "Title", "type": "string"},
        "summary": {"default": "", "title": "Summary", "type": "string"},
        "sections": {
            "items": {"$ref": "#/$defs/GeneratedSection"},
            "title": "Sections",
            "type": "array",
        },
        "warnings": {
            "items": {"type": "string"},
            "title": "Warnings",
            "type": "array",
        },
        "next_actions": {
            "items": {"type": "string"},
            "title": "Next Actions",
            "type": "array",
        },
        "disclaimer": {"default": "", "title": "Disclaimer", "type": "string"},
    },
    "title": "GeneratedResponse",
    "type": "object",
}


def _read_requests(path: Path, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            messages = value.get("student_messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"{path}:{line_number}: non-empty student_messages required"
                )
            for message in messages:
                if (
                    not isinstance(message, Mapping)
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError(
                        f"{path}:{line_number}: malformed student_messages"
                    )
            if not isinstance(value.get("request"), Mapping):
                raise ValueError(f"{path}:{line_number}: request object required")
            case_id = str(value.get("id") or f"row-{line_number:06d}")
            if case_id in case_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {case_id!r}")
            case_ids.add(case_id)
            value = dict(value)
            value["id"] = case_id
            rows.append(value)
            if max_examples >= 0 and len(rows) >= max_examples:
                break
    if not rows:
        raise ValueError(f"{path}: no evaluation requests")
    return rows


def _ollama_chat(
    *,
    ollama_base_url: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    num_predict: int,
    num_ctx: int,
    timeout: float,
) -> tuple[str, int, Mapping[str, int]]:
    payload = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "stream": False,
        "format": GENERATED_RESPONSE_FORMAT,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    request = urllib.request.Request(
        ollama_base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    latency_ms = int((time.perf_counter() - started) * 1000)
    if not isinstance(body, Mapping):
        raise RuntimeError("Ollama returned a non-object response")
    if body.get("error"):
        raise RuntimeError(str(body["error"]))
    message = body.get("message")
    content = (
        str(message.get("content") or "") if isinstance(message, Mapping) else ""
    )
    if not content:
        raise RuntimeError("Ollama returned empty content")
    token_usage = {
        "input_tokens": int(body.get("prompt_eval_count") or 0),
        "output_tokens": int(body.get("eval_count") or 0),
    }
    return content, latency_ms, token_usage


def _failed_generation(exc: Exception) -> ResponseAssessment:
    return ResponseAssessment(
        passed=False,
        failure_types=("generation_error",),
        components=("__full_response_fallback__",),
        notes=(str(exc) or exc.__class__.__name__,),
    )


def _evaluate_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    ollama_base_url: str,
    model: str,
    temperature: float,
    num_predict: int,
    num_ctx: int,
    timeout: float,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    for row in rows:
        raw = ""
        latency_ms = 0
        token_usage: Mapping[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }
        generation_error: Optional[str] = None
        try:
            raw, latency_ms, token_usage = _ollama_chat(
                ollama_base_url=ollama_base_url,
                model=model,
                messages=row["student_messages"],
                temperature=temperature,
                num_predict=num_predict,
                num_ctx=num_ctx,
                timeout=timeout,
            )
            assessment = assess_response(row, raw)
        except Exception as exc:
            generation_error = str(exc) or exc.__class__.__name__
            assessment = _failed_generation(exc)
        failure_counts.update(assessment.failure_types)
        request = row["request"]
        cases.append(
            {
                "case_id": str(row["id"]),
                "variant": row.get("variant"),
                "role": request.get("role"),
                "task_type": request.get("task_type"),
                "passed": assessment.passed,
                "failure_types": list(assessment.failure_types),
                "selected_components": list(assessment.components),
                "notes": list(assessment.notes),
                "generation_error": generation_error,
                "latency_ms": latency_ms,
                "token_usage": dict(token_usage),
                "response": raw,
            }
        )
    passed = sum(int(case["passed"]) for case in cases)
    return {
        "model": model,
        "total_cases": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": passed / len(cases),
        "generation_errors": failure_counts["generation_error"],
        "avg_latency_ms": sum(case["latency_ms"] for case in cases) / len(cases),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "cases": cases,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def evaluate_paired(
    *,
    request_file: Path,
    output: Path,
    ollama_base_url: str,
    base_model: str,
    trained_model: str,
    temperature: float = 0.2,
    num_predict: int = 1200,
    num_ctx: int = 32768,
    timeout: float = 300.0,
    max_examples: int = -1,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = output.expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite")
    base_model = _ollama_model_name(base_model)
    trained_model = _ollama_model_name(trained_model)
    if base_model == trained_model:
        raise ValueError("base and trained model names must differ")
    rows = _read_requests(request_file, max_examples)
    check_ollama_models(
        ollama_base_url,
        (base_model, trained_model),
        timeout=min(timeout, 30.0),
    )
    common = {
        "ollama_base_url": ollama_base_url,
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
        "timeout": timeout,
    }
    base = _evaluate_arm(rows, model=base_model, **common)
    trained = _evaluate_arm(rows, model=trained_model, **common)
    base_cases = {case["case_id"]: case for case in base["cases"]}
    trained_cases = {case["case_id"]: case for case in trained["cases"]}
    paired_cases: list[dict[str, Any]] = []
    wins = losses = ties = 0
    for row in rows:
        case_id = str(row["id"])
        before = base_cases[case_id]
        after = trained_cases[case_id]
        delta = int(after["passed"]) - int(before["passed"])
        wins += int(delta > 0)
        losses += int(delta < 0)
        ties += int(delta == 0)
        paired_cases.append(
            {
                "case_id": case_id,
                "variant": row.get("variant"),
                "role": after["role"],
                "task_type": after["task_type"],
                "base_passed": before["passed"],
                "trained_passed": after["passed"],
                "delta": delta,
                "base_failure_types": before["failure_types"],
                "trained_failure_types": after["failure_types"],
                "base_latency_ms": before["latency_ms"],
                "trained_latency_ms": after["latency_ms"],
            }
        )
    result = {
        "schema_version": "care-opd-qzbc-native-heldout-comparison-v1",
        "request_file": str(request_file.expanduser().resolve()),
        "model_boundary": "qzbc student_messages -> GeneratedResponse",
        "ollama_base_url": ollama_base_url,
        "generation_options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
        "base_model": base_model,
        "trained_model": trained_model,
        "total_cases": len(rows),
        "paired_outcomes": {
            "trained_wins": wins,
            "trained_losses": losses,
            "ties": ties,
            "exact_mcnemar_p_value": _exact_mcnemar(wins, losses),
        },
        "metric_delta_trained_minus_base": {
            "passed": trained["passed"] - base["passed"],
            "pass_rate": trained["pass_rate"] - base["pass_rate"],
            "generation_errors": (
                trained["generation_errors"] - base["generation_errors"]
            ),
            "avg_latency_ms": trained["avg_latency_ms"] - base["avg_latency_ms"],
        },
        "base_report": base,
        "trained_report": trained,
        "case_pairs": paired_cases,
    }
    _atomic_json(target, result)
    csv_path = target.with_suffix(".cases.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(paired_cases[0]) if paired_cases else ["case_id"],
        )
        writer.writeheader()
        writer.writerows(paired_cases)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_paired(
        request_file=args.requests,
        output=args.output,
        ollama_base_url=args.ollama_base_url,
        base_model=args.base_model,
        trained_model=args.trained_model,
        temperature=args.temperature,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        timeout=args.timeout,
        max_examples=args.max_examples,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "total_cases": result["total_cases"],
                **result["paired_outcomes"],
                "pass_rate_delta": result[
                    "metric_delta_trained_minus_base"
                ]["pass_rate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
