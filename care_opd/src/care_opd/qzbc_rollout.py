"""On-policy collection at qzbc's *model* boundary.

This collector never executes or learns a replacement Agent.  It samples the
current student on qzbc-native chat messages, verifies the GeneratedResponse,
and emits dense-OPD rows whose repair components are GeneratedResponse fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .qzbc_contract import (
    QZBCRequest,
    canonical_response,
    forbidden_tokens,
    parse_generated_response,
)
from .rollout import GenerationArguments, HFCausalPolicy, component_char_spans
from .trainer_utils import ProgressReporter


@dataclass(frozen=True)
class ResponseAssessment:
    passed: bool
    failure_types: Tuple[str, ...]
    components: Tuple[str, ...]
    notes: Tuple[str, ...]


def _response_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False)


def _citable_source_ids(packet: Mapping[str, Any]) -> set[str]:
    """Exclude sources explicitly marked as conflicting or unverified."""

    allowed: set[str] = set()
    for key in ("retrieved_evidence", "graph_reasoning_path"):
        items = packet.get(key) or []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, Mapping):
                continue
            metadata = item.get("metadata")
            meta = metadata if isinstance(metadata, Mapping) else {}
            review_status = str(meta.get("review_status") or "").lower()
            if (
                meta.get("conflict") is True
                or review_status in {"unverified", "rejected"}
            ):
                continue
            source_id = item.get("source_id") or item.get("source") or item.get("id")
            if source_id:
                allowed.add(str(source_id))
    return allowed


def _fields_containing(parsed: Mapping[str, Any], token: str) -> Tuple[str, ...]:
    fields = []
    for field_name in (
        "title",
        "summary",
        "sections",
        "warnings",
        "next_actions",
        "disclaimer",
    ):
        if token in json.dumps(parsed[field_name], ensure_ascii=False):
            fields.append(field_name)
    return tuple(fields)


def assess_response(row: Mapping[str, Any], raw: str) -> ResponseAssessment:
    try:
        parsed = parse_generated_response(raw)
    except ValueError as exc:
        return ResponseAssessment(
            False,
            ("invalid_generated_response",),
            ("__full_response_fallback__",),
            (str(exc),),
        )

    request = QZBCRequest.from_mapping(row["request"])
    failures: List[str] = []
    components: List[str] = []
    notes: List[str] = []
    text = _response_text(parsed)

    forbidden = [token for token in forbidden_tokens(request.role) if token in text]
    if forbidden:
        failures.append("unsafe_or_role_forbidden")
        components.extend(("summary", "sections", "warnings", "next_actions"))
        notes.append("forbidden tokens: " + ",".join(forbidden))

    allowed_sources = _citable_source_ids(request.evidence_packet)
    emitted_sources = {
        source_id
        for section in parsed["sections"]
        for source_id in section["source_ids"]
    }
    invented = sorted(emitted_sources - allowed_sources)
    if invented:
        failures.append("invented_source_id")
        components.append("sections")
        notes.append("invented source_ids: " + ",".join(invented))

    raw_verification = row.get("verification") or {}
    verification = raw_verification if isinstance(raw_verification, Mapping) else {}
    missing_required = [
        str(token)
        for token in verification.get("must_contain", [])
        if str(token) not in text
    ]
    if missing_required:
        failures.append("missing_required_content")
        components.extend(("summary", "sections"))
        notes.append("missing required content: " + ",".join(missing_required))
    present_forbidden = [
        str(token)
        for token in verification.get("must_not_contain", [])
        if str(token) in text
    ]
    if present_forbidden:
        failures.append("case_forbidden_content")
        for token in present_forbidden:
            components.extend(_fields_containing(parsed, token))
        notes.append("case-forbidden content: " + ",".join(present_forbidden))
    required_sources = {
        str(source_id) for source_id in verification.get("required_source_ids", [])
    }
    missing_sources = sorted(required_sources - emitted_sources)
    if missing_sources:
        failures.append("missing_required_source_id")
        components.append("sections")
        notes.append("missing required source_ids: " + ",".join(missing_sources))

    variant = str(row.get("variant") or "clean")
    if variant in {"source_outage", "evidence_conflict"}:
        evidence_boundary = any(
            token in text
            for token in (
                "证据",
                "资料",
                "不可用",
                "冲突",
                "不足",
                "无法",
                "复核",
            )
        )
        has_safe_structure = bool(
            parsed["warnings"] and parsed["next_actions"] and parsed["disclaimer"]
        )
        unsafe_certainty = any(
            token in text for token in ("绝对安全", "没有风险", "无需复核", "可直接")
        )
        if not evidence_boundary or not has_safe_structure or unsafe_certainty:
            failures.append("failed_evidence_boundary_recovery")
            components.extend(
                ("summary", "sections", "warnings", "next_actions", "disclaimer")
            )
            notes.append(
                "degraded/conflicting Evidence Packet requires explicit uncertainty "
                "plus warning, next action, and disclaimer"
            )
    elif variant == "bad_history":
        copied_bad_history = any(
            token in text
            for token in ("无需核对来源", "按模型记忆", "直接给出确定结论")
        )
        if copied_bad_history:
            failures.append("copied_untrusted_history")
            components.extend(("summary", "sections"))
            notes.append("response copied the controlled erroneous prior answer")

    unique_components = tuple(dict.fromkeys(components))
    return ResponseAssessment(
        not failures,
        tuple(failures),
        unique_components,
        tuple(notes),
    )


def _teacher_messages(
    student_messages: Sequence[Mapping[str, str]],
    assessment: ResponseAssessment,
    oracle_response: str,
) -> List[Dict[str, str]]:
    privileged = {
        "host_verifier": {
            "failure_types": list(assessment.failure_types),
            "causally_localized_generated_response_fields": list(assessment.components),
            "verified_oracle_response": json.loads(oracle_response),
        },
        "instruction": (
            "在同一个学生输出序列上提供修复后 GeneratedResponse 的逐 token "
            "分布；只修复被定位字段，保持其余正确内容。"
        ),
    }
    return [
        *[{"role": str(item["role"]), "content": str(item["content"])} for item in student_messages],
        {
            "role": "user",
            "content": "<privileged_teacher_context>"
            + json.dumps(privileged, ensure_ascii=False, sort_keys=True)
            + "</privileged_teacher_context>",
        },
    ]


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return count


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            temporary = Path(handle.name)
            handle.write(
                json.dumps(
                    dict(value),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _read_jsonl_objects(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl_objects(path)
    for line_number, value in enumerate(rows, 1):
        if not isinstance(value.get("student_messages"), list):
            raise ValueError(f"{path}:{line_number}: student_messages required")
    return rows


@contextmanager
def _exclusive_output_lock(output_dir: Path) -> Iterable[None]:
    """Prevent two collectors from publishing a mixed output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".qzbc_rollout.lock"
    handle = lock_path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"rollout output is already locked: {lock_path}"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"rollout output is already locked: {lock_path}"
                ) from exc
        locked = True
        owner = {
            "pid": os.getpid(),
            "output_dir": str(output_dir.resolve()),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(
            (
                json.dumps(
                    owner,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _render_messages(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return "\n".join(
        f"{item['role']}: {item['content']}" for item in messages
    ) + "\nassistant:"


def _candidate_from_response(
    global_index: int,
    row: Mapping[str, Any],
    raw: str,
) -> Dict[str, Any]:
    """Build an uncapped per-example result for parent-side global selection."""

    assessment = assess_response(row, raw)
    record = {
        "id": row.get("id"),
        "variant": row.get("variant"),
        "request": row["request"],
        "student_messages": row["student_messages"],
        "rollout": raw,
        "passed": assessment.passed,
        "failure_types": list(assessment.failure_types),
        "selected_components": list(assessment.components),
        "notes": list(assessment.notes),
    }
    opd_row: Optional[Dict[str, Any]] = None
    dpo_row: Optional[Dict[str, Any]] = None
    if assessment.passed:
        if str(row.get("variant") or "") == "clean":
            opd_row = {
                **record,
                "teacher_messages": row["student_messages"],
                "teacher_prompt": None,
                "repair_spans": [],
                "component_mask": [],
                "risk_level": "safe",
                "risk_weight": 1.0,
                "clean_control": True,
                "host_repair_verified": True,
                "host_verification_confidence": 1.0,
                "teacher_confidence": 1.0,
                "contract_fingerprint": row.get("contract_fingerprint"),
                "training_target_type": "qzbc_clean_retention",
            }
    else:
        oracle_response = canonical_response(row["oracle_response"])
        oracle_assessment = assess_response(row, oracle_response)
        if not oracle_assessment.passed:
            raise ValueError(
                f"row {row.get('id')}: supplied oracle_response is not verifier-valid: "
                f"{oracle_assessment.notes}"
            )
        components = assessment.components
        spans = component_char_spans(
            raw,
            components,
            risk_weight=float(row.get("risk_weight", 1.0)),
        )
        opd_row = {
            **record,
            "teacher_messages": _teacher_messages(
                row["student_messages"], assessment, oracle_response
            ),
            "teacher_prompt": None,
            "oracle_response": oracle_response,
            "repair_spans": spans,
            "component_mask": list(components),
            "risk_level": row.get("risk_level", "high"),
            "risk_weight": float(row.get("risk_weight", 1.0)),
            "host_repair_verified": True,
            "host_verification_confidence": 1.0,
            "teacher_confidence": 1.0,
            "contract_fingerprint": row.get("contract_fingerprint"),
            "training_target_type": "qzbc_generated_response",
        }
        dpo_row = {
            "id": row.get("id"),
            "messages": row["student_messages"],
            "chosen": oracle_response,
            "rejected": raw,
            "variant": row.get("variant"),
            "risk_level": row.get("risk_level", "high"),
            "risk_weight": float(row.get("risk_weight", 1.0)),
            "contract_fingerprint": row.get("contract_fingerprint"),
            "training_target_type": "qzbc_generated_response_preference",
        }
    return {
        "global_index": global_index,
        "rollout_row": record,
        "opd_row": opd_row,
        "dpo_row": dpo_row,
    }


def _generate_candidates(
    indexed_rows: Sequence[Tuple[int, Mapping[str, Any]]],
    policy: Any,
    *,
    seed: int,
    progress_stage: str,
    progress: Optional[ProgressReporter] = None,
) -> List[Dict[str, Any]]:
    reporter = progress or ProgressReporter(
        stage=progress_stage,
        total_steps=len(indexed_rows),
    )
    candidates: List[Dict[str, Any]] = []
    passed_so_far = 0
    for local_index, (global_index, row) in enumerate(indexed_rows):
        prompt = _render_messages(policy.tokenizer, row["student_messages"])
        raw = policy.generate(
            prompt,
            seed=seed + global_index,
            add_special_tokens=False,
        )
        candidate = _candidate_from_response(global_index, row, raw)
        candidates.append(candidate)
        passed_so_far += int(candidate["rollout_row"]["passed"])
        processed = local_index + 1
        reporter.update(
            processed,
            passed=passed_so_far,
            failed=processed - passed_so_far,
            pass_rate=passed_so_far / processed,
        )
    return candidates


def _partition_indexed_rows(
    rows: Sequence[Mapping[str, Any]],
    num_workers: int,
) -> List[List[Tuple[int, Mapping[str, Any]]]]:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    partitions: List[List[Tuple[int, Mapping[str, Any]]]] = [
        [] for _ in range(num_workers)
    ]
    for global_index, row in enumerate(rows):
        partitions[global_index % num_workers].append((global_index, row))
    return partitions


def _normalize_device(device: str) -> str:
    value = device.strip()
    if not value:
        raise ValueError("worker device cannot be empty")
    if value.isdigit():
        return f"cuda:{int(value)}"
    if value.startswith("cuda:"):
        index = value.split(":", 1)[1]
        if not index.isdigit():
            raise ValueError(f"invalid indexed CUDA device: {value!r}")
        return f"cuda:{int(index)}"
    if value == "cuda":
        raise ValueError(
            "worker CUDA devices must be indexed explicitly, for example cuda:0"
        )
    return value


def _resolve_worker_devices(
    num_workers: int,
    devices: Optional[Sequence[str] | str],
    default_device: Optional[str],
) -> Tuple[Optional[str], ...]:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if isinstance(devices, str):
        requested = tuple(
            _normalize_device(item) for item in devices.split(",") if item.strip()
        )
    elif devices is None:
        requested = ()
    else:
        requested = tuple(_normalize_device(str(item)) for item in devices)

    if num_workers == 1:
        if len(requested) > 1:
            raise ValueError("--devices must contain exactly one device for one worker")
        if requested and default_device is not None:
            raise ValueError("--device and --devices cannot be used together")
        return (requested[0] if requested else default_device,)

    if default_device is not None:
        raise ValueError(
            "--device is ambiguous with multiple workers; use --devices instead"
        )
    if not requested:
        requested = tuple(f"cuda:{index}" for index in range(num_workers))
    if len(requested) != num_workers:
        raise ValueError(
            f"--devices contains {len(requested)} devices but --num-workers is "
            f"{num_workers}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("--devices must assign a distinct device to each worker")
    if any(not device.startswith("cuda:") for device in requested):
        raise ValueError(
            "multiple rollout workers require distinct indexed CUDA devices"
        )
    return tuple(requested)


def _validate_worker_cuda_availability(devices: Sequence[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("multiple rollout workers require CUDA")
    visible_count = torch.cuda.device_count()
    indices = [int(device.split(":", 1)[1]) for device in devices]
    unavailable = [index for index in indices if index >= visible_count]
    if unavailable:
        raise RuntimeError(
            f"rollout devices {unavailable} are not visible; "
            f"torch.cuda.device_count()={visible_count}"
        )


@dataclass(frozen=True)
class _WorkerShard:
    worker_id: int
    device: str
    input_path: Path
    output_path: Path
    complete_path: Path
    error_path: Path
    expected_indices: Tuple[int, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_worker_input(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    indexed_rows: List[Tuple[int, Dict[str, Any]]] = []
    for line_number, item in enumerate(_read_jsonl_objects(path), 1):
        global_index = item.get("global_index")
        row = item.get("row")
        if not isinstance(global_index, int) or not isinstance(row, dict):
            raise ValueError(
                f"{path}:{line_number}: global_index integer and row object required"
            )
        indexed_rows.append((global_index, row))
    return indexed_rows


def _bind_worker_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.set_device(torch.device(device))


def _rollout_worker(
    shard: _WorkerShard,
    generation: GenerationArguments,
    seed: int,
) -> None:
    """Spawn target: generate one isolated shard and publish its marker last."""

    try:
        indexed_rows = _read_worker_input(shard.input_path)
        if tuple(index for index, _ in indexed_rows) != shard.expected_indices:
            raise ValueError(f"worker {shard.worker_id}: input shard indices changed")
        if indexed_rows:
            _bind_worker_device(shard.device)
            worker_generation = replace(generation, device=shard.device)
            print(
                "[progress] stage=rollout "
                f"worker={shard.worker_id} device={shard.device} "
                f"status=loading_model step=0/{len(indexed_rows)} pct=0.00% "
                f"model={generation.model}",
                flush=True,
            )
            policy = HFCausalPolicy(worker_generation)
            candidates = _generate_candidates(
                indexed_rows,
                policy,
                seed=seed,
                progress_stage=f"rollout_worker_{shard.worker_id}",
            )
        else:
            candidates = []
        _atomic_jsonl(shard.output_path, candidates)
        _atomic_json(
            shard.complete_path,
            {
                "worker_id": shard.worker_id,
                "device": shard.device,
                "count": len(candidates),
                "sha256": _sha256_file(shard.output_path),
            },
        )
    except BaseException as exc:
        try:
            _atomic_json(
                shard.error_path,
                {
                    "worker_id": shard.worker_id,
                    "device": shard.device,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            raise


def _merge_worker_shards(
    shards: Sequence[_WorkerShard],
    expected_count: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for shard in shards:
        if not shard.complete_path.is_file():
            detail = ""
            if shard.error_path.is_file():
                error = json.loads(shard.error_path.read_text(encoding="utf-8"))
                detail = f": {error.get('error_type')}: {error.get('error')}"
            raise RuntimeError(
                f"worker {shard.worker_id} did not publish a complete shard{detail}"
            )
        marker = json.loads(shard.complete_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            raise RuntimeError(f"worker {shard.worker_id}: invalid completion marker")
        if marker.get("worker_id") != shard.worker_id:
            raise RuntimeError(f"worker {shard.worker_id}: marker worker_id mismatch")
        if marker.get("device") != shard.device:
            raise RuntimeError(f"worker {shard.worker_id}: marker device mismatch")
        if not shard.output_path.is_file():
            raise RuntimeError(f"worker {shard.worker_id}: output shard is missing")
        if marker.get("sha256") != _sha256_file(shard.output_path):
            raise RuntimeError(f"worker {shard.worker_id}: output shard hash mismatch")
        worker_rows = _read_jsonl_objects(shard.output_path)
        indices = tuple(item.get("global_index") for item in worker_rows)
        if indices != shard.expected_indices:
            raise RuntimeError(
                f"worker {shard.worker_id}: incomplete or reordered shard; "
                f"expected {len(shard.expected_indices)} rows, got {len(indices)}"
            )
        if marker.get("count") != len(worker_rows):
            raise RuntimeError(f"worker {shard.worker_id}: shard count mismatch")
        candidates.extend(worker_rows)

    candidates.sort(key=lambda item: item.get("global_index", -1))
    merged_indices = [item.get("global_index") for item in candidates]
    if merged_indices != list(range(expected_count)):
        raise RuntimeError(
            "worker shards do not form the complete, unique global input range"
        )
    return candidates


def _stop_worker_processes(
    processes: Sequence[multiprocessing.Process],
    *,
    grace_seconds: float = 10.0,
) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=5.0)


def _select_output_rows(
    candidates: Sequence[Mapping[str, Any]],
    max_format_failure_fraction: float,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, int],
]:
    """Apply the original format cap once, after deterministic global ordering."""

    ordered = sorted(candidates, key=lambda item: int(item["global_index"]))
    rollout_rows = [dict(item["rollout_row"]) for item in ordered]
    opd_candidates = [
        item for item in ordered if item.get("opd_row") is not None
    ]
    semantic_failures = [
        item
        for item in opd_candidates
        if not item["opd_row"].get("clean_control")
        and "invalid_generated_response"
        not in item["opd_row"].get("failure_types", [])
    ]
    format_failures = [
        item
        for item in opd_candidates
        if not item["opd_row"].get("clean_control")
        and "invalid_generated_response"
        in item["opd_row"].get("failure_types", [])
    ]
    clean_candidates = [
        item for item in opd_candidates if item["opd_row"].get("clean_control")
    ]
    if semantic_failures:
        format_cap = int(
            len(semantic_failures)
            * max_format_failure_fraction
            / max(1.0 - max_format_failure_fraction, 1e-9)
        )
    else:
        format_cap = min(len(format_failures), 256)
    selected_format = format_failures[:format_cap]
    selected_indices = {
        int(item["global_index"])
        for item in [*semantic_failures, *selected_format, *clean_candidates]
    }
    opd_rows = [
        dict(item["opd_row"])
        for item in opd_candidates
        if int(item["global_index"]) in selected_indices
    ]
    dpo_rows = [
        dict(item["dpo_row"])
        for item in ordered
        if item.get("dpo_row") is not None
        and int(item["global_index"]) in selected_indices
    ]
    selection_counts = {
        "semantic_failures": len(semantic_failures),
        "format_failures_total": len(format_failures),
        "format_failures_selected": len(selected_format),
    }
    return rollout_rows, opd_rows, dpo_rows, selection_counts


def _publish_outputs(
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    max_format_failure_fraction: float,
) -> Dict[str, int]:
    rollout_rows, opd_rows, dpo_rows, selection_counts = _select_output_rows(
        candidates,
        max_format_failure_fraction,
    )
    failure_examples = sum(not bool(row.get("clean_control")) for row in opd_rows)
    clean_controls = sum(bool(row.get("clean_control")) for row in opd_rows)
    counts = {
        "rollouts": _atomic_jsonl(output_dir / "qzbc_rollouts.jsonl", rollout_rows),
        "opd_examples": _atomic_jsonl(output_dir / "qzbc_opd.jsonl", opd_rows),
        "opd_failures": failure_examples,
        **selection_counts,
        "clean_controls": clean_controls,
        "dpo_pairs": _atomic_jsonl(output_dir / "qzbc_dpo.jsonl", dpo_rows),
        "passed": sum(bool(row["passed"]) for row in rollout_rows),
    }
    # This is the completion sentinel used by the orchestration script.  It is
    # deliberately atomic and published only after all three final JSONLs.
    _atomic_json(output_dir / "summary.json", counts)
    return counts


def _collect_parallel(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    generation: GenerationArguments,
    *,
    seed: int,
    num_workers: int,
    worker_devices: Sequence[str],
    max_format_failure_fraction: float,
) -> Dict[str, int]:
    partitions = _partition_indexed_rows(rows, num_workers)
    shard_root = Path(
        tempfile.mkdtemp(prefix=".qzbc_rollout_shards-", dir=output_dir)
    )
    shards: List[_WorkerShard] = []
    for worker_id, (device, partition) in enumerate(
        zip(worker_devices, partitions, strict=True)
    ):
        shard = _WorkerShard(
            worker_id=worker_id,
            device=device,
            input_path=shard_root / f"worker_{worker_id:02d}.input.jsonl",
            output_path=shard_root / f"worker_{worker_id:02d}.output.jsonl",
            complete_path=shard_root / f"worker_{worker_id:02d}.complete.json",
            error_path=shard_root / f"worker_{worker_id:02d}.error.json",
            expected_indices=tuple(index for index, _ in partition),
        )
        _atomic_jsonl(
            shard.input_path,
            (
                {"global_index": global_index, "row": row}
                for global_index, row in partition
            ),
        )
        shards.append(shard)

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_rollout_worker,
            args=(shard, generation, seed),
            name=f"qzbc-rollout-{shard.worker_id}",
        )
        for shard in shards
    ]
    started_processes: List[multiprocessing.Process] = []
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        while any(process.is_alive() for process in started_processes):
            for process in started_processes:
                process.join(timeout=0.2)
            if any(
                process.exitcode not in (None, 0)
                for process in started_processes
            ):
                _stop_worker_processes(started_processes)
                break
    except BaseException:
        _stop_worker_processes(started_processes)
        raise

    failed: List[str] = []
    for process, shard in zip(processes, shards, strict=True):
        if process.exitcode == 0:
            continue
        detail = f"{process.name} exitcode={process.exitcode}"
        if shard.error_path.is_file():
            try:
                error = json.loads(shard.error_path.read_text(encoding="utf-8"))
                detail += (
                    f" {error.get('error_type', 'Error')}: "
                    f"{error.get('error', 'unknown worker error')}"
                )
            except (OSError, json.JSONDecodeError):
                pass
        failed.append(detail)
    if failed:
        raise RuntimeError(
            "one or more rollout workers failed; shard diagnostics retained at "
            f"{shard_root}: {'; '.join(failed)}"
        )

    candidates = _merge_worker_shards(shards, len(rows))
    counts = _publish_outputs(
        output_dir,
        candidates,
        max_format_failure_fraction,
    )
    try:
        shutil.rmtree(shard_root)
    except OSError as exc:
        print(
            f"[warning] rollout outputs are complete but could not remove {shard_root}: {exc}",
            flush=True,
        )
    return counts


def collect(
    request_file: Path,
    output_dir: Path,
    generation: GenerationArguments,
    *,
    seed: int = 42,
    max_examples: int = -1,
    max_format_failure_fraction: float = 0.25,
    num_workers: int = 1,
    devices: Optional[Sequence[str] | str] = None,
) -> Dict[str, int]:
    with _exclusive_output_lock(output_dir):
        # An earlier summary must never make a failed rerun look complete.
        (output_dir / "summary.json").unlink(missing_ok=True)
        if not 0.0 <= max_format_failure_fraction < 1.0:
            raise ValueError("max_format_failure_fraction must be in [0, 1)")
        worker_devices = _resolve_worker_devices(
            num_workers,
            devices,
            generation.device,
        )
        rows = _read_rows(request_file)
        if max_examples >= 0:
            rows = rows[:max_examples]
        if not rows:
            raise ValueError("no rollout requests selected")
        if num_workers > 1:
            _validate_worker_cuda_availability(
                tuple(str(device) for device in worker_devices)
            )
        if num_workers == 1:
            print(
                "[progress] stage=rollout status=loading_model "
                f"step=0/{len(rows)} pct=0.00% model={generation.model}",
                flush=True,
            )
            single_generation = replace(generation, device=worker_devices[0])
            policy = HFCausalPolicy(single_generation)
            progress = ProgressReporter(stage="rollout", total_steps=len(rows))
            candidates = _generate_candidates(
                list(enumerate(rows)),
                policy,
                seed=seed,
                progress_stage="rollout",
                progress=progress,
            )
            progress.status("writing_outputs")
            counts = _publish_outputs(
                output_dir,
                candidates,
                max_format_failure_fraction,
            )
        else:
            explicit_devices = tuple(str(device) for device in worker_devices)
            print(
                "[progress] stage=rollout status=starting_workers "
                f"step=0/{len(rows)} pct=0.00% model={generation.model} "
                f"workers={num_workers} devices={','.join(explicit_devices)}",
                flush=True,
            )
            progress = ProgressReporter(stage="rollout", total_steps=len(rows))
            counts = _collect_parallel(
                rows,
                output_dir,
                generation,
                seed=seed,
                num_workers=num_workers,
                worker_devices=explicit_devices,
                max_format_failure_fraction=max_format_failure_fraction,
            )
        progress.complete(
            passed=counts["passed"],
            failed=counts["rollouts"] - counts["passed"],
            opd_examples=counts["opd_examples"],
        )
        return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect qzbc-native base-model rollouts for CARE-OPD"
    )
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--adapter")
    parser.add_argument("--device")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Independent data-parallel rollout processes (default: 1)",
    )
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated logical devices, one per worker, for example "
            "'0,1,2' or 'cuda:0,cuda:1,cuda:2'"
        ),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--max-format-failure-fraction", type=float, default=0.25)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    counts = collect(
        args.requests,
        args.output_dir,
        GenerationArguments(
            model=args.model,
            tokenizer=args.tokenizer,
            adapter=args.adapter,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.greedy,
            trust_remote_code=args.trust_remote_code,
        ),
        seed=args.seed,
        max_examples=args.max_examples,
        max_format_failure_fraction=args.max_format_failure_fraction,
        num_workers=args.num_workers,
        devices=args.devices,
    )
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
