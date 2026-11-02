#!/usr/bin/env python3
"""Verify model identity, evaluation contracts, and common warm-start ancestry.

The audit deliberately distinguishes three statements:

* the files currently on disk match the frozen evaluation suite;
* the trained branches trace through ``export_manifest.json`` to a common
  warm-start policy; and
* the frozen model matches an independently fixed, pre-training origin
  reference.

The first two statements do not by themselves prove the third one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_LABELS = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)
REQUIRED_MODEL_FILES = ("config.json", "tokenizer_config.json")
OPTIONAL_IDENTITY_PATTERNS = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "chat_template*",
    "special_tokens_map.json",
    "added_tokens.json",
    "processor_config.json",
    "preprocessor_config.json",
    "generation_config.json",
)
WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
DIRECT_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
FORBIDDEN_ADAPTER_PATTERNS = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
RUN_CONTRACT_FIELDS = (
    "protocol_version",
    "model_label",
    "model_path",
    "prompt_mode",
    "num_workers",
    "devices",
    "fingerprints",
    "evaluator_sha256",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
IMMUTABLE_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    """Hash a stable file, caching only within this audit process."""

    resolved = path.resolve(strict=True)
    before = resolved.stat()
    cache_key = (str(resolved), before.st_size, before.st_mtime_ns)
    cached = _SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {resolved}")
    value = digest.hexdigest()
    _SHA256_CACHE[cache_key] = value
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _safe_relative_name(value: Any) -> str:
    name = str(value)
    pure = PurePosixPath(name)
    if (
        not name
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in name
    ):
        raise ValueError(f"unsafe inventory path: {name!r}")
    return pure.as_posix()


def normalize_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("model inventory must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("model inventory entries must be objects")
        name = _safe_relative_name(raw.get("name"))
        if name in names:
            raise ValueError(f"duplicate model inventory path: {name}")
        names.add(name)
        size = int(raw.get("bytes", -1))
        digest = str(raw.get("sha256", ""))
        if size < 0 or SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid inventory metadata for {name}")
        normalized.append({"name": name, "bytes": size, "sha256": digest})
    return sorted(normalized, key=lambda item: item["name"])


def inventory_fingerprint(value: Any) -> str:
    return canonical_hash(normalize_inventory(value))


def _adapter_traces(root: Path) -> list[str]:
    traces = []
    for pattern in FORBIDDEN_ADAPTER_PATTERNS:
        traces.extend(path.name for path in root.glob(pattern) if path.is_file())
    return sorted(set(traces))


def _weight_files(root: Path) -> list[Path]:
    indexes = [root / name for name in WEIGHT_INDEX_FILES if (root / name).is_file()]
    direct = [root / name for name in DIRECT_WEIGHT_FILES if (root / name).is_file()]
    if len(indexes) > 1 or (indexes and direct):
        raise ValueError(f"ambiguous weight layouts in {root}")
    if indexes:
        index_path = indexes[0]
        payload = load_json(index_path)
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"invalid or empty weight index: {index_path}")
        names = sorted({_safe_relative_name(name) for name in weight_map.values()})
        files = [root.joinpath(*PurePosixPath(name).parts) for name in names]
        if not all(path.is_file() for path in files):
            missing = [str(path) for path in files if not path.is_file()]
            raise ValueError(f"weight index references missing shards: {missing}")
        discovered = {
            path.resolve()
            for pattern in ("model*.safetensors", "pytorch_model*.bin")
            for path in root.glob(pattern)
            if path.is_file()
        }
        expected = {path.resolve() for path in files}
        if discovered != expected:
            extras = sorted(str(path) for path in discovered - expected)
            missing = sorted(str(path) for path in expected - discovered)
            raise ValueError(
                f"weight inventory/index disagreement in {root}: extras={extras}, missing={missing}"
            )
        return [index_path, *files]
    if len(direct) != 1:
        raise ValueError(f"expected one standalone weight file or one weight index in {root}")
    other_weights = {
        path.resolve()
        for pattern in ("model*.safetensors", "pytorch_model*.bin")
        for path in root.glob(pattern)
        if path.is_file()
    }
    if other_weights != {direct[0].resolve()}:
        raise ValueError(f"unexpected additional model weights in {root}")
    return direct


def identity_files(
    root: Path,
    *,
    require_export_manifest: bool = False,
    include_export_manifest: bool = False,
) -> list[Path]:
    root = root.resolve(strict=True)
    required = [root / name for name in REQUIRED_MODEL_FILES]
    if require_export_manifest:
        required.append(root / "export_manifest.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"model is not a complete standalone checkpoint; missing {missing}")
    traces = _adapter_traces(root)
    if traces:
        raise ValueError(f"standalone checkpoint retains adapter dependencies: {traces}")
    files: list[Path] = [root / name for name in REQUIRED_MODEL_FILES]
    for pattern in OPTIONAL_IDENTITY_PATTERNS:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    if include_export_manifest:
        export_manifest = root / "export_manifest.json"
        if export_manifest.is_file():
            files.append(export_manifest)
    files.extend(_weight_files(root))
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _inventory_from_files(root: Path, files: Sequence[Path]) -> list[dict[str, Any]]:
    inventory = []
    for index, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        print(
            f"[lineage] hashing {index}/{len(files)} {root.name}/{relative} "
            f"({path.stat().st_size} bytes)",
            flush=True,
        )
        inventory.append(
            {
                "name": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return normalize_inventory(inventory)


def fingerprint_directory(root: Path) -> dict[str, Any]:
    """Hash behavior-affecting standalone content, excluding path metadata."""

    root = root.resolve(strict=True)
    files = identity_files(root, require_export_manifest=False, include_export_manifest=False)
    inventory = _inventory_from_files(root, files)
    return {
        "canonical_path": str(root),
        "model_fingerprint": canonical_hash(inventory),
        "inventory": inventory,
    }


def suite_entry(suite: Mapping[str, Any], label: str) -> dict[str, Any]:
    matches = [entry for entry in suite.get("models", []) if entry.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label!r} entry in model suite")
    return dict(matches[0])


def audit_suite_entry(
    entry: Mapping[str, Any], *, label: str, rehash: bool
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    report: dict[str, Any] = {"label": label, "rehash_requested": rehash}
    try:
        root = Path(str(entry["canonical_path"])).resolve(strict=True)
        report["canonical_path"] = str(root)
        if str(root) != str(entry.get("canonical_path")):
            errors.append(f"{label}: suite path is not canonical")
        declared = normalize_inventory(entry.get("inventory"))
        declared_fp = canonical_hash(declared)
        report["declared_model_fingerprint"] = entry.get("model_fingerprint")
        report["inventory_recomputed_fingerprint"] = declared_fp
        report["inventory_fingerprint_closes"] = declared_fp == entry.get("model_fingerprint")
        if not report["inventory_fingerprint_closes"]:
            errors.append(f"{label}: declared inventory does not close model_fingerprint")

        require_export = label != "frozen"
        files = identity_files(
            root,
            require_export_manifest=require_export,
            include_export_manifest=require_export,
        )
        actual_names = [path.relative_to(root).as_posix() for path in files]
        declared_by_name = {item["name"]: item for item in declared}
        if set(actual_names) != set(declared_by_name):
            errors.append(
                f"{label}: current identity file set differs from frozen suite inventory"
            )
        size_mismatches = [
            name
            for name in set(actual_names) & set(declared_by_name)
            if (root / name).stat().st_size != declared_by_name[name]["bytes"]
        ]
        if size_mismatches:
            errors.append(f"{label}: size changed for {size_mismatches}")
        report["identity_files"] = actual_names
        report["size_only_match"] = not size_mismatches and set(actual_names) == set(
            declared_by_name
        )
        if rehash:
            actual_inventory = _inventory_from_files(root, files)
            actual_fp = canonical_hash(actual_inventory)
            report["actual_model_fingerprint"] = actual_fp
            report["content_match"] = actual_fp == entry.get("model_fingerprint")
            if not report["content_match"]:
                errors.append(f"{label}: rehashed content differs from model suite")
            report["verification_level"] = "REHASHED_CONTENT"
        else:
            report["actual_model_fingerprint"] = None
            report["content_match"] = None
            report["verification_level"] = "PATH_NAME_SIZE_ONLY"
    except Exception as exc:  # retain a complete multi-model audit report
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        report.setdefault("verification_level", "FAILED")
    report["status"] = "PASS" if not errors else "FAIL"
    return report, errors


def _run_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {name: manifest.get(name) for name in RUN_CONTRACT_FIELDS}


def audit_run(
    run_root: Path,
    *,
    entry: Mapping[str, Any],
    label: str,
    prompt_mode: str,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    report: dict[str, Any] = {"label": label, "run_root": str(run_root)}
    try:
        manifest = load_json(run_root / "run_manifest.json")
        completion = load_json(run_root / "COMPLETE.json")
        summary = load_json(run_root / "summary.json")
        run_fp = str(manifest.get("run_fingerprint", ""))
        recomputed = canonical_hash(_run_contract(manifest))
        report["declared_run_fingerprint"] = run_fp
        report["recomputed_run_fingerprint"] = recomputed
        report["run_contract_closes"] = run_fp == recomputed
        if not report["run_contract_closes"]:
            errors.append(f"{label}: run_manifest run_fingerprint does not close")
        if manifest.get("model_label") != label:
            errors.append(f"{label}: run manifest model_label mismatch")
        if manifest.get("prompt_mode") != prompt_mode:
            errors.append(f"{label}: run manifest prompt_mode mismatch")
        if manifest.get("model_path") != entry.get("canonical_path"):
            errors.append(f"{label}: run manifest model_path mismatch")
        fingerprints = manifest.get("fingerprints")
        if not isinstance(fingerprints, Mapping):
            raise ValueError("run manifest fingerprints is not an object")
        if fingerprints.get("model_fingerprint") != entry.get("model_fingerprint"):
            errors.append(f"{label}: run contract model fingerprint mismatch")
        model_record = manifest.get("model")
        if not isinstance(model_record, Mapping) or model_record.get(
            "model_fingerprint"
        ) != entry.get("model_fingerprint"):
            errors.append(f"{label}: embedded run model fingerprint mismatch")
        if isinstance(model_record, Mapping) and model_record.get(
            "canonical_path"
        ) != entry.get("canonical_path"):
            errors.append(f"{label}: embedded run model path mismatch")

        prompt = manifest.get("prompt")
        generation = manifest.get("generation")
        dataset = manifest.get("dataset")
        if prompt is None or canonical_hash(prompt) != fingerprints.get(
            "prompt_fingerprint"
        ):
            errors.append(f"{label}: prompt fingerprint does not close")
        if generation is None or canonical_hash(generation) != fingerprints.get(
            "generation_fingerprint"
        ):
            errors.append(f"{label}: generation fingerprint does not close")
        if not isinstance(dataset, Mapping) or dataset.get("sha256") != fingerprints.get(
            "dataset_fingerprint"
        ):
            errors.append(f"{label}: dataset fingerprint does not close")
        elif dataset.get("path"):
            dataset_path = Path(str(dataset["path"]))
            if not dataset_path.is_file():
                errors.append(
                    f"{label}: dataset path recorded in run manifest is missing: "
                    f"{dataset_path}"
                )
            elif sha256_file(dataset_path) != dataset.get("sha256"):
                errors.append(f"{label}: current dataset content differs from run manifest")

        if completion.get("run_fingerprint") != run_fp or summary.get(
            "run_fingerprint"
        ) != run_fp:
            errors.append(f"{label}: manifest/COMPLETE/summary run fingerprints differ")
        if completion.get("fingerprints") != fingerprints:
            errors.append(f"{label}: COMPLETE fingerprints differ from run manifest")
        if summary.get("fingerprints") != fingerprints:
            errors.append(f"{label}: summary fingerprints differ from run manifest")
        if completion.get("status") != "complete" or summary.get("status") != "complete":
            errors.append(f"{label}: public evaluation is not complete")

        predictions_path = run_root / "predictions.jsonl"
        if not predictions_path.is_file():
            errors.append(f"{label}: predictions.jsonl is missing")
        else:
            rows = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ids = [str(row.get("id")) for row in rows]
            if len(ids) != len(set(ids)):
                errors.append(f"{label}: predictions contain duplicate IDs")
            if completion.get("rows") is not None and int(completion["rows"]) != len(rows):
                errors.append(f"{label}: COMPLETE row count differs from predictions")
            for row in rows:
                if row.get("run_fingerprint") != run_fp:
                    errors.append(f"{label}: prediction row run fingerprint mismatch")
                    break
                for key, value in fingerprints.items():
                    if row.get(key) != value:
                        errors.append(f"{label}: prediction row {key} mismatch")
                        break
            report["prediction_rows"] = len(rows)
            report["unique_prediction_ids"] = len(set(ids))
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
    report["status"] = "PASS" if not errors else "FAIL"
    return report, errors


def _resolve_base_model(current: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    raw = Path(text).expanduser()
    candidates = [raw] if raw.is_absolute() else [current.parent / raw, raw]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve(strict=True)
    return None


def trace_export_chain(start: Path, target: Path, max_depth: int = 32) -> dict[str, Any]:
    """Follow local export ``base_model`` links until the fixed warm root."""

    current = start.resolve(strict=True)
    target = target.resolve(strict=True)
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for depth in range(max_depth + 1):
        key = str(current)
        if key in seen:
            errors.append(f"cycle in export base_model chain at {current}")
            break
        seen.add(key)
        try:
            identity_files(current)
        except Exception as exc:
            errors.append(f"invalid standalone chain node {current}: {exc}")
            break
        node: dict[str, Any] = {"depth": depth, "canonical_path": key}
        chain.append(node)
        if current == target:
            return {
                "status": "PASS",
                "reached_common_warm_start": True,
                "target": str(target),
                "chain": chain,
                "errors": [],
            }
        manifest_path = current / "export_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"chain terminated before warm root; no export manifest in {current}")
            break
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:
            errors.append(f"cannot parse {manifest_path}: {exc}")
            break
        node["export_manifest_sha256"] = sha256_file(manifest_path)
        node["base_model"] = manifest.get("base_model")
        if manifest.get("adapter_merged") is not True:
            errors.append(f"export does not attest adapter_merged=true: {manifest_path}")
            break
        dependencies = manifest.get("runtime_dependencies")
        if isinstance(dependencies, Mapping) and dependencies.get("peft_adapter") is not False:
            errors.append(f"export still declares a PEFT runtime dependency: {manifest_path}")
            break
        next_model = _resolve_base_model(current, manifest.get("base_model"))
        if next_model is None:
            errors.append(
                f"cannot resolve export base_model before reaching warm root: "
                f"{manifest.get('base_model')!r}"
            )
            break
        current = next_model
    else:
        errors.append(f"export base_model chain exceeded {max_depth} links")
    return {
        "status": "FAIL",
        "reached_common_warm_start": False,
        "target": str(target),
        "chain": chain,
        "errors": errors,
    }


def _origin_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve(strict=True)
    payload = load_json(resolved)
    fingerprint = str(payload.get("model_fingerprint", ""))
    revision = str(payload.get("revision", "")).strip().lower()
    if (
        SHA256_RE.fullmatch(fingerprint) is None
        or IMMUTABLE_REVISION_RE.fullmatch(revision) is None
    ):
        raise ValueError(
            "origin reference requires a 64-hex model_fingerprint and a full "
            "40- or 64-hex immutable revision"
        )
    return {
        **payload,
        "reference_path": str(resolved),
        "reference_sha256": sha256_file(resolved),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--prompt-mode", default="official")
    parser.add_argument("--expected-frozen", type=Path, required=True)
    parser.add_argument("--care-warm-start", type=Path, required=True)
    parser.add_argument("--baseline-warm-start", type=Path, required=True)
    parser.add_argument(
        "--origin-reference",
        type=Path,
        help=(
            "JSON fixed before training with model_fingerprint and immutable revision. "
            "Without it, raw-initial origin remains NOT_VERIFIED."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rehash-suite",
        action="store_true",
        help="Rehash every model artifact; otherwise the suite check is path/name/size only.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []
    suite = load_json(args.suite.resolve(strict=True))
    raw_models = suite.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("model suite models must be a list")
    labels = tuple(str(item.get("label")) for item in raw_models)
    if labels != EXPECTED_LABELS:
        errors.append(
            f"model suite labels/order mismatch: actual={labels}, expected={EXPECTED_LABELS}"
        )

    suite_reports: dict[str, Any] = {}
    entries: dict[str, dict[str, Any]] = {}
    for label in EXPECTED_LABELS:
        try:
            entry = suite_entry(suite, label)
            entries[label] = entry
            report, entry_errors = audit_suite_entry(entry, label=label, rehash=args.rehash_suite)
        except Exception as exc:
            report = {"label": label, "status": "FAIL"}
            entry_errors = [f"{label}: {type(exc).__name__}: {exc}"]
        suite_reports[label] = report
        errors.extend(entry_errors)

    if len(entries) == len(EXPECTED_LABELS):
        expected_suite_fp = canonical_hash(
            [(label, entries[label].get("model_fingerprint")) for label in EXPECTED_LABELS]
        )
        if suite.get("suite_fingerprint") != expected_suite_fp:
            errors.append("model_suite.json suite_fingerprint does not close")
    else:
        expected_suite_fp = None

    expected_frozen = str(args.expected_frozen.resolve(strict=True))
    if entries.get("frozen", {}).get("canonical_path") != expected_frozen:
        errors.append(
            f"frozen path mismatch: suite={entries.get('frozen', {}).get('canonical_path')} "
            f"expected={expected_frozen}"
        )

    run_reports: dict[str, Any] = {}
    for label in EXPECTED_LABELS:
        if label not in entries:
            continue
        run_report, run_errors = audit_run(
            args.eval_root / args.prompt_mode / label,
            entry=entries[label],
            label=label,
            prompt_mode=args.prompt_mode,
        )
        run_reports[label] = run_report
        errors.extend(run_errors)

    care_warm = fingerprint_directory(args.care_warm_start)
    baseline_warm = fingerprint_directory(args.baseline_warm_start)
    warm_start_equal = care_warm["model_fingerprint"] == baseline_warm["model_fingerprint"]
    if not warm_start_equal:
        errors.append("CARE and baseline warm-start checkpoints are not content-identical")

    ancestry_targets = {
        "clean_sft": args.baseline_warm_start,
        "recovery_sft": args.baseline_warm_start,
        "recovery_dpo": args.baseline_warm_start,
        "gkd_full": args.baseline_warm_start,
        "care_opd": args.care_warm_start,
    }
    ancestry: dict[str, Any] = {}
    for label, target in ancestry_targets.items():
        entry = entries.get(label)
        if entry is None:
            continue
        try:
            trace = trace_export_chain(Path(str(entry["canonical_path"])), target)
        except Exception as exc:
            trace = {
                "status": "FAIL",
                "reached_common_warm_start": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        ancestry[label] = trace
        errors.extend(f"{label}: {message}" for message in trace.get("errors", []))

    try:
        origin_reference = _origin_reference(args.origin_reference)
    except Exception as exc:
        origin_reference = None
        errors.append(f"invalid origin reference: {type(exc).__name__}: {exc}")
    frozen_actual_fp = suite_reports.get("frozen", {}).get("actual_model_fingerprint")
    if origin_reference is None:
        origin_status = "NOT_VERIFIED"
        warnings.append(
            "No immutable pre-training origin reference was supplied; frozen is only a local "
            "suite label, not a verified raw initial base."
        )
    elif not args.rehash_suite or frozen_actual_fp is None:
        origin_status = "NOT_VERIFIED"
        warnings.append("Origin reference supplied, but frozen content was not rehashed")
    elif frozen_actual_fp != origin_reference.get("model_fingerprint"):
        origin_status = "MISMATCH"
        errors.append("frozen model does not match the pre-training origin reference")
    else:
        # A locally supplied JSON can close the byte-level comparison, but cannot
        # independently prove that the reference itself was fixed before training
        # or obtained from the claimed upstream repository.  Keep that distinction
        # explicit instead of allowing a self-authored reference to certify itself.
        origin_status = "REFERENCE_MATCHED"
        warnings.append(
            "Frozen content matches the supplied immutable-revision reference, but "
            "the reference provenance was not independently verified by this audit."
        )

    local_lineage_status = "PASS" if not errors else "FAIL"
    if errors:
        status = "FAIL"
    elif not args.rehash_suite or origin_status != "INDEPENDENTLY_VERIFIED":
        status = "NOT_VERIFIED"
    else:
        status = "PASS"

    payload = {
        "schema_version": "care-opd-model-lineage-audit-v2",
        "status": status,
        "local_lineage_status": local_lineage_status,
        "origin_status": origin_status,
        "raw_initial_base_verified": (
            status == "PASS" and origin_status == "INDEPENDENTLY_VERIFIED"
        ),
        "suite_content_verification": (
            "REHASHED_CONTENT" if args.rehash_suite else "PATH_NAME_SIZE_ONLY"
        ),
        "suite_fingerprint": suite.get("suite_fingerprint"),
        "recomputed_suite_fingerprint": expected_suite_fp,
        "suite_models": suite_reports,
        "evaluation_runs": run_reports,
        "frozen_expected_path": expected_frozen,
        "origin_reference": origin_reference,
        "care_warm_start": care_warm,
        "baseline_warm_start": baseline_warm,
        "warm_start_content_identical": warm_start_equal,
        "branch_ancestry": ancestry,
        "errors": errors,
        "warnings": warnings,
        "interpretation": (
            "PASS requires rehashed content, all six suite/run contracts, common warm-start "
            "ancestry, and independently verified provenance for the pre-training origin. "
            "REFERENCE_MATCHED means the bytes match a supplied immutable-revision reference, "
            "but that local reference cannot certify its own provenance. NOT_VERIFIED means "
            "local consistency may hold but the raw-initial identity claim is not closed. "
            "This audit does not establish equivalence to an independently quantized Ollama/GGUF model."
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if status == "PASS":
        return 0
    if status == "NOT_VERIFIED":
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
