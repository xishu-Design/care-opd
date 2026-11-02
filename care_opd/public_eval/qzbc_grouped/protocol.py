#!/usr/bin/env python3
"""Fail-closed identities and split audit for the formal QZBC grouped suite."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_ORDER = (
    "frozen",
    "clean_sft",
    "recovery_sft",
    "recovery_dpo",
    "gkd_full",
    "care_opd",
)
EXPECTED_VARIANTS = (
    "clean",
    "bad_history",
    "source_outage",
    "evidence_conflict",
)
EXPECTED_REQUEST_ROWS = 4200
EXPECTED_SEED = 42
FORMAL_FAMILY_CLEAN_CONTRACT = {
    "seed_rows": 975,
    "request_rows": 3900,
    "kept_seed_ids_fingerprint": (
        "f68e74e04fdc91f5ecaf96acaff9c9a52d00c224fd44e12c09fa2e0c413dcd4d"
    ),
    "excluded_seed_ids_fingerprint": (
        "dd129dbec60d2b93157ed0be4f833133af09dc8a25777e1b03a8f11d4348ef1b"
    ),
    "request_ids_fingerprint": (
        "9ab70bd2237ba505203717b5d44af0f651c109bac69085da77bcff5f63a4cd3a"
    ),
}
FORMAL_FAMILY_CLEAN_TASK_COUNTS = {
    "doctor_ai_chat": 137,
    "doctor_review_text": 0,
    "patient_diet_guidance": 150,
    "patient_education": 150,
    "quiz_generation": 142,
    "student_ai_chat": 106,
    "student_analysis": 140,
    "student_quiz_explanation": 150,
}
PROTOCOL_VERSION = "care-opd-qzbc-grouped-identity-bound-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {resolved}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{resolved}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{resolved}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def jsonl_identity(
    path: Path, *, allowed_seed_ids: set[str] | None = None
) -> dict[str, Any]:
    """Validate/count/hash a JSONL file in one bounded-memory pass."""

    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    rows = 0
    with resolved.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{resolved}:{line_number}: invalid JSONL row: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number}: row must be an object")
            if allowed_seed_ids is not None:
                seed_id = str(value.get("seed_id") or "")
                if not seed_id or seed_id not in allowed_seed_ids:
                    raise ValueError(
                        f"{resolved}:{line_number}: seed_id is outside the declared split"
                    )
            rows += 1
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "rows": rows,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def jsonl_unique_string_field(path: Path, field: str) -> set[str]:
    values: set[str] = set()
    resolved = path.resolve(strict=True)
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = str(row.get(field) or "") if isinstance(row, Mapping) else ""
            if not value or value in values:
                raise ValueError(
                    f"{resolved}:{line_number}: empty/duplicate string field {field!r}"
                )
            values.add(value)
    return values


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _canonical_existing_path(raw: Any, *, context: str) -> Path:
    text = str(raw or "")
    if not text:
        raise ValueError(f"{context}: canonical path is absent")
    path = Path(text)
    resolved = path.resolve(strict=True)
    if str(resolved) != text:
        raise ValueError(f"{context}: path is not canonical: {text!r}")
    return resolved


def validate_model_suite(
    path: Path,
    *,
    rehash_labels: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate suite closure plus current checkpoint inventory.

    Every selected label in ``rehash_labels`` is fully SHA-256 rehashed. Other
    labels are checked by canonical path, size and the frozen mtime recorded by
    the suite; the suite's content-hash inventory must still close.
    """

    resolved = path.resolve(strict=True)
    payload = read_json(resolved)
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"model suite has no model list: {resolved}")
    labels = tuple(str(item.get("label") or "") for item in models)
    if labels != MODEL_ORDER:
        raise ValueError(f"model suite order changed: {labels}")
    rehash = set(rehash_labels)
    unknown = rehash - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"unknown model labels requested for rehash: {sorted(unknown)}")
    semantic: list[tuple[str, str]] = []
    normalized: list[dict[str, Any]] = []
    for entry in models:
        label = str(entry["label"])
        root = _canonical_existing_path(
            entry.get("canonical_path"), context=f"model {label}"
        )
        if not root.is_dir():
            raise ValueError(f"model is not a directory: {root}")
        raw_inventory = entry.get("inventory")
        if not isinstance(raw_inventory, list) or not raw_inventory:
            raise ValueError(f"model {label}: inventory is absent")
        inventory: list[dict[str, Any]] = []
        names: set[str] = set()
        for raw in raw_inventory:
            if not isinstance(raw, Mapping):
                raise ValueError(f"model {label}: invalid inventory row")
            name = str(raw.get("name") or "")
            if not name or name in names or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"model {label}: invalid/duplicate inventory name {name!r}")
            names.add(name)
            artifact = (root / name).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"model {label}: inventory path escapes root") from exc
            if not artifact.is_file():
                raise FileNotFoundError(f"model {label}: missing artifact {artifact}")
            size = int(raw.get("bytes", -1))
            expected_sha = str(raw.get("sha256") or "")
            if artifact.stat().st_size != size or len(expected_sha) != 64:
                raise ValueError(f"model {label}: artifact metadata changed: {name}")
            if raw.get("mtime_ns") is not None and artifact.stat().st_mtime_ns != int(
                raw["mtime_ns"]
            ):
                raise ValueError(f"model {label}: artifact mtime changed: {name}")
            if label in rehash and sha256_file(artifact) != expected_sha:
                raise ValueError(f"model {label}: artifact SHA-256 changed: {name}")
            inventory.append({"name": name, "bytes": size, "sha256": expected_sha})
        fingerprint = canonical_hash(inventory)
        if fingerprint != str(entry.get("model_fingerprint") or ""):
            raise ValueError(f"model {label}: model fingerprint does not close")
        semantic.append((label, fingerprint))
        normalized.append(
            {
                "label": label,
                "canonical_path": str(root),
                "model_fingerprint": fingerprint,
                "inventory_items": len(inventory),
                "inventory_fingerprint": canonical_hash(inventory),
                "identity_check": "full_sha256" if label in rehash else "size_mtime",
            }
        )
    suite_fingerprint = canonical_hash(semantic)
    if payload.get("suite_fingerprint") != suite_fingerprint:
        raise ValueError("model suite fingerprint does not close")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "suite_fingerprint": suite_fingerprint,
        "models": normalized,
        "payload": payload,
    }


def suite_entry(suite_identity: Mapping[str, Any], label: str) -> dict[str, Any]:
    matches = [item for item in suite_identity["models"] if item["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"model label {label!r} is not present exactly once")
    return dict(matches[0])


def _verify_recorded_file(
    recorded: Mapping[str, Any], *, context: str
) -> tuple[Path, int, str]:
    path = _canonical_existing_path(recorded.get("path"), context=context)
    rows = int(recorded.get("rows", -1))
    sha = str(recorded.get("sha256") or "")
    identity = jsonl_identity(path)
    if identity["sha256"] != sha:
        raise ValueError(f"{context}: SHA-256 changed")
    observed_rows = int(identity["rows"])
    if rows != observed_rows:
        raise ValueError(f"{context}: row count changed ({observed_rows} != {rows})")
    return path, rows, sha


def _seed_asset_identities(
    rows: Sequence[Mapping[str, Any]], *, split: str
) -> tuple[
    set[str],
    set[tuple[str, str, str]],
    dict[str, set[tuple[str, ...]]],
    set[str],
    dict[str, int],
    int,
]:
    ids: set[str] = set()
    assets: set[tuple[str, str, str]] = set()
    families_by_seed: dict[str, set[tuple[str, ...]]] = {}
    request_target_hashes: set[str] = set()
    task_counts: dict[str, int] = {}
    empty_asset_rows = 0
    for index, row in enumerate(rows, 1):
        item_id = str(row.get("id") or "")
        if not item_id or item_id in ids:
            raise ValueError(f"{split} seed row {index}: empty or duplicate id")
        if str(row.get("split") or "") != split:
            raise ValueError(f"{split} seed row {item_id}: top-level split mismatch")
        ids.add(item_id)
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{split} seed row {item_id}: metadata is absent")
        task = str(metadata.get("task_type") or "")
        if not task:
            raise ValueError(f"{split} seed row {item_id}: task_type is absent")
        task_counts[task] = task_counts.get(task, 0) + 1
        raw_assets = metadata.get("curated_assets")
        if not isinstance(raw_assets, list):
            raise ValueError(f"{split} seed row {item_id}: curated_assets is not a list")
        families_by_seed[item_id] = set()
        if not raw_assets:
            empty_asset_rows += 1
        for raw_asset in raw_assets:
            if not isinstance(raw_asset, Mapping):
                raise ValueError(f"{split} seed row {item_id}: invalid curated asset")
            identity = tuple(
                str(raw_asset.get(key) or "").strip()
                for key in ("asset_type", "asset_id", "source_record_id")
            )
            source_id = str(raw_asset.get("source_id") or "").strip()
            if not source_id or not identity[0] or not identity[1]:
                raise ValueError(
                    f"{split} seed row {item_id}: empty curated asset identity {identity}"
                )
            assets.add(identity)
            if identity[2]:
                family = ("source_record", source_id, identity[2])
            else:
                family = ("asset_fallback", source_id, identity[0], identity[1])
            families_by_seed[item_id].add(family)
        request_target_hashes.add(
            canonical_hash({"request": row.get("request"), "target": row.get("target")})
        )
    return (
        ids,
        assets,
        families_by_seed,
        request_target_hashes,
        dict(sorted(task_counts.items())),
        empty_asset_rows,
    )


def _verify_qzbc_source_assets(
    qzbc_root: Path, seed_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    root = qzbc_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"QZBC root is not a directory: {root}")
    contract = seed_manifest.get("qzbc_contract")
    if not isinstance(contract, Mapping) or not isinstance(contract.get("files"), Mapping):
        raise ValueError("seed manifest qzbc_contract/files is absent")
    observed_contract: dict[str, str] = {}
    contract_inventory: list[dict[str, Any]] = []
    for relative, expected_sha in sorted(contract["files"].items()):
        target = (root / str(relative)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("QZBC contract path escapes root") from exc
        observed_contract[str(relative)] = sha256_file(target)
        if observed_contract[str(relative)] != str(expected_sha):
            raise ValueError(f"QZBC contract file changed: {relative}")
        contract_inventory.append(
            {
                "path": str(target),
                "bytes": target.stat().st_size,
                "mtime_ns": target.stat().st_mtime_ns,
                "sha256": observed_contract[str(relative)],
            }
        )
    if canonical_hash(observed_contract) != str(contract.get("contract_fingerprint") or ""):
        raise ValueError("QZBC contract fingerprint does not close")
    grounding = seed_manifest.get("qzbc_grounding_assets")
    if not isinstance(grounding, Mapping) or not isinstance(grounding.get("files"), Mapping):
        raise ValueError("seed manifest qzbc_grounding_assets/files is absent")
    observed_assets: dict[str, dict[str, Any]] = {}
    grounding_inventory: list[dict[str, Any]] = []
    for relative, raw in sorted(grounding["files"].items()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid grounding asset record: {relative}")
        target = (root / str(relative)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("QZBC asset path escapes root") from exc
        observed = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
        if observed != {"bytes": int(raw.get("bytes", -1)), "sha256": str(raw.get("sha256") or "")}:
            raise ValueError(f"QZBC grounding asset changed: {relative}")
        observed_assets[str(relative)] = observed
        grounding_inventory.append(
            {
                "path": str(target),
                "bytes": target.stat().st_size,
                "mtime_ns": target.stat().st_mtime_ns,
                "sha256": observed["sha256"],
            }
        )
    return {
        "canonical_root": str(root),
        "contract_fingerprint": str(contract["contract_fingerprint"]),
        "contract_files_fingerprint": canonical_hash(observed_contract),
        "grounding_asset_files_fingerprint": canonical_hash(observed_assets),
        "contract_files": contract_inventory,
        "grounding_asset_files": grounding_inventory,
    }


def _validate_compiled_manifest(
    *,
    path: Path,
    split: str,
    seed_path: Path,
    seed_rows: int,
    contract_fingerprint: str,
    allowed_seed_ids: set[str],
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    manifest = read_json(resolved)
    if manifest.get("schema_version") != "care-opd-qzbc-data-v1":
        raise ValueError(f"{split} compiled manifest schema changed")
    source = _canonical_existing_path(
        manifest.get("source"), context=f"{split} compiled source"
    )
    if source != seed_path or int(manifest.get("source_rows", -1)) != seed_rows:
        raise ValueError(f"{split} compiled manifest is not bound to its seed output")
    if tuple(manifest.get("variants") or ()) != EXPECTED_VARIANTS:
        raise ValueError(f"{split} compiled variants changed")
    compiled_contract = manifest.get("qzbc_contract")
    if not isinstance(compiled_contract, Mapping) or str(
        compiled_contract.get("contract_fingerprint") or ""
    ) != contract_fingerprint:
        raise ValueError(f"{split} compiled QZBC contract changed")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError(f"{split} compiled counts are absent")
    output_names = {
        "sft": "qzbc_sft.jsonl",
        "clean_sft": "qzbc_clean_sft.jsonl",
        "recovery_sft": "qzbc_recovery_sft.jsonl",
        "stream_retention_sft": "qzbc_stream_retention_sft.jsonl",
        "clean_mixed_sft": "qzbc_clean_mixed_sft.jsonl",
        "on_policy_requests": "qzbc_on_policy_requests.jsonl",
        "host_overridden_retention": "qzbc_host_overridden_retention.jsonl",
    }
    if set(counts) != set(output_names):
        raise ValueError(f"{split} compiled output inventory changed")
    output_files: dict[str, dict[str, Any]] = {}
    for key, filename in output_names.items():
        target = (resolved.parent / filename).resolve(strict=True)
        identity = jsonl_identity(target, allowed_seed_ids=allowed_seed_ids)
        observed_rows = int(identity["rows"])
        if observed_rows != int(counts[key]):
            raise ValueError(f"{split} compiled {key} row count changed")
        output_files[key] = {
            **identity,
        }
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "mtime_ns": resolved.stat().st_mtime_ns,
        "sha256": sha256_file(resolved),
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "source_sha256": sha256_file(source),
        "source_rows": seed_rows,
        "on_policy_requests": int(counts.get("on_policy_requests", -1)),
        "outputs": output_files,
    }


def _verify_test_deterministic_recompile(
    *,
    qzbc_root: Path,
    seed_path: Path,
    compiled_test: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the test compiler outputs and require byte-identical JSONLs."""

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from care_opd.qzbc_dataset import compile_dataset

    with tempfile.TemporaryDirectory(
        prefix=".qzbc-test-recompile-",
        dir=Path(str(compiled_test["path"])).parent,
    ) as temporary:
        rebuilt_root = Path(temporary)
        compile_dataset(seed_path, qzbc_root, rebuilt_root)
        verified: dict[str, dict[str, Any]] = {}
        for key, expected in compiled_test["outputs"].items():
            filename = Path(str(expected["path"])).name
            rebuilt = jsonl_identity(rebuilt_root / filename)
            if (
                rebuilt["rows"] != expected["rows"]
                or rebuilt["sha256"] != expected["sha256"]
            ):
                raise ValueError(
                    f"test compiled output is not a deterministic qzbc_dataset rebuild: {key}"
                )
            verified[key] = {
                "rows": rebuilt["rows"],
                "sha256": rebuilt["sha256"],
            }
    return {
        "status": "PASS",
        "scope": "all seven test compiler JSONLs byte-identical after deterministic rebuild",
        "compiler": "care_opd.qzbc_dataset.compile_dataset",
        "outputs": verified,
        "outputs_fingerprint": canonical_hash(verified),
    }


def audit_data_contract(
    *,
    qzbc_root: Path,
    requests_path: Path,
    seed_manifest_path: Path,
    compiled_manifests: Mapping[str, Path],
    expected_seed: int = EXPECTED_SEED,
    expected_request_rows: int = EXPECTED_REQUEST_ROWS,
    expected_family_clean_contract: Mapping[str, Any] = FORMAL_FAMILY_CLEAN_CONTRACT,
    expected_family_clean_task_counts: Mapping[str, int] | None = (
        FORMAL_FAMILY_CLEAN_TASK_COUNTS
    ),
    verify_test_recompile: bool = True,
) -> dict[str, Any]:
    """Recompute the complete train/dev/test provenance and disjointness proof."""

    seed_manifest_resolved = seed_manifest_path.resolve(strict=True)
    seed_manifest = read_json(seed_manifest_resolved)
    if seed_manifest.get("schema_version") != "care-opd-qzbc-seeds-v2":
        raise ValueError("seed manifest schema changed")
    if int(seed_manifest.get("seed", -1)) != expected_seed:
        raise ValueError("seed manifest training/split seed changed")
    data_policy = seed_manifest.get("data_policy")
    if not isinstance(data_policy, Mapping) or data_policy.get(
        "knowledge_assets_are_exact_split_disjoint"
    ) is not True:
        raise ValueError("seed manifest does not declare exact asset split isolation")
    split_declaration = seed_manifest.get("grounding_asset_split")
    if not isinstance(split_declaration, Mapping) or (
        split_declaration.get("strategy") != "exact_asset_disjoint_deterministic_shuffle"
        or split_declaration.get("overlap_check") != "PASS"
        or int(split_declaration.get("seed", -1)) != expected_seed
    ):
        raise ValueError("seed manifest grounding split declaration changed")
    source_assets = _verify_qzbc_source_assets(qzbc_root, seed_manifest)
    outputs = seed_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("seed manifest outputs are absent")
    split_files: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, set[str]] = {}
    split_assets: dict[str, set[tuple[str, str, str]]] = {}
    split_families_by_seed: dict[str, dict[str, set[tuple[str, ...]]]] = {}
    split_request_target_hashes: dict[str, set[str]] = {}
    split_tasks: dict[str, dict[str, int]] = {}
    split_task_by_seed: dict[str, dict[str, str]] = {}
    split_visibility_by_seed: dict[str, dict[str, str]] = {}
    split_empty_asset_rows: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        raw_output = outputs.get(split)
        if not isinstance(raw_output, Mapping):
            raise ValueError(f"seed manifest output {split!r} is absent")
        path, rows, sha = _verify_recorded_file(raw_output, context=f"seed {split}")
        loaded = read_jsonl(path)
        (
            ids,
            assets,
            families_by_seed,
            request_target_hashes,
            tasks,
            empty_asset_rows,
        ) = _seed_asset_identities(loaded, split=split)
        split_ids[split] = ids
        split_assets[split] = assets
        split_families_by_seed[split] = families_by_seed
        split_request_target_hashes[split] = request_target_hashes
        split_tasks[split] = tasks
        split_task_by_seed[split] = {
            str(row["id"]): str(row["metadata"]["task_type"]) for row in loaded
        }
        split_visibility_by_seed[split] = {
            str(row["id"]): str(
                row["metadata"].get("deployment_visibility") or "model_visible"
            )
            for row in loaded
        }
        split_empty_asset_rows[split] = empty_asset_rows
        split_files[split] = {
            "path": str(path),
            "rows": rows,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha,
        }
    raw_all = outputs.get("all")
    if not isinstance(raw_all, Mapping):
        raise ValueError("seed manifest output 'all' is absent")
    all_path, all_rows, all_sha = _verify_recorded_file(raw_all, context="seed all")
    all_ids = jsonl_unique_string_field(all_path, "id")
    union_ids = split_ids["train"] | split_ids["dev"] | split_ids["test"]
    if (
        all_rows != sum(len(ids) for ids in split_ids.values())
        or all_ids != union_ids
    ):
        raise ValueError("seed all output is not the exact train/dev/test union")
    split_files["all"] = {
        "path": str(all_path),
        "rows": all_rows,
        "bytes": all_path.stat().st_size,
        "mtime_ns": all_path.stat().st_mtime_ns,
        "sha256": all_sha,
    }
    declared_split_counts = seed_manifest.get("split_counts")
    if not isinstance(declared_split_counts, Mapping):
        raise ValueError("seed manifest split_counts is absent")
    for split in ("train", "dev", "test"):
        if dict(declared_split_counts.get(split) or {}) != split_tasks[split]:
            raise ValueError(f"seed manifest split_counts changed for {split}")
    source_family_overlap_counts: dict[str, int] = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        id_overlap = split_ids[left] & split_ids[right]
        asset_overlap = split_assets[left] & split_assets[right]
        request_target_overlap = (
            split_request_target_hashes[left] & split_request_target_hashes[right]
        )
        if id_overlap:
            raise ValueError(f"seed ID overlap {left}/{right}: {sorted(id_overlap)[:5]}")
        if asset_overlap:
            raise ValueError(
                f"curated asset overlap {left}/{right}: {sorted(asset_overlap)[:5]}"
            )
        if request_target_overlap:
            raise ValueError(
                f"canonical request/target overlap {left}/{right}: "
                f"{sorted(request_target_overlap)[:5]}"
            )
        left_families = set().union(*split_families_by_seed[left].values())
        right_families = set().union(*split_families_by_seed[right].values())
        source_family_overlap_counts[f"{left}__{right}"] = len(
            left_families & right_families
        )
    if set(compiled_manifests) != {"train", "dev", "test"}:
        raise ValueError("compiled train/dev/test manifests are all required")
    contract_fingerprint = source_assets["contract_fingerprint"]
    compiled: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        compiled[split] = _validate_compiled_manifest(
            path=compiled_manifests[split],
            split=split,
            seed_path=Path(split_files[split]["path"]),
            seed_rows=int(split_files[split]["rows"]),
            contract_fingerprint=contract_fingerprint,
            allowed_seed_ids=split_ids[split],
        )
    deterministic_test_recompile = (
        _verify_test_deterministic_recompile(
            qzbc_root=qzbc_root,
            seed_path=Path(split_files["test"]["path"]),
            compiled_test=compiled["test"],
        )
        if verify_test_recompile
        else {
            "status": "TEST_FIXTURE_SKIP",
            "scope": "disabled only by direct library caller; formal CLI never disables it",
        }
    )

    requests = requests_path.resolve(strict=True)
    expected_requests_path = Path(
        compiled["test"]["outputs"]["on_policy_requests"]["path"]
    )
    if requests != expected_requests_path:
        raise ValueError("formal requests are not the audited test compiled output")
    request_rows = read_jsonl(requests)
    if len(request_rows) != expected_request_rows:
        raise ValueError(
            f"formal request cardinality changed: {len(request_rows)} != {expected_request_rows}"
        )
    request_ids: set[str] = set()
    variants_by_seed: dict[str, set[str]] = {}
    for index, row in enumerate(request_rows, 1):
        item_id = str(row.get("id") or "")
        seed_id = str(row.get("seed_id") or "")
        variant = str(row.get("variant") or "")
        if not item_id or item_id in request_ids:
            raise ValueError(f"request row {index}: empty or duplicate id")
        if seed_id not in split_ids["test"]:
            raise ValueError(f"request {item_id}: seed_id is not in the test seed split")
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"request {item_id}: unexpected variant {variant!r}")
        if str(row.get("contract_fingerprint") or "") != contract_fingerprint:
            raise ValueError(f"request {item_id}: QZBC contract fingerprint changed")
        expected_id = hashlib.sha256(
            f"{seed_id}|{variant}|{contract_fingerprint}".encode("utf-8")
        ).hexdigest()[:24]
        if item_id != expected_id:
            raise ValueError(f"request {item_id}: stable compiled ID formula changed")
        request_ids.add(item_id)
        variants_by_seed.setdefault(seed_id, set()).add(variant)
    bad_variant_sets = {
        seed_id: sorted(variants)
        for seed_id, variants in variants_by_seed.items()
        if variants != set(EXPECTED_VARIANTS)
    }
    if bad_variant_sets:
        raise ValueError(
            "test requests do not contain exactly one complete variant family per seed: "
            f"{list(bad_variant_sets.items())[:5]}"
        )
    if len(variants_by_seed) * len(EXPECTED_VARIANTS) != len(request_rows):
        raise ValueError("duplicate seed/variant request rows detected")
    if int(compiled["test"]["on_policy_requests"]) != len(request_rows):
        raise ValueError("test compiled manifest request count differs from formal requests")
    expected_model_visible_test_seeds = {
        seed_id
        for seed_id, visibility in split_visibility_by_seed["test"].items()
        if visibility != "host_overridden_retention"
    }
    if set(variants_by_seed) != expected_model_visible_test_seeds:
        raise ValueError(
            "test request seed set is not the exact model-visible test seed set"
        )

    blocking_families = set().union(
        *split_families_by_seed["train"].values(),
        *split_families_by_seed["dev"].values(),
    )
    family_clean_seed_ids = sorted(
        seed_id
        for seed_id, families in split_families_by_seed["test"].items()
        if not (families & blocking_families)
    )
    excluded_seed_ids = sorted(split_ids["test"] - set(family_clean_seed_ids))
    family_clean_request_ids = sorted(
        str(row["id"])
        for row in request_rows
        if str(row["seed_id"]) in set(family_clean_seed_ids)
    )
    family_clean_task_counts: dict[str, int] = {}
    for seed_id in family_clean_seed_ids:
        task = split_task_by_seed["test"][seed_id]
        family_clean_task_counts[task] = family_clean_task_counts.get(task, 0) + 1
    if expected_family_clean_task_counts is not None:
        observed_tasks_with_zeroes = {
            task: family_clean_task_counts.get(task, 0)
            for task in expected_family_clean_task_counts
        }
        if observed_tasks_with_zeroes != dict(expected_family_clean_task_counts):
            raise ValueError(
                "family-clean task composition changed: "
                f"{observed_tasks_with_zeroes} != {dict(expected_family_clean_task_counts)}"
            )
    expected_family_clean = dict(expected_family_clean_contract)
    observed_family_clean = {
        "seed_rows": len(family_clean_seed_ids),
        "request_rows": len(family_clean_request_ids),
        "kept_seed_ids_fingerprint": canonical_hash(family_clean_seed_ids),
        "excluded_seed_ids_fingerprint": canonical_hash(excluded_seed_ids),
        "request_ids_fingerprint": canonical_hash(family_clean_request_ids),
    }
    if observed_family_clean != expected_family_clean:
        raise ValueError(
            "family-clean result-blind filter changed: "
            f"observed={observed_family_clean}, expected={expected_family_clean}"
        )
    clean_family_union = set().union(
        *(split_families_by_seed["test"][seed_id] for seed_id in family_clean_seed_ids)
    )
    if clean_family_union & blocking_families:
        raise AssertionError("family-clean test still overlaps train/dev provenance families")

    deterministic = {
        "protocol_version": PROTOCOL_VERSION,
        "claim_scope": (
            "in-domain entity-held-out evaluation. The full view shares source-record "
            "provenance families across splits; a retrospective result-blind family-clean "
            "sensitivity view removes those test seeds. Neither is template-level or OOD evidence"
        ),
        "expected_seed": expected_seed,
        "expected_request_rows": expected_request_rows,
        "variants": list(EXPECTED_VARIANTS),
        "qzbc_source_assets": source_assets,
        "seed_manifest": {
            "path": str(seed_manifest_resolved),
            "bytes": seed_manifest_resolved.stat().st_size,
            "mtime_ns": seed_manifest_resolved.stat().st_mtime_ns,
            "sha256": sha256_file(seed_manifest_resolved),
            "schema_version": seed_manifest["schema_version"],
            "seed": int(seed_manifest["seed"]),
            "dev_ratio": seed_manifest.get("dev_ratio"),
            "test_ratio": seed_manifest.get("test_ratio"),
        },
        "seed_outputs": split_files,
        "compiled_manifests": compiled,
        "deterministic_test_recompile": deterministic_test_recompile,
        "requests": {
            "path": str(requests),
            "bytes": requests.stat().st_size,
            "mtime_ns": requests.stat().st_mtime_ns,
            "sha256": sha256_file(requests),
            "rows": len(request_rows),
            "unique_ids": len(request_ids),
            "unique_seed_ids": len(variants_by_seed),
        },
        "recomputed_disjointness": {
            "status": "PASS",
            "seed_ids_pairwise_disjoint": True,
            "curated_asset_identities_pairwise_disjoint": True,
            "canonical_request_target_hashes_pairwise_disjoint": True,
            "asset_identity_fields": ["asset_type", "asset_id", "source_record_id"],
            "holdout_level": "ENTITY_HOLDOUT_ONLY",
            "source_record_families_pairwise_disjoint": not any(
                source_family_overlap_counts.values()
            ),
            "source_record_family_overlap_counts": source_family_overlap_counts,
            "seed_counts": {split: len(split_ids[split]) for split in split_ids},
            "curated_asset_identity_counts": {
                split: len(split_assets[split]) for split in split_assets
            },
            "task_counts": split_tasks,
            "synthetic_only_seed_rows_with_empty_curated_assets": split_empty_asset_rows,
        },
        "family_clean_view": {
            "status": "PASS",
            "provenance": (
                "retrospective result-blind contamination-sensitivity view, frozen before "
                "the identity-bound rerun; not a preregistered independent benchmark"
            ),
            "filter": (
                "exclude a whole test seed if any result-blind provenance-family key occurs "
                "in train or dev; use (source_record,source_id,source_record_id), otherwise "
                "(asset_fallback,source_id,asset_type,asset_id)"
            ),
            **observed_family_clean,
            "excluded_seed_rows": len(excluded_seed_ids),
            "source_family_overlap_with_train_dev": 0,
            "task_counts": dict(sorted(family_clean_task_counts.items())),
            "kept_seed_ids": family_clean_seed_ids,
            "excluded_seed_ids": excluded_seed_ids,
            "request_ids": family_clean_request_ids,
        },
    }
    return {
        "schema_version": "care-opd-qzbc-data-audit-v1",
        **deterministic,
        "data_contract_fingerprint": canonical_hash(deterministic),
    }


def validate_frozen_data_audit_quick(
    audit_path: Path,
    *,
    requests_path: Path,
) -> dict[str, Any]:
    """Validate a prepared audit without rehashing multi-GB compiled outputs.

    Preparation and final suite summarization perform the full streaming audit.
    Per-model inference checks the audit closure, exact inference-request hash,
    and path/size/mtime of every other bound input.
    """

    payload = read_json(audit_path)
    if payload.get("schema_version") != "care-opd-qzbc-data-audit-v1":
        raise ValueError("prepared QZBC data audit schema changed")
    deterministic = {
        key: value
        for key, value in payload.items()
        if key not in {"schema_version", "data_contract_fingerprint"}
    }
    if canonical_hash(deterministic) != payload.get("data_contract_fingerprint"):
        raise ValueError("prepared QZBC data audit fingerprint does not close")

    records: list[Mapping[str, Any]] = []
    records.extend(payload["qzbc_source_assets"]["contract_files"])
    records.extend(payload["qzbc_source_assets"]["grounding_asset_files"])
    records.append(payload["seed_manifest"])
    records.extend(payload["seed_outputs"].values())
    for compiled in payload["compiled_manifests"].values():
        records.append(compiled)
        records.extend(compiled["outputs"].values())
    for record in records:
        path = _canonical_existing_path(record.get("path"), context="prepared data input")
        if (
            path.stat().st_size != int(record.get("bytes", -1))
            or path.stat().st_mtime_ns != int(record.get("mtime_ns", -1))
        ):
            raise ValueError(f"prepared data input changed: {path}")

    requests = requests_path.resolve(strict=True)
    recorded_requests = payload["requests"]
    if str(requests) != str(recorded_requests.get("path") or ""):
        raise ValueError("inference requests path differs from prepared data audit")
    identity = jsonl_identity(requests)
    for key in ("rows", "bytes", "mtime_ns", "sha256"):
        if identity[key] != recorded_requests.get(key):
            raise ValueError(f"inference requests {key} differs from prepared data audit")
    return payload


def artifact_inventory(paths: Sequence[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    root = relative_to.resolve()
    rows = []
    for path in paths:
        resolved = path.resolve(strict=True)
        try:
            name = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"artifact escapes output directory: {resolved}") from exc
        rows.append(
            {"name": name, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}
        )
    return sorted(rows, key=lambda row: row["name"])


def validate_artifact_inventory(root: Path, rows: Any) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    resolved_root = root.resolve()
    names: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            return False
        name = str(raw.get("name") or "")
        if not name or name in names or Path(name).is_absolute() or ".." in Path(name).parts:
            return False
        names.add(name)
        path = (resolved_root / name).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            return False
        if (
            not path.is_file()
            or path.stat().st_size != int(raw.get("bytes", -1))
            or sha256_file(path) != str(raw.get("sha256") or "")
        ):
            return False
    return True
