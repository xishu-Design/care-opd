#!/usr/bin/env python3
"""Prepare a result-blind SLAKE visual-evidence transfer benchmark.

Only Chinese, closed, binary visual questions are retained.  Each source
question always produces a clean task and a no-image counterfactual.  A
swapped-image task is added only when the same whitespace-normalized question
has an opposite binary answer on a different image in the same source split.

Gold annotations are written to a separate ``gold.private.jsonl`` file and
are never copied into ``tasks.jsonl``.  Selection is deterministic and does
not inspect model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SEED = 20260915
SELECTION_SCHEMA = "care_opd.visual_transfer.selection.v1"
TASK_SCHEMA = "care_opd.visual_transfer.task.v1"
GOLD_SCHEMA = "care_opd.visual_transfer.gold.v1"
DATASET_REPO = "BoKelvin/SLAKE"
DATASET_REVISION = "a9083ce6c34ac3ffb17671a605962924d8a8f9e9"
CONDITIONS = ("clean", "no_image", "swapped_image")
SPLIT_FILENAMES = {
    "train": ("train.json",),
    "validation": ("validation.json", "validate.json", "val.json"),
    "test": ("test.json",),
}

# SLAKE labels nominally marked CLOSED contain both binary and categorical
# answers.  This allowlist intentionally accepts only unambiguous affirmative
# or negative surface forms; organs, modalities, planes, colours, etc. remain
# excluded even though upstream labels them CLOSED.
YES_FORMS = frozenset(
    {
        "YES",
        "是",
        "是的",
        "包含",
        "包括",
        "存在",
        "有",
        "可以",
        "正常",
        "健康",
        "都是",
    }
)
NO_FORMS = frozenset(
    {
        "NO",
        "否",
        "不是",
        "不包含",
        "不包括",
        "不存在",
        "没有",
        "不可以",
        "不正常",
        "无异常",
    }
)
CONTEXT_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class PreparationError(ValueError):
    """Raised when SLAKE inputs violate the frozen benchmark contract."""


@dataclass(frozen=True)
class SourceItem:
    qid: str
    question: str
    normalized_question: str
    answer: str
    image_reference: str
    image_key: str
    source_image_id: str
    row_index: int


class _DisjointSet:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self._parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Lexical rooting makes the component stable regardless of edge order.
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root

    def components(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in self._parent:
            grouped[self.find(value)].append(value)
        return sorted((sorted(values) for values in grouped.values()), key=lambda x: x[0])


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return (
        "".join(_canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")
    )


def _manifest_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise PreparationError(f"{path}: invalid JSON") from exc
    if not isinstance(value, list) or not value:
        raise PreparationError(f"{path}: expected a non-empty JSON array")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise PreparationError(f"{path}: row {index} is not an object")
        rows.append(row)
    return rows


def _normalized_qid(value: Any) -> str:
    if value is None or isinstance(value, bool):
        raise PreparationError("eligible SLAKE row lacks a valid qid")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    result = unicodedata.normalize("NFKC", str(value)).strip()
    if not result:
        raise PreparationError("eligible SLAKE row has an empty qid")
    return result


def _normalize_question(value: Any) -> tuple[str, str]:
    if not isinstance(value, str):
        raise PreparationError("eligible SLAKE row lacks a string question")
    question = unicodedata.normalize("NFC", value).strip()
    normalized = re.sub(r"\s+", " ", question)
    if not normalized:
        raise PreparationError("eligible SLAKE row has an empty question")
    return question, normalized


def normalize_binary_answer(value: Any) -> str | None:
    """Map the explicit binary SLAKE surface-form allowlist to YES/NO."""

    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        return None
    upper = normalized.upper()
    if upper in YES_FORMS or normalized in YES_FORMS:
        return "YES"
    if upper in NO_FORMS or normalized in NO_FORMS:
        return "NO"
    return None


def _image_reference(row: Mapping[str, Any]) -> str:
    for key in ("img_name", "image_path", "image", "img"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\\", "/")
    raise PreparationError("eligible SLAKE row lacks an image reference")


def _safe_relative_image(reference: str) -> PurePosixPath:
    if re.match(r"^[A-Za-z]:", reference):
        raise PreparationError(f"absolute image reference is forbidden: {reference!r}")
    path = PurePosixPath(reference)
    if path.is_absolute() or not path.parts:
        raise PreparationError(f"absolute/empty image reference is forbidden: {reference!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PreparationError(f"unsafe image reference: {reference!r}")
    return path


def _image_key(reference: str) -> str:
    relative = _safe_relative_image(reference)
    parts = relative.parts[1:] if relative.parts[0].lower() == "imgs" else relative.parts
    if not parts:
        raise PreparationError(f"image reference has no filename: {reference!r}")
    return PurePosixPath(*parts).as_posix()


def _source_image_id(row: Mapping[str, Any], image_key: str) -> str:
    value = row.get("img_id")
    if value is None or isinstance(value, bool) or not str(value).strip():
        return image_key
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return unicodedata.normalize("NFKC", str(value)).strip()


def _explicit_image_id(row: Mapping[str, Any]) -> str | None:
    """Return an upstream image identifier without inventing a fallback value."""

    value = row.get("img_id")
    if value is None or isinstance(value, bool) or not str(value).strip():
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return unicodedata.normalize("NFKC", str(value)).strip()


def _extract_items(rows: Sequence[Mapping[str, Any]]) -> tuple[list[SourceItem], dict[str, Any]]:
    exclusions: Counter[str] = Counter()
    candidates: list[SourceItem] = []
    for row_index, row in enumerate(rows):
        if str(row.get("base_type", "")).strip().lower() != "vqa":
            exclusions["base_type_not_vqa"] += 1
            continue
        if str(row.get("answer_type", "")).strip().upper() != "CLOSED":
            exclusions["answer_type_not_closed"] += 1
            continue
        if str(row.get("q_lang", "")).strip().lower() != "zh":
            exclusions["question_language_not_zh"] += 1
            continue
        answer = normalize_binary_answer(row.get("answer"))
        if answer is None:
            exclusions["closed_answer_not_binary_allowlist"] += 1
            continue
        qid = _normalized_qid(row.get("qid"))
        question, normalized_question = _normalize_question(row.get("question"))
        reference = _image_reference(row)
        key = _image_key(reference)
        candidates.append(
            SourceItem(
                qid=qid,
                question=question,
                normalized_question=normalized_question,
                answer=answer,
                image_reference=reference,
                image_key=key,
                source_image_id=_source_image_id(row, key),
                row_index=row_index,
            )
        )

    # Determinism must not depend on upstream JSON ordering.
    candidates.sort(
        key=lambda item: (
            item.qid,
            item.image_key,
            item.normalized_question,
            item.answer,
            item.row_index,
        )
    )
    by_qid: dict[str, SourceItem] = {}
    duplicate_qid_rows = 0
    for item in candidates:
        previous = by_qid.get(item.qid)
        if previous is None:
            by_qid[item.qid] = item
            continue
        identity = (item.image_key, item.normalized_question, item.answer)
        old_identity = (previous.image_key, previous.normalized_question, previous.answer)
        if identity != old_identity:
            raise PreparationError(f"qid {item.qid!r} is bound to conflicting records")
        duplicate_qid_rows += 1

    by_image_question: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in by_qid.values():
        by_image_question[(item.image_key, item.normalized_question)].add(item.answer)
    conflicts = [
        (image, question, sorted(answers))
        for (image, question), answers in by_image_question.items()
        if len(answers) > 1
    ]
    if conflicts:
        image, question, answers = conflicts[0]
        raise PreparationError(
            "same-image/same-question answer conflict: "
            f"image={image!r} question={question!r} answers={answers!r}"
        )

    # Explicitly deduplicate same image/question/answer rows with distinct qids.
    content_seen: dict[tuple[str, str, str], SourceItem] = {}
    deduplicated: list[SourceItem] = []
    duplicate_content_rows = 0
    for item in sorted(by_qid.values(), key=lambda value: (value.qid, value.row_index)):
        key = (item.image_key, item.normalized_question, item.answer)
        if key in content_seen:
            duplicate_content_rows += 1
            continue
        content_seen[key] = item
        deduplicated.append(item)

    if not deduplicated:
        raise PreparationError("no eligible Chinese closed binary SLAKE questions")
    return deduplicated, {
        "source_rows": len(rows),
        "eligible_before_deduplication": len(candidates),
        "eligible_after_deduplication": len(deduplicated),
        "duplicate_qid_rows_removed": duplicate_qid_rows,
        "duplicate_content_rows_removed": duplicate_content_rows,
        "same_image_question_conflicts": 0,
        "exclusions": dict(sorted(exclusions.items())),
    }


def _find_split_file(dataset_root: Path, split: str) -> Path | None:
    matches = [dataset_root / name for name in SPLIT_FILENAMES[split] if (dataset_root / name).is_file()]
    if not matches:
        return None
    return matches[0].resolve()


def _source_provenance(dataset_root: Path, source: Path) -> dict[str, Any]:
    """Verify the source against our downloader manifest when one is present.

    A local filename alone is not evidence that bytes came from the expected HF
    commit.  Without a valid asset manifest we therefore record an unverified
    local copy, while preserving the expected upstream identity separately.
    """

    source_hash = _sha256_file(source)
    asset_manifest_path = dataset_root / "slake_assets_manifest.json"
    base: dict[str, Any] = {
        "expected_repo_id": DATASET_REPO,
        "expected_upstream_revision": DATASET_REVISION,
        "source_json": str(source),
        "source_json_sha256": source_hash,
        "verification_status": "UNVERIFIED_LOCAL_COPY",
        "verified_source_commit": None,
        "asset_manifest": None,
        "asset_manifest_sha256": None,
    }
    if not asset_manifest_path.is_file():
        return base
    try:
        asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(
            f"invalid SLAKE asset manifest: {asset_manifest_path}"
        ) from exc
    if not isinstance(asset_manifest, dict):
        raise PreparationError(f"invalid SLAKE asset manifest object: {asset_manifest_path}")
    if asset_manifest.get("status") != "COMPLETE":
        raise PreparationError("SLAKE asset manifest is not COMPLETE")
    if asset_manifest.get("repo_id") != DATASET_REPO:
        raise PreparationError("SLAKE asset manifest repo_id mismatch")
    if asset_manifest.get("resolved_commit") != DATASET_REVISION:
        raise PreparationError("SLAKE asset manifest resolved_commit mismatch")
    downloaded = asset_manifest.get("downloaded_files")
    if not isinstance(downloaded, list):
        raise PreparationError("SLAKE asset manifest downloaded_files is invalid")
    try:
        relative = source.relative_to(dataset_root).as_posix()
    except ValueError:
        relative = ""
    matching = [
        entry
        for entry in downloaded
        if isinstance(entry, dict) and entry.get("path") == relative
    ]
    base.update(
        {
            "asset_manifest": str(asset_manifest_path.resolve()),
            "asset_manifest_sha256": _sha256_file(asset_manifest_path),
        }
    )
    if not matching:
        # A custom source-json can still be used, but it must not inherit the
        # pinned-origin claim of unrelated files in the same directory.
        return base
    if len(matching) != 1 or matching[0].get("sha256") != source_hash:
        raise PreparationError("source JSON does not match the frozen SLAKE asset manifest")
    expected_size = matching[0].get("size_bytes")
    if expected_size != source.stat().st_size:
        raise PreparationError("source JSON size does not match the frozen asset manifest")
    base.update(
        {
            "verification_status": "VERIFIED_PINNED_HF_ASSET",
            "verified_source_commit": DATASET_REVISION,
        }
    )
    return base


def _identity_aliases(
    row: Mapping[str, Any],
    *,
    dataset_root: Path,
    images_root: Path,
    hash_cache: dict[str, str | None],
) -> tuple[str, set[str], str | None]:
    """Build conservative image identities from path, upstream id, and file bytes.

    Path and explicit ``img_id`` aliases are always available.  A raw-file SHA-256
    alias is added when the image exists.  Failure to resolve an image is recorded
    by the caller rather than hidden; selected benchmark images are still checked
    strictly later by :func:`_resolve_image`.
    """

    reference = _image_reference(row)
    key = _image_key(reference)
    aliases = {f"img_name:{key}"}
    explicit_id = _explicit_image_id(row)
    if explicit_id is not None:
        aliases.add(f"img_id:{explicit_id}")
    if key not in hash_cache:
        try:
            path = _resolve_image(
                reference, dataset_root=dataset_root, images_root=images_root
            )
            hash_cache[key] = _sha256_file(path)
        except PreparationError:
            hash_cache[key] = None
    image_sha = hash_cache[key]
    if image_sha is not None:
        aliases.add(f"sha256:{image_sha}")
    return key, aliases, image_sha


def audit_split_image_overlap(
    dataset_root: Path,
    *,
    images_root: Path | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """Describe official split overlap and return validation image aliases.

    SLAKE's published JSON files are not image-disjoint.  This experiment does
    not silently relabel them as such: validation remains the pilot split and
    formal evaluation derives ``official_test_minus_validation_images`` by a
    fixed, output-independent exclusion.  Train overlap is disclosed, but it is
    not a blocker because this mechanism experiment performs no fitting and does
    not use official train examples for prompt or threshold optimization.
    """

    image_root = (images_root or (dataset_root / "imgs")).resolve()
    files = {split: _find_split_file(dataset_root, split) for split in SPLIT_FILENAMES}
    missing = sorted(split for split, path in files.items() if path is None)
    hash_cache: dict[str, str | None] = {}
    split_records: dict[str, dict[str, set[str]]] = {}
    split_file_hashes: dict[str, str | None] = {}
    unresolved: dict[str, list[str]] = {}
    for split, path in files.items():
        if path is None:
            split_records[split] = {}
            split_file_hashes[split] = None
            unresolved[split] = []
            continue
        split_file_hashes[split] = _sha256_file(path)
        records: dict[str, set[str]] = defaultdict(set)
        unresolved_keys: set[str] = set()
        for row in _load_json_rows(path):
            key, aliases, image_sha = _identity_aliases(
                row,
                dataset_root=dataset_root,
                images_root=image_root,
                hash_cache=hash_cache,
            )
            records[key].update(aliases)
            if image_sha is None:
                unresolved_keys.add(key)
        split_records[split] = dict(records)
        unresolved[split] = sorted(unresolved_keys)

    def matched_right_keys(left: str, right: str) -> list[str]:
        left_aliases = {
            alias for aliases in split_records[left].values() for alias in aliases
        }
        return sorted(
            key
            for key, aliases in split_records[right].items()
            if aliases & left_aliases
        )

    pairwise: dict[str, list[str]] = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        pairwise[f"{left}__{right}"] = matched_right_keys(left, right)

    validation_aliases = {
        alias
        for aliases in split_records["validation"].values()
        for alias in aliases
    }
    status = (
        "RECORDED_WITH_MISSING_SPLITS"
        if missing
        else "RECORDED_OFFICIAL_SPLIT_OVERLAP"
    )
    audit = {
        "status": status,
        "policy": {
            "validation": "official validation used only for pilot selection",
            "test": (
                "official test with all validation-overlapping images removed before "
                "eligibility filtering or pairing"
            ),
            "train": (
                "overlap disclosed as a warning, not a blocker; official train is not "
                "used for fitting, prompting, threshold selection, or model selection"
            ),
            "forbidden_claim": "the official SLAKE splits are image-disjoint",
        },
        "missing_splits": missing,
        "split_files": {
            split: str(path) if path else None for split, path in files.items()
        },
        "split_json_sha256": split_file_hashes,
        "unique_image_counts": {
            split: len(records) for split, records in split_records.items()
        },
        "unresolved_image_counts": {
            split: len(values) for split, values in unresolved.items()
        },
        "unresolved_image_examples": {
            split: values[:20] for split, values in unresolved.items() if values
        },
        "pairwise_overlap_counts": {
            pair: len(values) for pair, values in pairwise.items()
        },
        "pairwise_overlap_image_keys": pairwise,
        "train_overlap_is_nonblocking_warning": bool(
            pairwise["train__validation"] or pairwise["train__test"]
        ),
    }
    return audit, validation_aliases


def _exclude_validation_images_from_test(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_aliases: set[str],
    dataset_root: Path,
    images_root: Path,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Apply the frozen formal-test exclusion before all content filtering."""

    hash_cache: dict[str, str | None] = {}
    kept: list[Mapping[str, Any]] = []
    excluded_keys: set[str] = set()
    excluded_explicit_ids: set[str] = set()
    excluded_qids: list[str] = []
    match_basis: Counter[str] = Counter()
    for row in rows:
        key, aliases, _ = _identity_aliases(
            row,
            dataset_root=dataset_root,
            images_root=images_root,
            hash_cache=hash_cache,
        )
        shared = aliases & validation_aliases
        if not shared:
            kept.append(row)
            continue
        excluded_keys.add(key)
        explicit_id = _explicit_image_id(row)
        if explicit_id is not None:
            excluded_explicit_ids.add(explicit_id)
        raw_qid = row.get("qid")
        if raw_qid is not None and not isinstance(raw_qid, bool):
            excluded_qids.append(_normalized_qid(raw_qid))
        for prefix in ("img_name:", "img_id:", "sha256:"):
            if any(alias.startswith(prefix) for alias in shared):
                match_basis[prefix[:-1]] += 1

    # Prove the derived rows are disjoint from validation under every identity
    # available to this build.  This is a property of our derived slice, not a
    # claim about the upstream official split.
    remaining_overlap: list[str] = []
    for row in kept:
        key, aliases, _ = _identity_aliases(
            row,
            dataset_root=dataset_root,
            images_root=images_root,
            hash_cache=hash_cache,
        )
        if aliases & validation_aliases:
            remaining_overlap.append(key)
    if remaining_overlap:
        raise PreparationError(
            "internal error: formal-test validation image exclusion was incomplete: "
            f"{sorted(set(remaining_overlap))[:10]}"
        )
    if not kept:
        raise PreparationError(
            "formal test is empty after removing validation-overlapping images"
        )
    return kept, {
        "rule": (
            "remove official-test rows whose image matches any validation image by "
            "normalized img_name, explicit img_id, or resolvable file SHA-256"
        ),
        "selection_independence": (
            "fixed metadata/image-identity rule applied before eligibility filtering, "
            "pairing, prompting, inference, and output inspection"
        ),
        "source_rows_before_exclusion": len(rows),
        "source_rows_after_exclusion": len(kept),
        "excluded_row_count": len(rows) - len(kept),
        "excluded_unique_image_count": len(excluded_keys),
        "excluded_image_keys": sorted(excluded_keys),
        "excluded_explicit_image_ids": sorted(excluded_explicit_ids),
        "excluded_qids": sorted(excluded_qids),
        "match_basis_row_counts": dict(sorted(match_basis.items())),
        "remaining_validation_overlap_count": 0,
    }


def _resolve_image(
    reference: str,
    *,
    dataset_root: Path,
    images_root: Path,
) -> Path:
    relative = _safe_relative_image(reference)
    roots = []
    for value in (dataset_root, images_root, dataset_root / "imgs"):
        resolved = value.resolve()
        if resolved not in roots and resolved.exists():
            roots.append(resolved)
    if not roots:
        raise PreparationError("no existing image search root")

    stripped = PurePosixPath(*relative.parts[1:]) if relative.parts[0].lower() == "imgs" else relative
    raw_candidates = [
        dataset_root / Path(*relative.parts),
        dataset_root / "imgs" / Path(*stripped.parts),
        images_root / Path(*stripped.parts),
        images_root / Path(*relative.parts),
    ]
    matches: list[Path] = []
    for candidate in raw_candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if not resolved.is_file():
            continue
        if not any(_is_within(resolved, root) for root in roots):
            raise PreparationError(f"resolved image escapes approved roots: {reference!r}")
        if resolved not in matches:
            matches.append(resolved)
    if not matches:
        raise PreparationError(f"image file not found for {reference!r}")
    if len(matches) > 1:
        hashes = {_sha256_file(path) for path in matches}
        if len(hashes) != 1:
            raise PreparationError(
                f"ambiguous image reference resolves to different files: {reference!r}"
            )
        matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _stable_rank(seed: int, split: str, qid: str) -> str:
    return hashlib.sha256(f"{seed}|select|{split}|{qid}".encode("utf-8")).hexdigest()


def _select_counterpart(
    item: SourceItem,
    index: Mapping[tuple[str, str], Sequence[SourceItem]],
    *,
    image_sha256_by_key: Mapping[str, str],
    seed: int,
    split: str,
) -> SourceItem | None:
    opposite = "NO" if item.answer == "YES" else "YES"
    candidates = [
        candidate
        for candidate in index.get((item.normalized_question, opposite), ())
        if candidate.image_key != item.image_key
        and image_sha256_by_key[candidate.image_key]
        != image_sha256_by_key[item.image_key]
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{seed}|swap|{split}|{item.qid}|{candidate.qid}|{candidate.image_key}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )


def _context_code(seed: int, split: str, family_id: str, occupied: set[str]) -> str:
    attempt = 0
    while True:
        material = hashlib.sha256(
            f"{seed}|context|{split}|{family_id}|{attempt}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(material[:8], "big"))
        code = "".join(rng.choice(CONTEXT_ALPHABET) for _ in range(8))
        if code not in occupied:
            occupied.add(code)
            return code
        attempt += 1


def _task_id(split: str, family_id: str, condition: str, counterpart_qid: str | None) -> str:
    digest = hashlib.sha256(
        f"{split}|{family_id}|{condition}|{counterpart_qid or ''}".encode("utf-8")
    ).hexdigest()[:20]
    return f"slake-{split}-{digest}"


def build_outputs(
    *,
    dataset_root: Path,
    split: str,
    source_json: Path | None = None,
    images_root: Path | None = None,
    max_items: int = 0,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise PreparationError("split must be validation (pilot) or test (formal)")
    if max_items < 0:
        raise PreparationError("max_items must be zero (all) or positive")
    dataset_root = dataset_root.resolve(strict=True)
    if not dataset_root.is_dir():
        raise PreparationError(f"dataset root is not a directory: {dataset_root}")
    source = source_json.resolve(strict=True) if source_json else _find_split_file(dataset_root, split)
    if source is None:
        raise PreparationError(f"could not find the {split} source JSON under {dataset_root}")
    if not source.is_file() or not _is_within(source, dataset_root):
        raise PreparationError("source JSON must be a file inside dataset_root")
    image_root = (images_root or (dataset_root / "imgs")).resolve()
    source_provenance = _source_provenance(dataset_root, source)

    overlap_audit, validation_aliases = audit_split_image_overlap(
        dataset_root, images_root=image_root
    )
    source_rows = _load_json_rows(source)
    derived_test_audit: dict[str, Any] | None = None
    effective_split = split
    if split == "test":
        if _find_split_file(dataset_root, "validation") is None:
            raise PreparationError(
                "formal test preparation requires validation.json so validation-overlapping "
                "images can be removed deterministically"
            )
        source_rows, derived_test_audit = _exclude_validation_images_from_test(
            source_rows,
            validation_aliases=validation_aliases,
            dataset_root=dataset_root,
            images_root=image_root,
        )
        effective_split = "official_test_minus_validation_images"
    eligible, dedup_audit = _extract_items(source_rows)

    # Resolve and hash every eligible image before counterpart construction.
    # Distinct filenames that contain identical bytes are the same bootstrap
    # unit and may never be used as an apparent visual swap.
    image_cache: dict[str, tuple[Path, str]] = {}
    for item in eligible:
        if item.image_key not in image_cache:
            path = _resolve_image(
                item.image_reference,
                dataset_root=dataset_root,
                images_root=image_root,
            )
            image_cache[item.image_key] = (path, _sha256_file(path))

    by_raw_image_question: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in eligible:
        image_sha = image_cache[item.image_key][1]
        by_raw_image_question[(image_sha, item.normalized_question)].add(item.answer)
    raw_conflicts = [
        (image_sha, question, sorted(answers))
        for (image_sha, question), answers in by_raw_image_question.items()
        if len(answers) > 1
    ]
    if raw_conflicts:
        image_sha, question, answers = raw_conflicts[0]
        raise PreparationError(
            "byte-identical-image/same-question answer conflict: "
            f"sha256={image_sha} question={question!r} answers={answers!r}"
        )

    raw_seen: dict[tuple[str, str, str], SourceItem] = {}
    raw_deduplicated: list[SourceItem] = []
    raw_duplicate_rows = 0
    for item in eligible:
        identity = (
            image_cache[item.image_key][1],
            item.normalized_question,
            item.answer,
        )
        if identity in raw_seen:
            raw_duplicate_rows += 1
            continue
        raw_seen[identity] = item
        raw_deduplicated.append(item)
    eligible = raw_deduplicated
    dedup_audit.update(
        {
            "duplicate_raw_image_content_rows_removed": raw_duplicate_rows,
            "byte_identical_image_question_conflicts": 0,
            "eligible_after_raw_image_deduplication": len(eligible),
        }
    )

    image_sha256_by_key = {
        key: value[1] for key, value in image_cache.items()
    }
    counterpart_index: dict[tuple[str, str], list[SourceItem]] = defaultdict(list)
    for item in eligible:
        counterpart_index[(item.normalized_question, item.answer)].append(item)
    for values in counterpart_index.values():
        values.sort(key=lambda item: (item.qid, item.image_key))

    selected = sorted(
        eligible,
        key=lambda item: (_stable_rank(seed, effective_split, item.qid), item.qid),
    )
    if max_items:
        selected = selected[:max_items]
    if not selected:
        raise PreparationError("selection produced no items")

    selected_with_counterpart: list[tuple[SourceItem, SourceItem | None]] = []
    graph = _DisjointSet()
    sha_anchor: dict[str, str] = {}
    for item in selected:
        counterpart = _select_counterpart(
            item,
            counterpart_index,
            image_sha256_by_key=image_sha256_by_key,
            seed=seed,
            split=effective_split,
        )
        graph.add(item.image_key)
        item_sha = image_sha256_by_key[item.image_key]
        if item_sha in sha_anchor:
            graph.union(item.image_key, sha_anchor[item_sha])
        else:
            sha_anchor[item_sha] = item.image_key
        if counterpart is not None:
            graph.union(item.image_key, counterpart.image_key)
            counterpart_sha = image_sha256_by_key[counterpart.image_key]
            if counterpart_sha in sha_anchor:
                graph.union(counterpart.image_key, sha_anchor[counterpart_sha])
            else:
                sha_anchor[counterpart_sha] = counterpart.image_key
        selected_with_counterpart.append((item, counterpart))

    components = graph.components()
    image_to_cluster: dict[str, str] = {}
    component_rows: list[dict[str, Any]] = []
    for images in components:
        cluster_id = "slake-imgcomp-" + hashlib.sha256(
            "\n".join(images).encode("utf-8")
        ).hexdigest()[:16]
        for image in images:
            image_to_cluster[image] = cluster_id
        component_rows.append({"cluster_id": cluster_id, "image_ids": images})

    def resolved_image(item: SourceItem) -> tuple[Path, str]:
        return image_cache[item.image_key]

    tasks: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    occupied_contexts: set[str] = set()
    condition_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    for item, counterpart in selected_with_counterpart:
        context = _context_code(seed, effective_split, item.qid, occupied_contexts)
        cluster_id = image_to_cluster[item.image_key]
        clean_path, clean_hash = resolved_image(item)
        variants: list[tuple[str, SourceItem | None, str]] = [
            ("clean", item, item.answer),
            ("no_image", None, "ABSTAIN"),
        ]
        if counterpart is not None:
            variants.append(("swapped_image", counterpart, counterpart.answer))
        for condition, visible_item, answer in variants:
            counterpart_qid = counterpart.qid if condition == "swapped_image" and counterpart else None
            task_id = _task_id(effective_split, item.qid, condition, counterpart_qid)
            if visible_item is None:
                image_path: str | None = None
                image_sha256: str | None = None
                visible_image_id = item.source_image_id
            elif condition == "clean":
                image_path = str(clean_path)
                image_sha256 = clean_hash
                visible_image_id = item.source_image_id
            else:
                swapped_path, swapped_hash = resolved_image(visible_item)
                image_path = str(swapped_path)
                image_sha256 = swapped_hash
                visible_image_id = visible_item.source_image_id
            tasks.append(
                {
                    "schema_version": TASK_SCHEMA,
                    "task_id": task_id,
                    "family_id": item.qid,
                    "cluster_id": cluster_id,
                    "condition": condition,
                    "question": item.question,
                    "image_path": image_path,
                    "image_sha256": image_sha256,
                    "context_value": context,
                    "dataset_qid": item.qid,
                    "source_image_id": visible_image_id,
                }
            )
            gold.append(
                {
                    "schema_version": GOLD_SCHEMA,
                    "task_id": task_id,
                    "family_id": item.qid,
                    "cluster_id": cluster_id,
                    "condition": condition,
                    "answer": answer,
                    "original_answer": item.answer,
                    "context_value": context,
                    "counterpart_qid": counterpart_qid,
                }
            )
            condition_counts[condition] += 1
            answer_counts[answer] += 1

    task_ids = [row["task_id"] for row in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise AssertionError("internal error: duplicate task_id")
    if [row["task_id"] for row in gold] != task_ids:
        raise AssertionError("internal error: task/gold order mismatch")
    if any(key in row for row in tasks for key in ("answer", "gold", "original_answer")):
        raise AssertionError("internal error: gold annotation leaked into tasks")

    manifest = {
        "schema_version": SELECTION_SCHEMA,
        "status": "FROZEN_RESULT_BLIND_SELECTION",
        "dataset": {
            **source_provenance,
            "source_split": split,
            "split": effective_split,
            "split_role": "pilot" if split == "validation" else "formal",
            "dataset_root": str(dataset_root),
            "images_root": str(image_root),
            "boundary": (
                "official validation pilot"
                if split == "validation"
                else (
                    "derived subset of the official public test split after a fixed removal "
                    "of validation-overlapping images; not an upstream image-disjoint split"
                )
            ),
        },
        "selection": {
            "seed": seed,
            "max_items": max_items,
            "max_items_zero_means_all": True,
            "ranking": "sha256(seed|select|effective_split|qid)",
            "filters": {
                "base_type": "vqa",
                "answer_type": "CLOSED",
                "q_lang": "zh",
                "answer_allowlist_yes": sorted(YES_FORMS),
                "answer_allowlist_no": sorted(NO_FORMS),
            },
            "selected_families": len(selected),
            "eligible_families": len(eligible),
        },
        "counterfactual_contract": {
            "conditions": list(CONDITIONS),
            "clean_answer": "normalized source YES/NO",
            "no_image_answer": "ABSTAIN",
            "swapped_image_rule": (
                "same split and whitespace-normalized question, different image, opposite "
                "normalized label; visible image and its source label are rebound together"
            ),
            "unmatched_questions_do_not_receive_swapped_image": True,
            "swapped_families": condition_counts["swapped_image"],
        },
        "counts": {
            "tasks": len(tasks),
            "gold_rows": len(gold),
            "conditions": dict(sorted(condition_counts.items())),
            "answers": dict(sorted(answer_counts.items())),
            "resolved_image_references": len(image_cache),
            "resolved_unique_image_hashes": len(set(image_sha256_by_key.values())),
            "bootstrap_components": len(components),
        },
        "deduplication_audit": dedup_audit,
        "split_image_overlap_audit": overlap_audit,
        "derived_test_exclusion": derived_test_audit,
        "bootstrap_clustering": {
            "unit": "connected component of original/swapped image graph",
            "component_count": len(components),
            "components": component_rows,
            "all_task_cluster_ids_are_components": True,
            "byte_identical_image_references_are_connected": True,
        },
        "context_control": {
            "length": 8,
            "alphabet": CONTEXT_ALPHABET,
            "derivation": (
                "seeded hash of effective split and family_id only; independent of answer"
            ),
            "fixed_within_family": True,
            "unique_across_selected_families": True,
        },
    }
    return tasks, gold, manifest


def _write_outputs_once(
    output_dir: Path,
    *,
    task_bytes: bytes,
    gold_bytes: bytes,
    manifest_bytes: bytes,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        output_dir / "tasks.jsonl": task_bytes,
        output_dir / "gold.private.jsonl": gold_bytes,
        output_dir / "selection_manifest.json": manifest_bytes,
    }
    existing = {path: path.exists() for path in payloads}
    if any(existing.values()):
        if not all(existing.values()):
            raise FileExistsError(
                f"partial frozen output exists under {output_dir}; choose a fresh output directory"
            )
        mismatched = [path for path, payload in payloads.items() if path.read_bytes() != payload]
        if mismatched:
            raise FileExistsError(
                "refusing to overwrite non-identical frozen outputs: "
                + ", ".join(str(path) for path in mismatched)
            )
        return "IDEMPOTENT_EXISTING"

    temporary: dict[Path, Path] = {}
    created: list[Path] = []
    try:
        for target, payload in payloads.items():
            handle = tempfile.NamedTemporaryFile(delete=False, dir=output_dir)
            try:
                with handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary[target] = Path(handle.name)
            except BaseException:
                Path(handle.name).unlink(missing_ok=True)
                raise
        for target, staged in temporary.items():
            # Hard-linking is an atomic, no-overwrite publication operation.
            os.link(staged, target)
            created.append(target)
        return "CREATED"
    except BaseException:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    finally:
        for staged in temporary.values():
            staged.unlink(missing_ok=True)


def prepare(
    *,
    dataset_root: Path,
    output_dir: Path,
    split: str,
    source_json: Path | None = None,
    images_root: Path | None = None,
    max_items: int = 0,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    tasks, gold, manifest = build_outputs(
        dataset_root=dataset_root,
        split=split,
        source_json=source_json,
        images_root=images_root,
        max_items=max_items,
        seed=seed,
    )
    task_bytes = _jsonl_bytes(tasks)
    gold_bytes = _jsonl_bytes(gold)
    finalized_manifest = dict(manifest)
    finalized_manifest.update(
        {
            "tasks_file": "tasks.jsonl",
            "gold_file": "gold.private.jsonl",
            "tasks_sha256": _sha256_bytes(task_bytes),
            "gold_sha256": _sha256_bytes(gold_bytes),
        }
    )
    manifest_bytes = _manifest_bytes(finalized_manifest)
    write_status = _write_outputs_once(
        output_dir.resolve(),
        task_bytes=task_bytes,
        gold_bytes=gold_bytes,
        manifest_bytes=manifest_bytes,
    )
    return {
        "status": write_status,
        "output_dir": str(output_dir.resolve()),
        "split": finalized_manifest["dataset"]["split"],
        "source_split": split,
        "families": finalized_manifest["selection"]["selected_families"],
        "tasks": len(tasks),
        "tasks_sha256": finalized_manifest["tasks_sha256"],
        "gold_sha256": finalized_manifest["gold_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="maximum source families after deterministic ranking; 0 selects all",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split=args.split,
        source_json=args.source_json,
        images_root=args.images_root,
        max_items=args.max_items,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
