#!/usr/bin/env python3
"""Fetch pinned SLAKE assets, or explicitly fetch the frozen VLM snapshot.

The default SLAKE command downloads only public split metadata and ``imgs.zip``;
the optional SLAKE knowledge graph is deliberately excluded.  A model snapshot
is downloaded only through the explicit ``vlm`` subcommand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


SLAKE_REPO = "BoKelvin/SLAKE"
SLAKE_COMMIT = "a9083ce6c34ac3ffb17671a605962924d8a8f9e9"
SLAKE_FILES = (
    "train.json",
    "validation.json",
    "test.json",
    "README.md",
    "mask.txt",
    "imgs.zip",
)
SLAKE_MANIFEST = "slake_assets_manifest.json"
VLM_MANIFEST = "vlm_snapshot_manifest.json"
DEFAULT_VLM_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"

MAX_ZIP_ENTRIES = 200_000
MAX_ZIP_TOTAL_BYTES = 12 * 1024**3
MAX_ZIP_MEMBER_BYTES = 1024**3
MAX_ZIP_RATIO = 500.0
COPY_BLOCK = 8 * 1024 * 1024


class AssetError(RuntimeError):
    """Raised when a remote or local asset violates the frozen contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(COPY_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _manifest_member_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise FileExistsError("asset manifest contains an invalid relative path")
    normalized = relative.replace("\\", "/")
    if normalized.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", normalized):
        raise FileExistsError("asset manifest contains an absolute path")
    value = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in value.parts):
        raise FileExistsError("asset manifest contains a traversing path")
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / Path(*value.parts)).resolve(strict=False)
    if not _is_within(path, resolved_root):
        raise FileExistsError("asset manifest path escapes its destination")
    return path


def _safe_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
    raw = info.filename.replace("\\", "/")
    if not raw or "\x00" in raw or raw.startswith(("/", "//")):
        raise AssetError(f"unsafe absolute/empty zip member: {info.filename!r}")
    if re.match(r"^[A-Za-z]:", raw):
        raise AssetError(f"unsafe drive-qualified zip member: {info.filename!r}")
    member = PurePosixPath(raw)
    if not member.parts or any(part in {"", ".", ".."} for part in member.parts):
        raise AssetError(f"unsafe traversing zip member: {info.filename!r}")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode):
        raise AssetError(f"zip symlink is forbidden: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise AssetError(f"encrypted zip member is forbidden: {info.filename!r}")
    return member


def safe_extract_zip(
    archive: Path,
    destination: Path,
    *,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_total_bytes: int = MAX_ZIP_TOTAL_BYTES,
    max_member_bytes: int = MAX_ZIP_MEMBER_BYTES,
    max_ratio: float = MAX_ZIP_RATIO,
) -> dict[str, Any]:
    """Safely and streaming-extract an untrusted ZIP archive.

    Extraction rejects path traversal, links, case-fold collisions, encrypted
    members, excessive member counts/sizes, suspicious compression ratios, and
    any attempt to overwrite a pre-existing staging file.
    """

    archive = archive.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve(strict=True)
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        if len(infos) > max_entries:
            raise AssetError(
                f"zip has {len(infos)} entries, exceeding limit {max_entries}"
            )
        declared_total = 0
        normalized_targets: set[str] = set()
        prepared: list[tuple[zipfile.ZipInfo, Path, bool]] = []
        for info in infos:
            member = _safe_zip_member(info)
            is_directory = info.is_dir() or info.filename.endswith("/")
            if info.file_size < 0 or info.compress_size < 0:
                raise AssetError(f"negative zip member size: {info.filename!r}")
            if info.file_size > max_member_bytes:
                raise AssetError(f"oversized zip member: {info.filename!r}")
            declared_total += info.file_size
            if declared_total > max_total_bytes:
                raise AssetError("zip declared uncompressed size exceeds safety limit")
            if info.file_size and info.compress_size == 0:
                raise AssetError(f"invalid zero compressed size: {info.filename!r}")
            if info.compress_size and info.file_size / info.compress_size > max_ratio:
                raise AssetError(f"suspicious zip compression ratio: {info.filename!r}")
            target = (root / Path(*member.parts)).resolve(strict=False)
            if not _is_within(target, root):
                raise AssetError(f"zip member escapes destination: {info.filename!r}")
            folded = str(target).casefold()
            if folded in normalized_targets:
                raise AssetError(f"duplicate/case-colliding zip target: {info.filename!r}")
            normalized_targets.add(folded)
            if target.exists():
                raise AssetError(f"zip member would overwrite an existing file: {target}")
            prepared.append((info, target, is_directory))

        extracted_total = 0
        extracted_files = 0
        for info, target, is_directory in prepared:
            if is_directory:
                # A parent can already have been materialized by an earlier file
                # even when the archive lists its directory entry later.
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member_total = 0
            try:
                with handle.open(info, "r") as source, target.open("xb") as sink:
                    while True:
                        block = source.read(COPY_BLOCK)
                        if not block:
                            break
                        member_total += len(block)
                        extracted_total += len(block)
                        if member_total > max_member_bytes or extracted_total > max_total_bytes:
                            raise AssetError("zip expanded beyond configured safety limit")
                        sink.write(block)
                if member_total != info.file_size:
                    raise AssetError(
                        f"zip member size mismatch for {info.filename!r}: "
                        f"declared={info.file_size} actual={member_total}"
                    )
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            extracted_files += 1
    return {
        "archive": str(archive),
        "entry_count": len(infos),
        "file_count": extracted_files,
        "uncompressed_bytes": extracted_total,
        "limits": {
            "max_entries": max_entries,
            "max_total_bytes": max_total_bytes,
            "max_member_bytes": max_member_bytes,
            "max_compression_ratio": max_ratio,
        },
    }


def _load_huggingface_hub() -> tuple[Any, Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise AssetError(
            "huggingface_hub is required for downloads; install it in the download environment"
        ) from exc
    return HfApi, hf_hub_download, snapshot_download


def _resolved_sha(info: Any) -> str:
    value = getattr(info, "sha", None)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", value):
        raise AssetError("Hugging Face did not return a valid resolved commit")
    return value.lower()


def _publish_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    os.replace(staging, destination)


def _verify_existing_slake(destination: Path) -> dict[str, Any]:
    manifest_path = destination / SLAKE_MANIFEST
    if not manifest_path.is_file():
        raise FileExistsError(
            f"existing SLAKE destination lacks {SLAKE_MANIFEST}: {destination}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repo_id") != SLAKE_REPO or manifest.get("resolved_commit") != SLAKE_COMMIT:
        raise FileExistsError("existing SLAKE destination has different provenance")
    for entry in manifest.get("downloaded_files", []):
        if not isinstance(entry, dict):
            raise FileExistsError("existing SLAKE downloaded file inventory is malformed")
        path = _manifest_member_path(destination, entry.get("path"))
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise FileExistsError(f"existing SLAKE asset is missing or changed: {path}")
        if _sha256_file(path) != entry["sha256"]:
            raise FileExistsError(f"existing SLAKE asset hash mismatch: {path}")
    extracted = manifest.get("extracted_images")
    if not isinstance(extracted, list) or not extracted:
        raise FileExistsError("existing SLAKE manifest lacks extracted image identities")
    for entry in extracted:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise FileExistsError("existing SLAKE extracted image manifest is malformed")
        path = _manifest_member_path(destination, entry.get("path"))
        if not path.is_file() or path.stat().st_size != entry.get("size_bytes"):
            raise FileExistsError(f"existing extracted SLAKE image is missing or changed: {path}")
        if _sha256_file(path) != entry.get("sha256"):
            raise FileExistsError(f"existing extracted SLAKE image hash mismatch: {path}")
    return {
        "status": "IDEMPOTENT_EXISTING",
        "destination": str(destination.resolve()),
        "repo_id": SLAKE_REPO,
        "resolved_commit": SLAKE_COMMIT,
        "manifest": str(manifest_path.resolve()),
    }


def fetch_slake(destination: Path, *, revision: str = SLAKE_COMMIT) -> dict[str, Any]:
    """Download and publish exactly the pinned SLAKE metadata and images."""

    if revision != SLAKE_COMMIT:
        raise AssetError(
            f"SLAKE revision is frozen at {SLAKE_COMMIT}; received {revision!r}"
        )
    destination = destination.resolve()
    if destination.exists():
        return _verify_existing_slake(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".slake-stage-", dir=destination.parent))
    try:
        HfApi, hf_hub_download, _ = _load_huggingface_hub()
        info = HfApi().dataset_info(SLAKE_REPO, revision=revision, files_metadata=True)
        resolved = _resolved_sha(info)
        if resolved != SLAKE_COMMIT:
            raise AssetError(
                f"resolved SLAKE commit mismatch: expected {SLAKE_COMMIT}, got {resolved}"
            )
        downloaded: list[dict[str, Any]] = []
        for filename in SLAKE_FILES:
            cached = Path(
                hf_hub_download(
                    repo_id=SLAKE_REPO,
                    repo_type="dataset",
                    filename=filename,
                    revision=resolved,
                )
            )
            target = staging / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
            downloaded.append(
                {
                    "path": filename,
                    "size_bytes": target.stat().st_size,
                    "sha256": _sha256_file(target),
                }
            )

        extraction = safe_extract_zip(staging / "imgs.zip", staging)
        # Avoid persisting a temporary absolute staging path in provenance.
        extraction["archive"] = "imgs.zip"
        extracted_images: list[dict[str, Any]] = []
        for path in sorted(staging.rglob("*"), key=lambda value: value.as_posix()):
            if (
                path.is_file()
                and path.name != "imgs.zip"
                and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ):
                extracted_images.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
        if not extracted_images:
            raise AssetError("imgs.zip produced no recognized image files")
        manifest = {
            "schema_version": "care_opd.visual_transfer.slake_assets.v1",
            "status": "COMPLETE",
            "repo_id": SLAKE_REPO,
            "requested_revision": revision,
            "resolved_commit": resolved,
            "downloaded_files": downloaded,
            "excluded_repository_assets": [
                {
                    "path": "KG.zip",
                    "reason": "not required by the frozen visual QA mechanism experiment",
                }
            ],
            "extraction": extraction,
            "extracted_image_count": len(extracted_images),
            "extracted_images": extracted_images,
        }
        _write_new(staging / SLAKE_MANIFEST, _json_bytes(manifest))
        _publish_directory(staging, destination)
        return {
            "status": "CREATED",
            "destination": str(destination),
            "repo_id": SLAKE_REPO,
            "resolved_commit": resolved,
            "manifest": str(destination / SLAKE_MANIFEST),
            "extracted_image_count": len(extracted_images),
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _iter_snapshot_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file() and VLM_MANIFEST not in path.parts and ".cache" not in path.parts:
            yield path


def _verify_existing_vlm(
    destination: Path, *, repo_id: str, resolved_commit: str
) -> dict[str, Any]:
    manifest_path = destination / VLM_MANIFEST
    if not manifest_path.is_file():
        raise FileExistsError(
            f"existing VLM destination lacks {VLM_MANIFEST}: {destination}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("repo_id") != repo_id
        or manifest.get("resolved_commit") != resolved_commit
    ):
        raise FileExistsError("existing VLM destination has different provenance")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise FileExistsError("existing VLM manifest has an empty/malformed file inventory")
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise FileExistsError("existing VLM file inventory is malformed")
        path = _manifest_member_path(destination, entry.get("path"))
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise FileExistsError(f"existing VLM snapshot is incomplete or changed: {path}")
        if _sha256_file(path) != entry.get("sha256"):
            raise FileExistsError(f"existing VLM snapshot hash mismatch: {path}")
    return {
        "status": "IDEMPOTENT_EXISTING",
        "destination": str(destination.resolve()),
        "repo_id": repo_id,
        "resolved_commit": resolved_commit,
        "manifest": str(manifest_path.resolve()),
    }


def fetch_vlm(
    destination: Path,
    *,
    repo_id: str = DEFAULT_VLM_REPO,
    revision: str | None = None,
) -> dict[str, Any]:
    """Explicitly download a complete VLM snapshot and freeze its commit."""

    if not repo_id.strip():
        raise AssetError("VLM repo id must not be empty")
    destination = destination.resolve()
    if destination.exists():
        # With no revision argument, an existing destination remains bound to
        # its recorded immutable commit; do not contact a potentially moved
        # default branch merely to establish idempotence.
        if revision is None:
            manifest_path = destination / VLM_MANIFEST
            if not manifest_path.is_file():
                raise FileExistsError(
                    f"existing VLM destination lacks {VLM_MANIFEST}: {destination}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            resolved = manifest.get("resolved_commit")
            if not isinstance(resolved, str):
                raise FileExistsError("existing VLM manifest lacks resolved_commit")
            return _verify_existing_vlm(
                destination, repo_id=repo_id, resolved_commit=resolved
            )
        # Explicit revisions are resolved again so a moved branch cannot be
        # mistaken for the prior frozen snapshot.
        HfApi, _, _ = _load_huggingface_hub()
        resolved = _resolved_sha(
            HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        )
        return _verify_existing_vlm(
            destination, repo_id=repo_id, resolved_commit=resolved
        )

    HfApi, _, snapshot_download = _load_huggingface_hub()
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    resolved = _resolved_sha(info)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    try:
        # Download into the persistent HF cache first.  Interrupted downloads
        # remain resumable through HF_HOME/HF_HUB_CACHE (recommended on /data)
        # instead of being destroyed with a random local_dir staging tree.
        cached_snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=resolved,
            )
        ).resolve(strict=True)
        staging = Path(tempfile.mkdtemp(prefix=".vlm-stage-", dir=destination.parent))
        shutil.copytree(
            cached_snapshot,
            staging,
            dirs_exist_ok=True,
            symlinks=False,
        )
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        files: list[dict[str, Any]] = []
        for path in _iter_snapshot_files(staging):
            files.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        if not files:
            raise AssetError("downloaded VLM snapshot has an empty file inventory")
        manifest = {
            "schema_version": "care_opd.visual_transfer.vlm_snapshot.v1",
            "status": "COMPLETE",
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_commit": resolved,
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(entry["size_bytes"] for entry in files),
            "download_is_explicit_only": True,
            "persistent_hf_cache_source": str(cached_snapshot),
        }
        _write_new(staging / VLM_MANIFEST, _json_bytes(manifest))
        _publish_directory(staging, destination)
        staging = None
        return {
            "status": "CREATED",
            "destination": str(destination),
            "repo_id": repo_id,
            "resolved_commit": resolved,
            "manifest": str(destination / VLM_MANIFEST),
            "file_count": len(files),
        }
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _add_slake_parser(subparsers: Any, name: str) -> None:
    parser = subparsers.add_parser(name, help="download pinned SLAKE metadata and images")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--revision", default=SLAKE_COMMIT)
    parser.set_defaults(command_kind="slake")


def _add_vlm_parser(subparsers: Any, name: str) -> None:
    parser = subparsers.add_parser(name, help="explicitly download a complete VLM snapshot")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_VLM_REPO)
    parser.add_argument("--revision")
    parser.set_defaults(command_kind="vlm")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("slake", "download-slake"):
        _add_slake_parser(commands, name)
    for name in ("vlm", "download-vlm"):
        _add_vlm_parser(commands, name)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command_kind == "slake":
        result = fetch_slake(args.dest, revision=args.revision)
    else:
        result = fetch_vlm(args.dest, repo_id=args.repo, revision=args.revision)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
