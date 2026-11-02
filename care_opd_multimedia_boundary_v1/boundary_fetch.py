"""Download only public ChartQA validation parquet, pinned to a commit SHA.

No models are downloaded. The dataset card declares GPL-3.0; original chart
rights still apply. DocVQA requires an authorized normalized local import.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

try:
    from .boundary_common import hash_file, json_once, read_json
except ImportError:
    from boundary_common import hash_file, json_once, read_json

REPO_ID = "HuggingFaceM4/ChartQA"
ALLOW_PATTERNS = ["data/val-*.parquet", "README.md", "LICENSE", "LICENSE.*"]
PROXY_NAMES = frozenset({"http_proxy", "https_proxy", "all_proxy", "no_proxy"})
REQUEST_NAME = "download_request.json"


def child_environment(environment: dict[str, str], ignore_proxy: bool) -> dict[str, str]:
    """Copy the environment; never alter parent process proxy settings."""
    result = dict(environment)
    if ignore_proxy:
        result = {key: value for key, value in result.items() if key.casefold() not in PROXY_NAMES}
    # This explicit dataset-fetch child needs network access even when its
    # parent's later, local-only inference process has offline mode enabled.
    result["HF_HUB_OFFLINE"] = "0"
    result["HF_DATASETS_OFFLINE"] = "0"
    return result


def _allowed_name(name: str) -> bool:
    return bool(isinstance(name, str) and "\\" not in name and ".." not in Path(name).parts
                and not Path(name).is_absolute()
                and (re.fullmatch(r"data/val-[^/]+\.parquet", name)
                     or name in {"README.md", "LICENSE"} or re.fullmatch(r"LICENSE\.[^/]+", name)))


def _read_request(destination: Path, revision: str) -> dict:
    request_path = destination / REQUEST_NAME
    if not request_path.resolve().is_relative_to(destination):
        raise ValueError("Locked download request path escapes destination")
    request = read_json(request_path)
    if (request.get("schema_version") != "care_opd.multimedia_boundary.download_request.v1"
            or request.get("repo_id") != REPO_ID or request.get("repo_type") != "dataset"
            or request.get("split") != "val" or request.get("allow_patterns") != ALLOW_PATTERNS
            or not re.fullmatch(r"[0-9a-f]{40}", str(request.get("resolved_revision")))):
        raise ValueError("Invalid locked ChartQA download request")
    if revision not in {request.get("requested_revision"), request.get("resolved_revision")}:
        raise ValueError("Incomplete download is locked to another revision; choose a new destination")
    shards, expected = request.get("validation_shards"), request.get("expected_files")
    if (not isinstance(shards, list) or not shards or any(not isinstance(name, str) for name in shards)
            or len(set(shards)) != len(shards) or any(not re.fullmatch(r"data/val-[^/]+\.parquet", name) for name in shards)
            or not isinstance(expected, list) or any(not isinstance(name, str) for name in expected)
            or len(set(expected)) != len(expected) or any(not _allowed_name(name) for name in expected)
            or not set(shards).issubset(expected)):
        raise ValueError("Invalid validation file list in locked request")
    return request


def _download_files(destination: Path, request: dict) -> dict[str, str]:
    """Hash only allowed pinned files, refusing escaped/foreign partial files."""
    files = {}
    for path in sorted(destination.rglob("*")):
        relative = path.relative_to(destination)
        if not path.resolve().is_relative_to(destination):
            raise ValueError("Unexpected escaped path in download destination: " + relative.as_posix())
        if ".cache" in relative.parts or relative.as_posix() in {REQUEST_NAME, "download_manifest.json"}:
            continue
        if not path.is_file():
            continue
        name = relative.as_posix()
        if not _allowed_name(name) or name not in request["expected_files"]:
            raise ValueError("Unexpected non-allowed/escaped file in download destination: " + name)
        files[name] = hash_file(path)
    return files


def _verify_existing(destination: Path, revision: str) -> dict:
    manifest_path = destination / "download_manifest.json"
    if not manifest_path.resolve().is_relative_to(destination):
        raise ValueError("Download manifest path escapes destination")
    existing = read_json(manifest_path)
    if (existing.get("schema_version") != "care_opd.multimedia_boundary.download.v1"
            or existing.get("repo_id") != REPO_ID or existing.get("repo_type") != "dataset"
            or existing.get("split") != "val" or not re.fullmatch(r"[0-9a-f]{40}", str(existing.get("resolved_revision")))):
        raise ValueError("Invalid ChartQA download manifest")
    if revision not in {existing.get("resolved_revision"), existing.get("requested_revision")}:
        raise ValueError("Existing download uses another revision; choose a new destination")
    request_path = destination / REQUEST_NAME
    if not request_path.is_file() or existing.get("download_request_sha256") != hash_file(request_path):
        raise ValueError("Download request is missing/changed; do not mix pinned revisions")
    request = _read_request(destination, revision)
    if request["resolved_revision"] != existing["resolved_revision"]:
        raise ValueError("Download request and manifest revisions differ")
    files = _download_files(destination, request)
    if files != existing.get("files") or set(files) != set(request["expected_files"]):
        raise ValueError("Existing pinned dataset is incomplete/changed; choose a new destination")
    return existing


def fetch_chartqa(destination: Path, revision: str = "main") -> dict:
    destination = Path(destination).resolve()
    if (destination / "download_manifest.json").exists():
        return _verify_existing(destination, revision)
    request_path = destination / REQUEST_NAME
    request = None
    if request_path.is_file():
        request = _read_request(destination, revision)
        _download_files(destination, request)
    elif destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Fetch requires a new/empty destination, or a valid locked request/completed manifest")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Fetching requires huggingface_hub; no model library is needed") from exc
    if request is None:
        info = HfApi().dataset_info(REPO_ID, revision=revision, files_metadata=False)
        sha = str(info.sha)
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Hub did not return a full immutable commit SHA")
        members = {entry.rfilename for entry in info.siblings}
        shards = sorted(name for name in members if re.fullmatch(r"data/val-[^/]+\.parquet", name))
        if not shards:
            raise ValueError("Pinned repository contains no expected validation parquet shards")
        request = {"schema_version": "care_opd.multimedia_boundary.download_request.v1", "repo_id": REPO_ID,
                   "repo_type": "dataset", "split": "val", "requested_revision": revision,
                   "resolved_revision": sha, "allow_patterns": ALLOW_PATTERNS,
                   "validation_shards": shards, "expected_files": sorted(name for name in members if _allowed_name(name))}
        destination.mkdir(parents=True, exist_ok=True)
        # Written BEFORE the first snapshot call. A retry of main must use this
        # immutable SHA, not whatever main points to after a network failure.
        json_once(request_path, request)
    sha = request["resolved_revision"]
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=sha, local_dir=str(destination),
                      allow_patterns=ALLOW_PATTERNS)
    files = _download_files(destination, request)
    for name in request["expected_files"]:
        if name not in files:
            raise ValueError(f"Pinned file not downloaded: {name}; retry the same command to resume")
    manifest = {"schema_version": "care_opd.multimedia_boundary.download.v1", "repo_id": REPO_ID,
                "repo_type": "dataset", "requested_revision": request["requested_revision"], "resolved_revision": sha,
                "split": "val", "allow_patterns": ALLOW_PATTERNS, "files": files,
                "download_request_sha256": hash_file(request_path),
                "source_url": "https://huggingface.co/datasets/" + REPO_ID,
                "license_declaration": "gpl-3.0", "license_notice": "Dataset-card declaration only; original chart/image rights still apply.",
                "downloads_models": False, "docvqa": "not_downloaded_authorized_normalized_import_only"}
    json_once(destination / "download_manifest.json", manifest)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", "--output", dest="destination", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Resolved once to an immutable full SHA")
    parser.add_argument("--ignore-proxy", action="store_true", help="Remove proxy variables in a child process only")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.child:
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--destination", str(args.destination),
                   "--revision", args.revision]
        return subprocess.run(command, env=child_environment(os.environ, args.ignore_proxy), check=False).returncode
    print("Source: HuggingFaceM4/ChartQA validation only. Card license: GPL-3.0; original chart rights still apply. No models downloaded.", flush=True)
    manifest = fetch_chartqa(args.destination, args.revision)
    print(json.dumps({"downloaded": True, "repo_id": REPO_ID, "revision": manifest["resolved_revision"],
                      "destination": str(args.destination.resolve()), "files": len(manifest["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
