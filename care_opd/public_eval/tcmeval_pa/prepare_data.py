#!/usr/bin/env python3
"""Download and verify the immutable TCMEval-PA v3 workbook.

Only Python's standard library is used so this step can run in the isolated
CARE-OPD evaluation environment before torch/transformers are imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ARTICLE_API = "https://api.figshare.com/v2/articles/29651261/versions/3"
ARTICLE_DOI = "10.6084/m9.figshare.29651261.v3"
EXPECTED_FILENAME = "TCMEval-PA.xlsx"
EXPECTED_SHA256 = "905cb28f145ba50f3b1c912265ff6649e0bb38f56adf6cd588d77ca768db92c1"
TCMEVAL_REPO_COMMIT = "12473b50ce94ad2fd22b1e984b826b0a618c4407"
USER_AGENT = "CARE-OPD-TCMEval-PA/1.0 (research benchmark downloader)"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def fetch_json(url: str, timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Figshare returned a non-object response for {url}")
    return payload


def select_download(article: Mapping[str, Any]) -> Mapping[str, Any]:
    files = article.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Figshare article response has no files list")
    matches = [
        item
        for item in files
        if isinstance(item, Mapping)
        and str(item.get("name", "")).casefold() == EXPECTED_FILENAME.casefold()
    ]
    if len(matches) != 1:
        names = [str(item.get("name")) for item in files if isinstance(item, Mapping)]
        raise RuntimeError(
            f"expected exactly one {EXPECTED_FILENAME!r} file, found {len(matches)}; "
            f"available files: {names}"
        )
    selected = matches[0]
    if not selected.get("download_url"):
        raise RuntimeError("selected Figshare file has no download_url")
    return selected


def download_verified(url: str, output: Path, timeout: float) -> tuple[int, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    digest = hashlib.sha256()
    total = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with os.fdopen(fd, "wb") as destination:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
                    total += len(block)
            destination.flush()
            os.fsync(destination.fileno())
        actual = digest.hexdigest()
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                "downloaded workbook checksum mismatch: "
                f"expected {EXPECTED_SHA256}, got {actual}"
            )
        os.replace(temporary_name, output)
        return total, actual
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def prepare(output: Path, *, timeout: float, verify_only: bool) -> Mapping[str, Any]:
    output = output.resolve()
    reused = False
    selected: Optional[Mapping[str, Any]] = None
    article: Optional[Mapping[str, Any]] = None

    if output.is_file():
        actual = sha256_file(output)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                f"existing file {output} has SHA256 {actual}, expected {EXPECTED_SHA256}; "
                "refusing to overwrite an unrecognized file"
            )
        reused = True
        size = output.stat().st_size
    else:
        if verify_only:
            raise FileNotFoundError(f"cannot verify missing workbook: {output}")
        article = fetch_json(ARTICLE_API, timeout)
        selected = select_download(article)
        size, actual = download_verified(str(selected["download_url"]), output, timeout)

    manifest = {
        "schema_version": "tcmeval-pa-data-manifest-v1",
        "benchmark": "TCMEval-PA",
        "figshare_article_api": ARTICLE_API,
        "figshare_doi": ARTICLE_DOI,
        "figshare_version": 3,
        "filename": EXPECTED_FILENAME,
        "path": str(output),
        "bytes": int(size),
        "sha256": actual,
        "expected_sha256": EXPECTED_SHA256,
        "tcmeval_repo_commit": TCMEVAL_REPO_COMMIT,
        "license": "CC BY 4.0",
        "reused_verified_file": reused,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if selected is not None:
        manifest["figshare_file_id"] = selected.get("id")
        manifest["download_url"] = selected.get("download_url")
    if article is not None:
        manifest["figshare_article_id"] = article.get("id")
        manifest["figshare_title"] = article.get("title")
    atomic_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--timeout", type=float, default=120.0)
    result.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing workbook without making a network request",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    manifest = prepare(args.output, timeout=args.timeout, verify_only=args.verify_only)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
