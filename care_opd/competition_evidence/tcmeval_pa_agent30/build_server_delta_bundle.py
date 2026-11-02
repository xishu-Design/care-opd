#!/usr/bin/env python3
"""Build the small overlay bundle needed for the GPU-3 30-question run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any


FILES = (
    "competition_evidence/execute_three_role_eval.py",
    "competition_evidence/local_four_agent_server/run_four_agent_local_server.sh",
    "competition_evidence/local_four_agent_server/serve_care_transformers.py",
    "competition_evidence/tcmeval_pa_agent30/README.md",
    "competition_evidence/tcmeval_pa_agent30/SERVER_COMMANDS.md",
    "competition_evidence/tcmeval_pa_agent30/prepare_tcmeval_pa_agent30.py",
    "competition_evidence/tcmeval_pa_agent30/score_tcmeval_pa_agent30.py",
    "competition_evidence/tcmeval_pa_agent30/run_tcmeval_pa_agent30_gpu3.sh",
    "public_eval/tcmeval_pa/evaluate_tcmeval_pa.py",
    "public_eval/tcmeval_pa/data/TCMEval-PA.xlsx",
    "public_eval/tcmeval_pa/data/DATA_PROVENANCE.md",
    "runs/public_eval/tcmeval_pa_agent30_fixed_seed20260825/tcmeval_pa_agent30.tasks.jsonl",
    "runs/public_eval/tcmeval_pa_agent30_fixed_seed20260825/tcmeval_pa_agent30.selection_manifest.json",
)


def care_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.care_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    sha_path = output.with_name(output.name + ".sha256")
    if output.exists() or sha_path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle or checksum: {output}")
    entries = []
    for relative in FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        entries.append(
            {
                "path_relative_to_care_root": relative,
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    contents = {
        "schema_version": "care_opd.competition.tcmeval_pa_agent30_bundle.v1",
        "status": "COMPLETE",
        "install_instruction": "extract this archive directly into the existing CARE_ROOT",
        "privacy": "contains public benchmark data and prompts; no real medical records or model outputs",
        "files": entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tcmeval_agent30_") as temporary:
        contents_path = Path(temporary) / "TCMEVAL_PA_AGENT30_BUNDLE_CONTENTS.json"
        contents_path.write_text(
            json.dumps(contents, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with output.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
            for entry in entries:
                relative = entry["path_relative_to_care_root"]
                info = archive.gettarinfo(str(root / relative), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mtime = 0
                with (root / relative).open("rb") as handle:
                    archive.addfile(info, handle)
            info = archive.gettarinfo(
                str(contents_path), arcname="TCMEVAL_PA_AGENT30_BUNDLE_CONTENTS.json"
            )
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with contents_path.open("rb") as handle:
                archive.addfile(info, handle)
    digest = sha256_file(output)
    sha_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return {
        "status": "COMPLETE",
        "bundle": str(output),
        "bundle_size": output.stat().st_size,
        "bundle_sha256": digest,
        "checksum": str(sha_path),
        "file_count": len(entries),
    }


def parser() -> argparse.ArgumentParser:
    root = care_root()
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--care-root", type=Path, default=root)
    value.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "runs"
            / "public_eval"
            / "tcmeval_pa_agent30_bundle"
            / "TCMEVAL_PA_AGENT30_GPU3_NO_DOWNLOAD_FINAL.tar.gz"
        ),
    )
    return value


def main() -> None:
    print(json.dumps(build(parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
