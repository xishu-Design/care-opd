"""Package only the two new experiment adapters; never include data or weights."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


PACKAGE_NAME = "CARE_OPD_TMM_VISUAL_TROPD_V1.tar.gz"
DIRECTORIES = ("visual_evidence_transfer", "tropd_backfill")


def build(output: Path, package_name: str = PACKAGE_NAME, *, visual_only: bool = False) -> dict:
    if Path(package_name).name != package_name or not package_name.endswith(".tar.gz"):
        raise ValueError("package name must be a plain .tar.gz filename")
    experiments = Path(__file__).resolve().parent.parent
    files = []
    for name in (("visual_evidence_transfer",) if visual_only else DIRECTORIES):
        directory = experiments / name
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in {".py", ".md", ".sh"}:
                if path.is_symlink():
                    raise ValueError(f"refusing a source symlink: {path}")
                files.append(path)
    for relative in (
        "visual_evidence_transfer/fetch_assets.py",
        "visual_evidence_transfer/prepare_slake.py",
        "visual_evidence_transfer/evaluate.py",
        "visual_evidence_transfer/score_results.py",
        "visual_evidence_transfer/SERVER_COMMANDS.md",
        "tropd_backfill/run_tropd_backfill.sh",
        "tropd_backfill/protocol_guard.py",
        "tropd_backfill/SERVER_COMMANDS.md",
    ):
        if visual_only and relative.startswith("tropd_backfill/"):
            continue
        if experiments / relative not in files:
            raise FileNotFoundError(relative)
    if visual_only:
        for name in ("gpu45.py", "run_visual_transfer_gpu45.sh", "SERVER_COMMANDS_GPU45.md"):
            if experiments / "visual_evidence_transfer" / name not in files:
                raise FileNotFoundError(name)
    inventory = []
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                data = path.read_bytes()
                relative = "mechanism_experiments/" + path.relative_to(experiments).as_posix()
                entry = tarfile.TarInfo(relative)
                entry.size = len(data)
                entry.mode = 0o700 if path.suffix == ".sh" else 0o600
                entry.mtime = 0
                archive.addfile(entry, io.BytesIO(data))
                inventory.append({"path": relative, "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()})
    payload = stream.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (
        (package_name, payload),
        (package_name + ".sha256", f"{sha}  {package_name}\n".encode()),
        (package_name + ".manifest.json", (json.dumps({
            "schema_version": "care_opd.experiment_adapter_bundle.v1",
            "sha256": sha, "archive": package_name, "files": inventory,
            "contains_dataset_or_model_weights": False,
            "requires_existing_care_opd_project": True,
            "gpu_experiments_executed_by_packager": False,
        }, indent=2, ensure_ascii=False) + "\n").encode("utf-8")),
    ):
        target = output / name
        if target.exists():
            if target.read_bytes() != data:
                raise FileExistsError(f"different existing bundle: {target}; use a new output directory")
        else:
            with target.open("xb") as handle:
                handle.write(data)
    return {"archive": str(output / package_name), "sha256": sha,
            "source_files": len(files), "bytes": len(payload)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "dist")
    parser.add_argument("--package-name", default=PACKAGE_NAME)
    parser.add_argument("--visual-only", action="store_true",
                        help="do not touch the independently running TROPD backfill adapter")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.package_name, visual_only=args.visual_only), indent=2))
