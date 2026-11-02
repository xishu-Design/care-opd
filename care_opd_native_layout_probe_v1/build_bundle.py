"""Build a deterministic source-only tarball; no private results or weights."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
NAME = "CARE_OPD_NATIVE_LAYOUT_PROBE_GPU1_V1.tar.gz"


def inventory():
    files = {}
    for path in sorted(ROOT.glob("*.py")):
        if path.name != "build_bundle.py":
            files[path.name] = path.read_bytes()
    for name in ("README.md", "SERVER_COMMANDS.md", "run_probe_gpu1.sh"):
        files[name] = (ROOT / name).read_bytes()
    for name in ("boundary_common.py", "boundary_score.py"):
        files["_vendor/" + name] = (ROOT.parent / "care_opd_multimedia_boundary_v1" / name).read_bytes()
    runtime = ROOT.parent / "care_opd/mechanism_experiments/visual_evidence_transfer"
    for name in ("common.py", "evaluate.py"):
        # Preserve EXACT source bytes: the old run's hash binds this runtime.
        files["runtime/" + name] = (runtime / name).read_bytes()
    manifest = {"schema_version": "care_opd.native_layout_probe.package.v1", "files": {
        n: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for n, data in sorted(files.items())}}
    files["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    return files


def build(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = inventory()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid = len(data), 0, 0, 0
            info.mode = 0o700 if name.endswith(".sh") else 0o600
            archive.addfile(info, io.BytesIO(data))
    payload = gzip.compress(stream.getvalue(), mtime=0)
    archive_path = destination / NAME
    if archive_path.exists() and archive_path.read_bytes() != payload:
        raise FileExistsError("package already exists with different content; use a new build directory")
    archive_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (destination / (NAME + ".sha256")).write_text(f"{digest}  {NAME}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive_path), "sha256": digest, "files": len(files)}, ensure_ascii=False))
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "dist")
    build(parser.parse_args().destination)
