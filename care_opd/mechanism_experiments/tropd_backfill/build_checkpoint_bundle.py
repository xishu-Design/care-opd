"""Build a minimal, reproducible checkpoint deployment patch (no data/weights)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

NAME = "CARE_OPD_QZBC_GPU1_CHECKPOINT_V1.tar.gz"
ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "src/care_opd/qzbc_rollout.py",
    "tests/test_qzbc_rollout_checkpoint.py",
    "mechanism_experiments/tropd_backfill/run_tropd_backfill.sh",
    "mechanism_experiments/tropd_backfill/protocol_guard.py",
    "mechanism_experiments/tropd_backfill/run_qzbc_gpu1_checkpoint.sh",
    "mechanism_experiments/tropd_backfill/SERVER_COMMANDS_GPU1_CHECKPOINT.md",
    "mechanism_experiments/tropd_backfill/test_checkpoint_launcher.py",
)


def build() -> dict:
    stream = io.BytesIO()
    inventory = []
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name in FILES:
                path = ROOT / name
                if path.is_symlink():
                    raise ValueError(f"source symlink refused: {path}")
                data = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.mode = 0o700 if name.endswith(".sh") else 0o600
                info.size = len(data)
                info.mtime = 0
                archive.addfile(info, io.BytesIO(data))
                inventory.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    payload = stream.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    report = {"archive": NAME, "sha256": sha, "files": inventory,
              "contains_models_or_dataset": False, "gpu_experiments_run_locally": False}
    output = Path(__file__).resolve().parent / "dist"
    output.mkdir(exist_ok=True)
    artifacts = {
        NAME: payload,
        NAME + ".sha256": f"{sha}  {NAME}\n".encode(),
        NAME + ".manifest.json": (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode(),
    }
    for name, data in artifacts.items():
        dest = output / name
        if dest.exists():
            if dest.read_bytes() != data:
                raise ValueError(f"refusing to replace an already published bundle: {dest}")
        else:
            with dest.open("xb") as handle:
                handle.write(data)
    print(json.dumps({"path": str(output / NAME), "bytes": len(payload), **report}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
