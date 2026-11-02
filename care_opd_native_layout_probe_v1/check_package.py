"""Verify all packaged source bytes, without importing model libraries."""
import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != "care_opd.native_layout_probe.package.v1":
        raise ValueError("unsupported package")
    for name, metadata in manifest["files"].items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("missing/unsafe package file: " + name)
        data = file.read_bytes()
        if len(data) != metadata["bytes"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise ValueError("package changed: " + name)
    print("PACKAGE_OK", len(manifest["files"]), "files")
    return manifest


if __name__ == "__main__":
    verify()
