from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.data_pipeline.validators.license_validator import DEFAULT_RAW_DIR


def inspect_raw_source(source_id: str, raw_dir: Path | str = DEFAULT_RAW_DIR) -> dict[str, Any]:
    source_path = Path(raw_dir) / source_id
    files: list[dict[str, Any]] = []
    if not source_path.exists():
        return {"source_id": source_id, "status": "missing", "files": []}
    for path in sorted(item for item in source_path.rglob("*") if item.is_file()):
        rel_path = path.relative_to(source_path)
        suffix = path.suffix.lower()
        files.append(
            {
                "path": str(rel_path),
                "type": _file_type(suffix),
                "size_bytes": path.stat().st_size,
                "candidate_parser": _candidate_parser(suffix),
            }
        )
    return {"source_id": source_id, "status": "ok", "files": files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect files under raw/{source_id}.")
    parser.add_argument("source_id")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_raw_source(args.source_id), ensure_ascii=False, indent=2))
    return 0


def _file_type(suffix: str) -> str:
    if suffix == ".csv":
        return "csv"
    if suffix in {".txt", ".text"}:
        return "txt"
    if suffix == ".json":
        return "json"
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        return "sqlite"
    return "unknown"


def _candidate_parser(suffix: str) -> str:
    return _file_type(suffix) if _file_type(suffix) != "unknown" else ""


if __name__ == "__main__":
    raise SystemExit(main())
