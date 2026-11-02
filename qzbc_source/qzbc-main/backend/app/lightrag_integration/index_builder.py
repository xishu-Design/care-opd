from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.lightrag_integration.config import DEFAULT_REPORTS_DIR, get_lightrag_config
from app.lightrag_integration.document_builder import load_lightrag_docs


def build_lightrag_index(
    *,
    docs_path: Path | str | None = None,
    working_dir: Path | str | None = None,
    report_path: Path | str = DEFAULT_REPORTS_DIR / "lightrag_index_report.json",
    rebuild: bool = False,
) -> dict[str, Any]:
    started_at = _now()
    config = get_lightrag_config()
    docs = load_lightrag_docs(docs_path)
    workdir = Path(working_dir) if working_dir else config.working_dir
    workdir.mkdir(parents=True, exist_ok=True)
    index_path = workdir / "guardtcm_lightrag_index.jsonl"
    if rebuild and index_path.exists():
        index_path.unlink()

    errors: list[str] = []
    docs_indexed = 0
    with index_path.open("w", encoding="utf-8") as file:
        for doc in docs:
            try:
                file.write(json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
                docs_indexed += 1
            except Exception as exc:  # pragma: no cover - defensive report path
                errors.append(f"{doc.doc_id}: {exc}")

    status = "success" if docs and not errors else "degraded" if not docs else "failed"
    if not docs:
        errors.append("no_lightrag_docs_found")
    report = {
        "run_id": f"lightrag_index_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "started_at": started_at,
        "ended_at": _now(),
        "docs_total": len(docs),
        "docs_indexed": docs_indexed,
        "failed": len(errors),
        "working_dir": str(workdir),
        "index_path": str(index_path),
        "backend": "lightrag",
        "mode": config.mode,
        "status": status,
        "errors": errors,
    }
    _write_json(Path(report_path), report)
    return report


def read_index_report(path: Path | str = DEFAULT_REPORTS_DIR / "lightrag_index_report.json") -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {"status": "not_run", "docs_total": 0, "docs_indexed": 0, "errors": []}
    with report_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {"status": "invalid", "errors": ["report_not_object"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GuardTCM LightRAG local index staging files.")
    parser.add_argument("--docs-path")
    parser.add_argument("--working-dir")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    report = build_lightrag_index(docs_path=args.docs_path, working_dir=args.working_dir, rebuild=args.rebuild)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
