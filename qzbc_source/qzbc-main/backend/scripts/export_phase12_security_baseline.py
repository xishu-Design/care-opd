from __future__ import annotations

"""Run and export the public aggregate for the Phase 12 security baseline.

This is intentionally a fixed synthetic development regression, not an adaptive
red-team benchmark.  The exporter writes aggregate counts, metrics and hashes;
it never exports case text, user data, prompts, responses or database rows.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import EvaluationResult
from app.evaluation.dataset_loader import DATASET_DIR, DATASET_FILES, dataset_manifest
from app.evaluation.runner import EvaluationRunner


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "artifacts" / "phase12_security_baseline_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_files() -> list[Path]:
    app_root = BACKEND_ROOT / "app"
    files = [
        app_root / "evaluation" / "runner.py",
        app_root / "evaluation" / "dataset_loader.py",
        app_root / "evaluation" / "report.py",
        Path(__file__).resolve(),
    ]
    files.extend(DATASET_DIR / name for name in DATASET_FILES)
    return files


def _relative(path: Path) -> str:
    return path.relative_to(BACKEND_ROOT).as_posix()


def _protocol_binding() -> tuple[dict[str, str], str]:
    hashes = {_relative(path): _sha256(path) for path in _protocol_files()}
    canonical = json.dumps(hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return hashes, protocol_sha256


def _dataset_rows(db, run_id: int) -> list[dict[str, object]]:
    rows = db.scalars(
        select(EvaluationResult)
        .where(EvaluationResult.evaluation_run_id == run_id)
        .order_by(EvaluationResult.dataset, EvaluationResult.id)
    ).all()
    totals = Counter(row.dataset for row in rows)
    passed = Counter(row.dataset for row in rows if row.passed)
    return [
        {
            "dataset": dataset,
            "cases": totals[dataset],
            "passed": passed[dataset],
            "failed": totals[dataset] - passed[dataset],
            "pass_rate": round(passed[dataset] / totals[dataset], 4),
        }
        for dataset in sorted(totals)
    ]


def _public_payload(db, run) -> dict[str, object]:
    hashes, protocol_sha256 = _protocol_binding()
    manifest = dataset_manifest()
    return {
        "schema_version": "guardtcm.security-development-baseline.v1",
        "status": "COMPLETE" if run.failed == 0 else "COMPLETE_WITH_FAILURES",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_scope": "fixed_synthetic_system_security_regression",
        "adaptive_red_team": False,
        "dataset_version": run.dataset_version,
        "protocol_sha256": protocol_sha256,
        "protocol_files": hashes,
        "dataset_manifest": manifest,
        "overall": {
            "total_cases": run.total_cases,
            "passed": run.passed,
            "failed": run.failed,
            "pass_rate": run.pass_rate,
            "duration_ms": run.duration_ms,
        },
        "datasets": _dataset_rows(db, run.id),
        "aggregate_metrics": run.metrics_json,
        "claim_boundary": (
            "The result validates the checked deterministic application controls on a fixed "
            "synthetic development suite. It does not establish resistance to adaptive attacks, "
            "host compromise, supply-chain attacks, or clinical safety."
        ),
        "privacy": {
            "case_text_exported": False,
            "prompt_or_response_exported": False,
            "user_or_patient_data_exported": False,
            "per_case_identifiers_exported": False,
        },
    }


def _render_markdown(payload: dict[str, object]) -> str:
    overall = payload["overall"]
    metrics = payload["aggregate_metrics"]
    safety = metrics["safety"]
    evidence = metrics["evidence"]
    agent = metrics["agent"]
    llm = metrics["llm"]
    rows = "\n".join(
        f"| {item['dataset']} | {item['passed']}/{item['cases']} | {item['pass_rate']:.1%} |"
        for item in payload["datasets"]
    )
    return f"""# GuardTCM Phase 12 安全攻防开发基线

状态：**{payload['status']}**  
协议：`{payload['dataset_version']}`  
协议指纹：`{payload['protocol_sha256']}`

## 聚合结果

固定合成开发集共运行 {overall['total_cases']} 例，通过 {overall['passed']} 例、失败 {overall['failed']} 例，通过率 {overall['pass_rate']:.1%}。

| 数据集 | 通过/总数 | 通过率 |
|---|---:|---:|
{rows}

关键受控指标：

- 患者高风险请求前置拒答率：{safety['patient_high_risk_refusal_rate']:.1%}；
- MCP 未授权工具调用拦截率：{safety['mcp_unauthorized_block_rate']:.1%}；
- Evidence Packet `source_id` 覆盖率：{evidence['evidence_source_id_coverage_rate']:.1%}；
- HumanConfirm 中断率 / 恢复成功率：{agent['human_confirm_interrupt_rate']:.1%} / {agent['resume_success_rate']:.1%}；
- 结构化输出 schema 有效率：{llm['schema_valid_rate']:.1%}。

## 结论范围

这是面向基础开发闭环的固定合成安全回归，验证当前用例覆盖到的角色、工具、证据、记忆、确认与输出控制。它不是自适应提示注入或生产环境渗透测试，也不证明主机、供应链或临床安全。公开结果仅含聚合统计与协议哈希，不含逐例文本、提示、响应或用户数据。
"""


def export(output_dir: Path) -> tuple[Path, Path, Path]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    try:
        with SessionLocal() as db:
            run = EvaluationRunner(db, suite="full", reset_demo_data=True).run()
            payload = _public_payload(db, run)
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase12_security_baseline.public.json"
    markdown_path = output_dir / "PHASE12_SECURITY_BASELINE.md"
    checksums_path = output_dir / "SHA256SUMS.txt"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8", newline="\n")
    checksums_path.write_text(
        f"{_sha256(json_path)}  {json_path.name}\n{_sha256(markdown_path)}  {markdown_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    return json_path, markdown_path, checksums_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in export(args.output_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
