#!/usr/bin/env python3
"""Build the deterministic held-out local_product overlay for the GPU-6 run."""

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
    "competition_evidence/three_role_eval_common.py",
    "competition_evidence/validate_local_four_agent_run.py",
    "competition_evidence/prepare_agent_preference_review.py",
    "competition_evidence/summarize_agent_preference_review.py",
    "competition_evidence/local_four_agent_server/model_suite.json",
    "competition_evidence/local_four_agent_server/run_four_agent_local_server.sh",
    "competition_evidence/local_four_agent_server/serve_care_transformers.py",
    "competition_evidence/tcm_best4sdt_agent30/README.md",
    "competition_evidence/tcm_best4sdt_agent30/SERVER_COMMANDS_HOLDOUT_GPU6.md",
    "competition_evidence/tcm_best4sdt_agent30/audit_runtime_retrieval_overlap.py",
    # Retained only so the separately documented LightRAG enhancement track
    # remains resumable; the local_product primary track never invokes it.
    "competition_evidence/tcm_best4sdt_agent30/bootstrap_full_product_lightrag.py",
    "competition_evidence/tcm_best4sdt_agent30/prepare_tcm_best4sdt_agent30.py",
    "competition_evidence/tcm_best4sdt_agent30/render_review_reading_packet.py",
    "competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh",
    "competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu5.sh",
    "competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu6.sh",
    "competition_evidence/tcm_best4sdt_agent30/test_case_reasoning_quality_gate.py",
    "competition_evidence/tcm_best4sdt_agent30/test_live_product_contract.py",
    "competition_evidence/tcm_best4sdt_agent30/validate_case_reasoning_quality.py",
    "public_eval/tcm_best4sdt/data/DATA_PROVENANCE.md",
    "public_eval/tcm_best4sdt/data/FIGSHARE_METADATA.json",
)

FROZEN_TASK_FILES = (
    "tcm_best4sdt_agent30.tasks.private.jsonl",
    "tcm_best4sdt_agent30.selection_manifest.json",
    "tcm_best4sdt_agent30.overlap_audit.json",
)
FROZEN_TASK_SOURCE_ROOT = Path(
    "runs/public_eval/tcm_best4sdt_agent30_holdout_fixed_seed20260827"
)
FROZEN_TASK_ARCHIVE_ROOT = Path(
    "runs/public_eval/tcm_best4sdt_agent30_holdout_reference_free_fixed_seed20260827"
)

QZBC_EXCLUDE_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "runtime",
}
QZBC_EXCLUDE_SUFFIXES = {
    ".bak",
    ".backup",
    ".db",
    ".gz",
    ".log",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".xls",
    ".xlsx",
    ".zip",
}


def care_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qzbc_sources(root: Path) -> list[tuple[Path, str]]:
    """Return the exact, secret-free backend snapshot used by the project arm."""

    qzbc_root = root.parent / "qzbc_source" / "qzbc-main"
    if not (qzbc_root / "backend" / "app" / "main.py").is_file():
        raise FileNotFoundError(qzbc_root / "backend" / "app" / "main.py")
    prefix = Path("competition_evidence/tcm_best4sdt_agent30/qzbc_full_product")
    selected: list[tuple[Path, str]] = []
    for relative in ("requirements.txt", "environment.yml"):
        source = qzbc_root / relative
        if source.is_file():
            selected.append((source, (prefix / relative).as_posix()))
    for source in sorted((qzbc_root / "backend").rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(qzbc_root)
        if any(part in QZBC_EXCLUDE_PARTS for part in relative.parts):
            continue
        if source.suffix.lower() in QZBC_EXCLUDE_SUFFIXES:
            continue
        if source.name.lower().startswith(".env"):
            continue
        if ".private." in source.name.lower():
            continue
        selected.append((source, (prefix / relative).as_posix()))
    return selected


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.care_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    sha_path = output.with_name(output.name + ".sha256")
    if output.exists() or sha_path.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle or checksum: {output}")
    entries: list[dict[str, Any]] = []
    sources: list[tuple[Path, str]] = []
    for relative in FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        sources.append((source, relative))
    for filename in FROZEN_TASK_FILES:
        source = root / FROZEN_TASK_SOURCE_ROOT / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        sources.append((source, (FROZEN_TASK_ARCHIVE_ROOT / filename).as_posix()))
    sources.extend(qzbc_sources(root))
    arc_paths = [arc_path for _, arc_path in sources]
    if len(arc_paths) != len(set(arc_paths)):
        raise RuntimeError("bundle contains duplicate archive paths")
    for source, arc_path in sources:
        entries.append(
            {
                "path_relative_to_care_root": arc_path,
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    contents = {
        "schema_version": "care_opd.competition.tcm_best4sdt_agent30_holdout_local_product_gpu6_fair_blind_review_bundle.v22",
        "status": "COMPLETE",
        "evaluation_profile": "local_product",
        "output_selection_policy": "first_user_visible_output_frozen_content_neutral",
        "content_triggered_regeneration": False,
        "content_review_policy": (
            "audit_only_expert_judged_no_content_triggered_regeneration"
        ),
        "legacy_baseline_reuse_requires_attempt_count_one": False,
        "all_four_systems_run_on_the_same_new_heldout_tasks": True,
        "prior_development_outputs_reused": False,
        "default_physical_gpu": 6,
        "active_gpu_process_check_policy": (
            "fail_closed_unless_operator_sets_exact_contention_risk_ack"
        ),
        "description": (
            "final held-out expert-blind track using the real role route, CARE-OPD, "
            "Graphiti, Mem0, SQL conversation context and frozen full content/data local "
            "structured retrieval; neutral role preference is written and read back through "
            "the first-party Mem0 interface; the already-structured neutral relation is "
            "persisted and queried through Graphiti/FalkorDB without redundant LLM extraction; "
            "CARE uses a dedicated 32k context while baselines remain on their frozen 8k "
            "protocol; all four systems start cleanly on the same newly frozen holdout tasks, "
            "and a single-writer lock prevents "
            "concurrent checkpoint corruption; provider and Graphiti failure labels retain "
            "content-free diagnostic codes, while any already-visible private response remains "
            "hash-frozen and is never replaced; medical case prompts are preflight-verified not "
            "to short-circuit into the preference-memory policy; comparison-model vLLM servers "
            "use the supported native sampler fallback to avoid the frozen image's incompatible "
            "FlashInfer JIT toolchain; HuatuoGPT-3 omits vLLM's qwen3 reasoning parser because "
            "its released tokenizer has no required think-delimiter tokens, while the frozen "
            "model selection remains unchanged; InternLM3 enables the custom architecture code "
            "from its immutable local snapshot because vLLM requires it to load that checkpoint; "
            "the complete anonymized teaching case is bound to the current doctor request as a "
            "hash-verified Evidence Packet item without bypassing Graphiti, Mem0, SQL context or "
            "local knowledge retrieval; the complete packet remains persisted for audit while a deterministic "
            "bounded prompt projection retains the full case and compact substantive fields from every retrieved "
            "knowledge item but removes repository paths, repeated source records and long keyword inventories; "
            "a dedicated uncertain-teaching-inference contract replaces "
            "the former evidence-status-only refusal while prohibiting invented facts, fixed doses, "
            "clinical finality and reference-answer injection; the evaluation-task portion of the "
            "server overlay carries only the checksum-frozen, reference-free TCM-BEST4SDT 30-task "
            "artifact and its pre-generation selection/audit metadata, while the full TCM-BEST4SDT "
            "300-case source and its independent reference artifact are physically absent and never "
            "opened at inference time; the frozen QZBC product snapshot can contain unrelated public "
            "data resources, but they are outside the active retrieval-source set for this track; the "
            "case output contract supplies "
            "all six required section objects with the request-bound case source, instead of the prior "
            "one-item array example; a deterministic compatibility recovery splits an older one-section "
            "response only when all six requested headings and their original provenance are present, "
            "and never invents missing content; the expert-blind protocol now freezes the first user-visible "
            "answer for every system and task. Structure, source citation, candidate specificity, fact-to-"
            "hypothesis links, treatment/formula direction, review target, abstention, length and safety "
            "properties are recorded as selection-neutral audit dimensions and never trigger answer rejection, "
            "regeneration or eligibility filtering. Only execution identity, task/hash binding, no-gold access, "
            "provider completion and persistence integrity remain hard prerequisites; this prevents project-only "
            "best-of-three selection before expert review. Infrastructure retries may occur only when the backend "
            "proves that no successful model generation exists; provider-failure sentinel strings are never "
            "misclassified as answers, while the first answer after a successful generation is frozen; the "
            "OpenAI-compatible parser "
            "enumerates later JSON objects and performs only "
            "content-preserving recovery for literal controls, trailing collection commas and clean EOF "
            "container closure, while malformed strings remain fail-closed; parser failures now retain the "
            "correct model and content-free parse/schema status; the documented recovery path preserves all "
            "prior 30-case outputs unchanged as development audits, while the final sample excludes both their "
            "benchmark IDs and normalized diagnosis hashes before any new inference; all four systems run from "
            "scratch in a new holdout root; generation "
            "aborts on a failed first task in the checksum-frozen pre-generation order; that task is "
            "audited as the holdout-specific operational canary rather than reusing a development-set ID. "
            "Blind-review packaging is blocked until both the four-system execution "
            "audit and the independent 30-case project quality report match their exact input hashes; "
            "LightRAG bootstrap is not required"
        ),
        "install_instruction": "extract this archive directly into the existing CARE_ROOT",
        "privacy": (
            "contains checksum-frozen public case stems without answer/diagnosis fields, "
            "pre-generation selection/provenance metadata, prompts, scripts and a frozen QZBC "
            "product code/content snapshot that can include unrelated public resources; contains no "
            "full TCM-BEST4SDT 300-case source, independent TCM-BEST4SDT reference artifact, model "
            "output, expert rating, blind secret, model-code mapping or private medical record"
        ),
        "evaluation_benchmark": "TCM-BEST4SDT",
        "evaluation_benchmark_upstream_source_contains_reference_answers": True,
        "evaluation_benchmark_full_300_case_source_physically_present_in_bundle": False,
        "evaluation_benchmark_independent_reference_answers_physically_present_in_bundle": False,
        "evaluation_task_file_or_model_request_contains_reference_answers": False,
        "frozen_product_snapshot_can_contain_unrelated_public_data_resources": True,
        "files": entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tcm_best_agent30_") as temporary:
        contents_path = Path(temporary) / "TCM_BEST4SDT_AGENT30_BUNDLE_CONTENTS.json"
        contents_path.write_text(
            json.dumps(contents, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with output.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
            source_by_arc = {arc_path: source for source, arc_path in sources}
            for entry in entries:
                relative = entry["path_relative_to_care_root"]
                source = source_by_arc[relative]
                info = archive.gettarinfo(str(source), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mtime = 0
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
            info = archive.gettarinfo(
                str(contents_path), arcname="TCM_BEST4SDT_AGENT30_BUNDLE_CONTENTS.json"
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


def build_parser() -> argparse.ArgumentParser:
    root = care_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--care-root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "runs"
            / "public_eval"
            / "tcm_best4sdt_agent30_bundle"
            / "TCM_BEST4SDT_AGENT30_HOLDOUT_GPU6_FROZEN_CANARY_XLSX_REVIEW_FIX_FINAL.tar.gz"
        ),
    )
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
