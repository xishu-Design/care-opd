#!/usr/bin/env python3
"""Prepare a reproducible, method-blind expert-review package.

This script consumes a locked real-thyroid QZBC benchmark and one private case
output JSONL per model.  It emits:

* ``blind_queue.private.jsonl``: clinical context plus shuffled candidates;
* ``blind_key.private.jsonl``: the only candidate-to-model/record mapping;
* ``review_scores.private.csv``: a blank long-form double-review template;
* ``review_manifest.private.json``: the complete private analysis lock; and
* ``public_manifest.json``: counts and file hashes only.

No model mapping, source identifiers, report text, or model response is written
to the public manifest.  The generated package contains no human ratings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "care_opd.competition.blind_review.v1"
DEFAULT_VARIANTS = ("clean", "source_outage", "evidence_conflict")
SCORE_COLUMNS = (
    "factual_fidelity_0_4",
    "critical_omission_0_1",
    "unsupported_major_claim_0_1",
    "safe_correct_0_1",
    "harm_severity_0_3",
    "role_fit_0_4",
    "evidence_handling_0_4",
    "review_burden_0_4",
    "overall_quality_1_5",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _opaque_token(secret: bytes, namespace: str, value: str, length: int = 24) -> str:
    payload = f"{namespace}\x00{value}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()[:length]


def _parse_case_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--case must have the form MODEL=PATH")
    model, raw_path = spec.split("=", 1)
    model = model.strip()
    if not model or not raw_path.strip():
        raise argparse.ArgumentTypeError("--case must have non-empty MODEL and PATH")
    return model, Path(raw_path).expanduser().resolve()


def _split_values(values: list[str] | None, default: tuple[str, ...] = ()) -> list[str]:
    if not values:
        return list(default)
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(result))


def _row_key(row: dict[str, Any]) -> str:
    for name in ("case_id", "id", "example_id"):
        value = row.get(name)
        if isinstance(value, str) and value:
            return value
    raise ValueError("benchmark row has no usable case_id/id/example_id")


def _family_key(row: dict[str, Any]) -> str:
    for name in ("report_group_hmac", "family_id"):
        value = row.get(name)
        if isinstance(value, str) and value:
            return value
    raise ValueError("benchmark row has no report_group_hmac/family_id")


def _ensure_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in index:
            raise ValueError(f"duplicate case key in {label}: {key}")
        index[key] = row
    return index


def _select_families(
    rows: list[dict[str, Any]], mode: str, sample_size: int | None, seed: int
) -> tuple[set[str], dict[str, Any]]:
    families = sorted({_family_key(row) for row in rows})
    if mode == "full":
        if sample_size is not None:
            raise ValueError("--sample-size is only valid with --sample-mode preregistered")
        selected = families
    else:
        if sample_size is None or sample_size <= 0:
            raise ValueError("--sample-size must be positive in preregistered mode")
        if sample_size > len(families):
            raise ValueError(
                f"sample size {sample_size} exceeds available family count {len(families)}"
            )
        selected = list(families)
        random.Random(seed).shuffle(selected)
        selected = sorted(selected[:sample_size])
    selection_digest = hashlib.sha256(
        "\n".join(selected).encode("utf-8")
    ).hexdigest()
    return set(selected), {
        "mode": mode,
        "strategy": (
            "census"
            if mode == "full"
            else "simple_random_without_replacement_at_report_group_level"
        ),
        "stratified": False,
        "available_family_count": len(families),
        "selected_family_count": len(selected),
        "selection_seed": seed if mode == "preregistered" else None,
        "selected_family_digest": selection_digest,
    }


def prepare(args: argparse.Namespace) -> dict[str, Path]:
    locked_path = args.locked.expanduser().resolve()
    if not locked_path.is_file():
        raise FileNotFoundError(locked_path)
    case_specs = [_parse_case_spec(spec) for spec in args.case]
    model_names = [model for model, _ in case_specs]
    if len(set(model_names)) != len(model_names):
        raise ValueError("duplicate model labels in --case")
    if len(model_names) < 2:
        raise ValueError("at least two --case MODEL=PATH inputs are required")
    for _, path in case_specs:
        if not path.is_file():
            raise FileNotFoundError(path)

    reviewers = _split_values(args.reviewers, ("reviewer_1", "reviewer_2"))
    if len(reviewers) != 2 or len(set(reviewers)) != 2:
        raise ValueError("exactly two distinct reviewer IDs are required")
    variants = _split_values(args.variant, DEFAULT_VARIANTS)

    locked_rows = _read_jsonl(locked_path)
    available_tracks = sorted(
        {
            str(row.get("evaluation_track", ""))
            for row in locked_rows
            if row.get("evaluation_track")
        }
    )
    track = args.track
    if track is None:
        if len(available_tracks) != 1:
            raise ValueError(
                "--track is required when the locked file contains zero or multiple tracks"
            )
        track = available_tracks[0]
    locked_rows = [
        row
        for row in locked_rows
        if row.get("evaluation_track") == track and row.get("variant") in variants
    ]
    if not locked_rows:
        raise ValueError("no locked rows remain after track/variant filtering")
    missing_variants = sorted(set(variants) - {str(row.get("variant")) for row in locked_rows})
    if missing_variants:
        raise ValueError(f"requested variants absent from locked input: {missing_variants}")

    selected_families, sampling = _select_families(
        locked_rows, args.sample_mode, args.sample_size, args.sample_seed
    )
    locked_rows = [row for row in locked_rows if _family_key(row) in selected_families]
    locked_index = _ensure_unique(locked_rows, "locked input")

    model_indexes: dict[str, dict[str, dict[str, Any]]] = {}
    input_hashes = {"locked": _sha256_file(locked_path)}
    for model, path in case_specs:
        rows = _read_jsonl(path)
        rows = [
            row
            for row in rows
            if row.get("evaluation_track") == track
            and row.get("variant") in variants
            and _family_key(row) in selected_families
        ]
        model_indexes[model] = _ensure_unique(rows, f"model {model}")
        expected = set(locked_index)
        observed = set(model_indexes[model])
        if expected != observed:
            raise ValueError(
                f"case coverage mismatch for {model}: "
                f"missing={len(expected - observed)}, extra={len(observed - expected)}"
            )
        input_hashes[f"case:{model}"] = _sha256_file(path)

    secret_material = json.dumps(
        {
            "blind_seed": args.blind_seed,
            "input_hashes": input_hashes,
            "track": track,
            "variants": variants,
            "sampling_digest": sampling["selected_family_digest"],
        },
        sort_keys=True,
    ).encode("utf-8")
    secret = hashlib.sha256(secret_material).digest()

    unit_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for case_key in sorted(locked_index):
        source = locked_index[case_key]
        family = _family_key(source)
        unit_id = "U-" + _opaque_token(secret, "unit", f"{family}|{case_key}")
        candidate_models = list(model_names)
        order_seed = int.from_bytes(
            hmac.new(secret, f"order\x00{case_key}".encode(), hashlib.sha256).digest()[:8],
            "big",
        )
        random.Random(order_seed).shuffle(candidate_models)
        candidates: list[dict[str, Any]] = []
        for position, model in enumerate(candidate_models, start=1):
            case = model_indexes[model][case_key]
            response = case.get("raw_response_private")
            if response is None:
                response = ""
            if not isinstance(response, str):
                raise ValueError(f"raw_response_private must be a string for model {model}")
            response_sha = hashlib.sha256(response.encode("utf-8")).hexdigest()
            recorded_sha = case.get("response_sha256")
            if recorded_sha and recorded_sha != response_sha:
                raise ValueError(f"response hash mismatch for model {model}, case {case_key}")
            code = f"C{position}"
            candidates.append(
                {
                    "candidate_code": code,
                    "response_private": response,
                    "response_sha256": response_sha,
                    "generation_error_private": case.get("generation_error"),
                }
            )
            key_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "review_unit_id": unit_id,
                    "candidate_code": code,
                    "model_label_private": model,
                    "case_id_private": case_key,
                    "family_id_private": source.get("family_id"),
                    "report_group_hmac_private": family,
                    "source_id_private": source.get("source_id"),
                    "evaluation_track": track,
                    "variant": source.get("variant"),
                    "modality": source.get("modality"),
                    "response_sha256": response_sha,
                }
            )
        unit_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "review_unit_id": unit_id,
                "evaluation_track": track,
                "variant": source.get("variant"),
                "modality": source.get("modality"),
                "clinical_context_private": source.get("request"),
                "candidates": candidates,
            }
        )

    queue_seed = int.from_bytes(
        hmac.new(secret, b"queue-order", hashlib.sha256).digest()[:8], "big"
    )
    random.Random(queue_seed).shuffle(unit_rows)
    unit_order = {row["review_unit_id"]: index for index, row in enumerate(unit_rows)}
    key_rows.sort(key=lambda row: (unit_order[row["review_unit_id"]], row["candidate_code"]))

    private_dir = args.output_private.expanduser().resolve()
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir = (
        args.output_public.expanduser().resolve()
        if args.output_public
        else private_dir / "public_safe"
    )
    public_dir.mkdir(parents=True, exist_ok=True)
    if private_dir == public_dir:
        raise ValueError("private and public output directories must be different")

    queue_path = private_dir / "blind_queue.private.jsonl"
    key_path = private_dir / "blind_key.private.jsonl"
    scores_path = private_dir / "review_scores.private.csv"
    private_manifest_path = private_dir / "review_manifest.private.json"
    public_manifest_path = public_dir / "public_manifest.json"

    _write_jsonl(queue_path, unit_rows)
    _write_jsonl(key_path, key_rows)
    score_header = [
        "reviewer_id",
        "review_unit_id",
        "candidate_code",
        "evaluation_track",
        "variant",
        *SCORE_COLUMNS,
        "review_notes_private",
        "completed_at_utc",
    ]
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_header)
        writer.writeheader()
        for reviewer in reviewers:
            for unit in unit_rows:
                for candidate in unit["candidates"]:
                    row = {name: "" for name in score_header}
                    row.update(
                        {
                            "reviewer_id": reviewer,
                            "review_unit_id": unit["review_unit_id"],
                            "candidate_code": candidate["candidate_code"],
                            "evaluation_track": unit["evaluation_track"],
                            "variant": unit["variant"],
                        }
                    )
                    writer.writerow(row)

    variant_counts = Counter(str(row["variant"]) for row in unit_rows)
    private_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "privacy": "PRIVATE: contains input paths, model labels, seeds, and mapping hashes",
        "evaluation_track": track,
        "variants": variants,
        "models_private": model_names,
        "reviewers_private": reviewers,
        "blind_seed": args.blind_seed,
        "sampling": sampling,
        "input_paths_private": {
            "locked": str(locked_path),
            "cases": {model: str(path) for model, path in case_specs},
        },
        "input_hashes": input_hashes,
        "unit_count": len(unit_rows),
        "candidate_count": len(key_rows),
        "score_row_count": len(key_rows) * len(reviewers),
        "variant_counts": dict(sorted(variant_counts.items())),
        "score_columns": list(SCORE_COLUMNS),
        "analysis_lock": {
            "primary_binary_endpoint": "safe_correct_0_1",
            "win_tie_loss_endpoint": "mean_reviewer_overall_quality_1_5",
            "bootstrap_cluster": "report_group_hmac_private",
            "binary_test": "two-sided exact McNemar",
            "ordinal_agreement": "quadratic-weighted Cohen kappa",
            "binary_disagreements": "third-reviewer adjudication required",
        },
    }
    _write_json(private_manifest_path, private_manifest)

    private_hashes = {
        path.name: _sha256_file(path)
        for path in (queue_path, key_path, scores_path, private_manifest_path)
    }
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": private_manifest["created_utc"],
        "status": "BLIND_REVIEW_QUEUE_PREPARED_NO_RATINGS",
        "data_scope": {
            "origin": "real-world medical records provided by Benyuan Zhishu",
            "processing": "de-identified and structured",
            "training_use": "external evaluation only; not used for model training",
            "evaluation_track": track,
            "variants": variants,
            "selected_report_group_count": sampling["selected_family_count"],
            "review_unit_count": len(unit_rows),
            "candidate_method_count": len(model_names),
            "reviewer_count": len(reviewers),
        },
        "private_artifact_sha256": private_hashes,
        "privacy_statement": (
            "This public manifest contains no model-to-code mapping, source/report/case "
            "identifier, clinical text, model response, or human score."
        ),
    }
    _write_json(public_manifest_path, public_manifest)
    return {
        "queue": queue_path,
        "key": key_path,
        "scores": scores_path,
        "private_manifest": private_manifest_path,
        "public_manifest": public_manifest_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked", required=True, type=Path)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="repeatable MODEL=PATH private case-output specification",
    )
    parser.add_argument("--track", help="evaluation_track filter; inferred if unique")
    parser.add_argument(
        "--variant",
        action="append",
        help="repeatable or comma-separated variant filter",
    )
    parser.add_argument(
        "--sample-mode", choices=("full", "preregistered"), default="full"
    )
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=20260824)
    parser.add_argument("--blind-seed", type=int, default=20260824)
    parser.add_argument(
        "--reviewers",
        action="append",
        help="exactly two IDs, repeatable or comma-separated",
    )
    parser.add_argument("--output-private", required=True, type=Path)
    parser.add_argument("--output-public", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = prepare(args)
    # Paths and hashes are safe; never print queue contents or mapping rows.
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
