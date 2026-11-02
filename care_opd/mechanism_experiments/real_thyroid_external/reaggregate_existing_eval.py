#!/usr/bin/env python3
"""Rebuild public thyroid-evaluation aggregates without model inference.

The completed private JSONL rows are the immutable source of truth.  This
utility validates their binding to the original private run manifest,
recomputes every per-case metric from the stored verifier assessment, and
writes a new publication-safe directory.  It never edits the private manifest
or case files and never loads a model checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # Keep direct ``python mechanism_experiments/.../reaggregate_existing_eval.py``
    # invocation working without requiring the repository root in PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanism_experiments.real_thyroid_external import thyroid_external_eval as evaluation


AGGREGATION_REVISION = "three_axis_metrics_v4_p1_3"
DEFAULT_PUBLIC_DIRNAME = "public_reaggregated_v4_p1_3"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PUBLIC_FILENAMES = {
    "model_metrics.csv",
    "paired_comparisons.csv",
    "run_manifest.json",
    "secondary_field_metrics.csv",
    "three_axis_metrics.csv",
    "summary.json",
    "summary.md",
}


def _contract_error(message: str) -> evaluation.EvaluationContractError:
    return evaluation.EvaluationContractError(message)


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise _contract_error(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _contract_error(f"invalid {description}: {path}") from exc
    if not isinstance(value, Mapping):
        raise _contract_error(f"{description} must be a JSON object")
    return dict(value)


def _required_hash(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _HEX64.fullmatch(text):
        raise _contract_error(f"{field} must be a lowercase SHA-256 digest")
    return text


def _manifest_contract(
    run_dir: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], list[str], int, list[str], int, int, str, bool
]:
    private_manifest = _load_json_object(
        run_dir / "private" / "run_manifest.private.json",
        description="private run manifest",
    )
    public_manifest = _load_json_object(
        run_dir / "public" / "run_manifest.json",
        description="source public run manifest",
    )
    run_fingerprint = _required_hash(
        private_manifest.get("run_fingerprint"), field="private run_fingerprint"
    )
    models = private_manifest.get("models")
    if not isinstance(models, list) or not models:
        raise _contract_error("private manifest must contain a non-empty models list")
    labels: list[str] = []
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise _contract_error(f"private manifest model {index} is not an object")
        label = model.get("label")
        if not isinstance(label, str) or not _LABEL.fullmatch(label):
            raise _contract_error(f"private manifest model {index} has an unsafe label")
        if label in labels:
            raise _contract_error(f"duplicate private manifest model label {label!r}")
        _required_hash(model.get("fingerprint"), field=f"model {label!r} fingerprint")
        labels.append(label)

    input_contract = private_manifest.get("input")
    if not isinstance(input_contract, Mapping):
        raise _contract_error("private manifest lacks an input contract")
    case_count = input_contract.get("case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 1:
        raise _contract_error("private manifest case_count must be a positive integer")
    evaluation_track = str(
        input_contract.get("evaluation_track") or evaluation.LEGACY_EVALUATION_TRACK
    )
    if evaluation_track not in evaluation.EVALUATION_TRACKS:
        raise _contract_error("private manifest has an unsupported evaluation_track")
    expected_challenge = evaluation_track == "raw_strict_extraction_challenge"
    challenge_value = input_contract.get("challenge_mode")
    challenge_mode = expected_challenge if challenge_value is None else challenge_value
    if not isinstance(challenge_mode, bool) or challenge_mode != expected_challenge:
        raise _contract_error("private manifest evaluation_track/challenge_mode mismatch")
    for manifest_name, manifest in (
        ("private", private_manifest),
        ("public", public_manifest),
    ):
        declared_track = manifest.get("evaluation_track")
        declared_challenge = manifest.get("challenge_mode")
        if declared_track is not None and declared_track != evaluation_track:
            raise _contract_error(f"{manifest_name} manifest evaluation_track mismatch")
        if declared_challenge is not None and declared_challenge != challenge_mode:
            raise _contract_error(f"{manifest_name} manifest challenge_mode mismatch")
    bootstrap = private_manifest.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise _contract_error("private manifest lacks a bootstrap contract")
    replicates = bootstrap.get("replicates")
    seed = bootstrap.get("seed")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise _contract_error("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise _contract_error("bootstrap seed must be an integer")
    references_value = private_manifest.get("reference_models")
    if references_value is None:
        references_value = [private_manifest.get("reference_model")]
    if not isinstance(references_value, list) or not references_value:
        raise _contract_error("private manifest lacks reference models")
    references: list[str] = []
    for value in references_value:
        if not isinstance(value, str) or value not in labels:
            raise _contract_error(f"unknown reference model {value!r}")
        if value not in references:
            references.append(value)

    public_labels = [
        item.get("label")
        for item in public_manifest.get("models", [])
        if isinstance(item, Mapping)
    ]
    if public_labels != labels:
        raise _contract_error("public/private manifest model order mismatch")
    if public_manifest.get("case_count") != case_count:
        raise _contract_error("public/private manifest case_count mismatch")
    if public_manifest.get("reference_models") != references:
        raise _contract_error("public/private manifest reference-model mismatch")
    if public_manifest.get("bootstrap") != dict(bootstrap):
        raise _contract_error("public/private manifest bootstrap mismatch")
    if public_manifest.get("verifier_sha256") != private_manifest.get("verifier_sha256"):
        raise _contract_error("public/private verifier binding mismatch")
    _required_hash(
        public_manifest.get("public_protocol_fingerprint"),
        field="source public_protocol_fingerprint",
    )
    return (
        private_manifest,
        public_manifest,
        labels,
        case_count,
        references,
        replicates,
        seed,
        evaluation_track,
        challenge_mode,
    )


def _load_private_rows(
    path: Path,
    *,
    label: str,
    case_count: int,
    per_model_fingerprint: str,
    evaluation_track: str,
    challenge_mode: bool,
    expected_cases: Sequence[tuple[Any, ...]] | None,
) -> tuple[list[dict[str, Any]], list[tuple[Any, ...]]]:
    if not path.is_file():
        raise _contract_error(f"missing private case file for model {label!r}")
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise _contract_error(f"cannot read private case file for model {label!r}") from exc
    if len(lines) != case_count:
        raise _contract_error(
            f"model {label!r} has {len(lines)} private rows; expected {case_count}"
        )
    rows_by_index: dict[int, dict[str, Any]] = {}
    for line_number, encoded in enumerate(lines, 1):
        if not encoded.strip():
            raise _contract_error(f"blank private row for model {label!r} at line {line_number}")
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _contract_error(
                f"invalid private JSON for model {label!r} at line {line_number}"
            ) from exc
        if not isinstance(value, Mapping):
            raise _contract_error(
                f"private row for model {label!r} at line {line_number} is not an object"
            )
        row = dict(value)
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise _contract_error(f"model {label!r} line {line_number} has invalid index")
        if index < 0 or index >= case_count or index in rows_by_index:
            raise _contract_error(f"model {label!r} has duplicate/out-of-range index {index}")
        if row.get("model_label") != label:
            raise _contract_error(f"model-label mismatch for {label!r} index {index}")
        if row.get("run_fingerprint") != per_model_fingerprint:
            raise _contract_error(f"stale private row fingerprint for {label!r} index {index}")
        for field in ("case_id", "family_id", "variant", "modality"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise _contract_error(f"model {label!r} index {index} lacks {field}")
        _required_hash(
            row.get("seed_record_sha256"),
            field=f"model {label!r} index {index} seed_record_sha256",
        )
        raw_response = row.get("raw_response_private")
        if not isinstance(raw_response, str):
            raise _contract_error(f"model {label!r} index {index} lacks raw response")
        response_sha256 = _required_hash(
            row.get("response_sha256"),
            field=f"model {label!r} index {index} response_sha256",
        )
        if response_sha256 != evaluation._sha256_bytes(raw_response.encode("utf-8")):
            raise _contract_error(f"response hash mismatch for {label!r} index {index}")
        if not isinstance(row.get("assessment"), Mapping):
            raise _contract_error(f"model {label!r} index {index} lacks an assessment")
        row_track, row_challenge = evaluation._result_evaluation_contract(row)
        if row_track != evaluation_track or row_challenge != challenge_mode:
            raise _contract_error(
                f"model {label!r} index {index} evaluation-track contract mismatch"
            )
        if row.get("metrics") is not None and not isinstance(row.get("metrics"), Mapping):
            raise _contract_error(f"model {label!r} index {index} has invalid cached metrics")
        rows_by_index[index] = row

    expected_indices = set(range(case_count))
    if set(rows_by_index) != expected_indices:
        raise _contract_error(f"model {label!r} private rows are incomplete")
    rows = [rows_by_index[index] for index in range(case_count)]
    signatures = [
        (
            row["case_id"],
            row["family_id"],
            row["variant"],
            row["modality"],
            row["seed_record_sha256"],
            row.get("report_group_hmac"),
            row_track,
            row_challenge,
        )
        for row in rows
    ]
    if expected_cases is not None and list(expected_cases) != signatures:
        raise _contract_error(f"model {label!r} does not match the locked cross-model case order")
    return rows, signatures


def _prepare_target(target: Path, *, private_dir: Path, overwrite: bool) -> None:
    resolved_target = target.resolve()
    resolved_private = private_dir.resolve()
    if resolved_target == resolved_private or resolved_target.is_relative_to(resolved_private):
        raise _contract_error("public reaggregation output cannot be inside the private directory")
    if resolved_target.exists() and not resolved_target.is_dir():
        raise _contract_error("public reaggregation target exists and is not a directory")
    if not resolved_target.exists():
        return
    entries = {entry.name for entry in resolved_target.iterdir()}
    if entries and not overwrite:
        raise FileExistsError(
            f"{resolved_target} is not empty; pass --overwrite-public-output to replace known artifacts"
        )
    unknown = entries - _PUBLIC_FILENAMES
    if unknown:
        raise _contract_error(
            "refusing to write into a directory containing unknown artifacts: "
            + ", ".join(sorted(unknown))
        )


def reaggregate_existing_run(
    *,
    run_dir: Path,
    public_output_dir: Path | None = None,
    overwrite_public_output: bool = False,
    acknowledge_private_clinical_data: bool = False,
) -> dict[str, Any]:
    if not acknowledge_private_clinical_data:
        raise _contract_error(
            "private clinical reaggregation requires --acknowledge-private-clinical-data"
        )
    run_dir = run_dir.expanduser().resolve()
    (
        private_manifest,
        source_public_manifest,
        labels,
        case_count,
        reference_labels,
        replicates,
        bootstrap_seed,
        evaluation_track,
        challenge_mode,
    ) = _manifest_contract(run_dir)
    private_dir = run_dir / "private"
    target = (
        public_output_dir.expanduser().resolve()
        if public_output_dir is not None
        else run_dir / DEFAULT_PUBLIC_DIRNAME
    )
    _prepare_target(target, private_dir=private_dir, overwrite=overwrite_public_output)

    run_fingerprint = str(private_manifest["run_fingerprint"])
    results: dict[str, list[dict[str, Any]]] = {}
    case_signatures: list[tuple[Any, ...]] | None = None
    for label in labels:
        per_model_fingerprint = evaluation._sha256_bytes(
            evaluation._canonical_json({"run": run_fingerprint, "model": label})
        )
        rows, signatures = _load_private_rows(
            private_dir / f"{label}.cases.private.jsonl",
            label=label,
            case_count=case_count,
            per_model_fingerprint=per_model_fingerprint,
            evaluation_track=evaluation_track,
            challenge_mode=challenge_mode,
            expected_cases=case_signatures,
        )
        if case_signatures is None:
            case_signatures = signatures
        results[label] = rows

    summaries = [evaluation.summarize_model(label, results[label]) for label in labels]
    modalities = sorted({str(row["modality"]) for row in results[labels[0]]})
    comparisons = evaluation.build_paired_family_comparisons(
        results,
        reference_labels=reference_labels,
        modalities=modalities,
        replicates=replicates,
        seed=bootstrap_seed,
    )

    public_manifest = dict(source_public_manifest)
    source_protocol_fingerprint = str(public_manifest.pop("public_protocol_fingerprint"))
    public_manifest.update(
        {
            "schema_version": evaluation.SCHEMA_VERSION + ".public_manifest",
            "evaluator_sha256": evaluation._sha256_file(Path(evaluation.__file__).resolve()),
            "aggregation_revision": AGGREGATION_REVISION,
            "aggregation_changes": [
                "metrics_recomputed_from_stored_assessment",
                "tuple_and_list_unsupported_errors_supported",
                "field_f1_computed_directly_from_tp_fp_fn",
                "state_protocol_and_content_axes_reported_separately",
                "state_task_success_preferred_with_legacy_fallback",
                "bad_history_only_conditional_retention_added",
                "extraction_axes_scoped_to_clean_and_bad_history_only",
                "axis_scopes_and_denominators_published_explicitly",
            ],
            "evaluation_track": evaluation_track,
            "challenge_mode": challenge_mode,
            "source_public_protocol_fingerprint": source_protocol_fingerprint,
        }
    )
    public_manifest["public_protocol_fingerprint"] = evaluation._sha256_bytes(
        evaluation._canonical_json(public_manifest)
    )
    evaluation.write_public_outputs(
        public_dir=target,
        summaries=summaries,
        comparisons=comparisons,
        public_manifest=public_manifest,
    )
    return {
        "status": "REAGGREGATED",
        "aggregation_revision": AGGREGATION_REVISION,
        "models": labels,
        "case_count": case_count,
        "public_output": str(target),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed evaluation directory containing private/ and public/.",
    )
    parser.add_argument(
        "--public-output-dir",
        type=Path,
        help=f"Destination (default: RUN_DIR/{DEFAULT_PUBLIC_DIRNAME}).",
    )
    parser.add_argument("--overwrite-public-output", action="store_true")
    parser.add_argument("--acknowledge-private-clinical-data", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = reaggregate_existing_run(
        run_dir=args.run_dir,
        public_output_dir=args.public_output_dir,
        overwrite_public_output=args.overwrite_public_output,
        acknowledge_private_clinical_data=args.acknowledge_private_clinical_data,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
