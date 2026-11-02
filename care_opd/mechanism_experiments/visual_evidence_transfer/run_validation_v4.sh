#!/usr/bin/env bash
# Validation only. Old assets are read-only; every V4 inference output is new.
set -euo pipefail
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HF_PY="${HF_PY:-/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python}"
WORK="${WORK:-/opt/care-data/care_opd_tmm_visual_gpu45_v4}"
OLD="${OLD:-/opt/care-data/care_opd_tmm_visual_gpu45_v3}"
STATE=initializing
trap 'rc=$?; printf "V4_STATE=FAILED stage=%s exit=%s\n" "$STATE" "$rc" >&2; exit "$rc"' ERR
if (( $# != 0 )); then
  printf '%s\n' 'Usage: bash run_validation_v4.sh (optional environment: HF_PY, WORK, OLD)' >&2
  exit 2
fi
[[ -x "$HF_PY" ]] || { printf 'Missing interpreter: %s\n' "$HF_PY" >&2; exit 2; }
export HF_PY HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf 'V4_STATE=INITIALIZING work=%s old=%s source=%s\n' "$WORK" "$OLD" "$SCRIPT_DIR"

"$HF_PY" - "$SCRIPT_DIR" "$OLD" "$WORK" <<'PY'
import sys
from pathlib import Path

source_dir = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(source_dir))
import evaluate
from common import PROTOCOL_VERSION, TEXT_ARMS, file_sha256, load_tasks, read_json, write_json_once


def absolute(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"absolute path required: {value}")
    return path.resolve()


def within(path, parent):
    return path == parent or parent in path.parents


old, work = map(absolute, sys.argv[2:4])
if not old.is_dir():
    raise ValueError(f"missing read-only old experiment: {old}")
if within(work, old) or within(old, work):
    raise ValueError("WORK and OLD must be disjoint; old paths and aliases are forbidden")
if within(source_dir, work) or within(work, source_dir):
    raise ValueError("the source tree and output WORK must be separate")
if work.exists() and not work.is_dir():
    raise ValueError(f"WORK is not a directory: {work}")
# Check existing descendants, including dangling aliases, before any V4 write.
# This also protects resume paths (logs/observations/arms/summary/runtime).
if work.exists():
    for child in work.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"symlinks are not allowed inside V4 WORK: {child}")

tasks_dir = (old / "tasks_validation10").resolve(strict=True)
if not within(tasks_dir, old):
    raise ValueError("validation tasks alias a directory outside OLD")
tasks, selection = load_tasks(tasks_dir)
if selection.get("dataset", {}).get("split") != "validation":
    raise ValueError("tasks_validation10 is not the frozen validation split")
manifest_path = old / "runs_test" / "run_manifest.json"
if manifest_path.exists() or manifest_path.is_symlink():
    source_path = manifest_path
    prior = read_json(source_path)
    source = prior["config"]
    source_kind = "prior_test_run_manifest_config"
else:
    source_path = old / "config_validation10.json"
    source = read_json(source_path)
    source_kind = "prior_validation_config_fallback_manifest_absent"
# An existing but invalid manifest is an error, not a reason to use the fallback.
if not isinstance(source, dict) or source.get("schema_version") != "care_opd.visual_transfer.config.v1":
    raise ValueError("unsupported prior config")
if set(source.get("models", {})) != {"vlm", *TEXT_ARMS}:
    raise ValueError("prior config does not contain exactly the five required models")
generation = source.get("generation")
if not isinstance(generation, dict) or set(generation) != set(evaluate.DEFAULT_GENERATION):
    raise ValueError("prior config must contain all generation settings; do not silently substitute defaults")
models = {}
for name, value in source["models"].items():
    path = absolute(value)
    if within(path, work) or within(work, path):
        raise ValueError(f"read-only model and output WORK overlap: {name} {path}")
    evaluate.model_files(path)
    models[name] = str(path)

config_path = work / "config_validation10.json"
run_root = work / "runs_validation10"
provenance_path = work / "config_validation10.provenance.json"
for target in (config_path, run_root, provenance_path, work / "logs"):
    resolved = target.resolve()
    if not within(resolved, work) or within(resolved, old):
        raise ValueError(f"output path escapes V4 WORK or aliases OLD: {target}")
if not config_path.exists() and run_root.exists() and any(run_root.iterdir()):
    raise ValueError("new config requires a fresh empty run root; preserve existing outputs")
config = {
    "schema_version": "care_opd.visual_transfer.config.v1",
    "protocol_version": PROTOCOL_VERSION,
    "evaluation_use": "validation",
    "tasks_dir": str(tasks_dir),
    "run_root": str(run_root),
    "models": models,
    "generation": generation,
}
if PROTOCOL_VERSION != "care_opd.visual_transfer.v4":
    raise ValueError("this launcher requires the separately installed V4 source package")
provenance = {
    "schema_version": "care_opd.visual_transfer.config_provenance.v1",
    "protocol_version": PROTOCOL_VERSION,
    "evaluation_use": "validation",
    "prior_test_outputs_inspected": True,
    "source_kind": source_kind,
    "source_path": str(source_path.resolve(strict=True)),
    "source_sha256": file_sha256(source_path),
    "tasks_dir": str(tasks_dir),
    "selection_manifest_sha256": file_sha256(tasks_dir / "selection_manifest.json"),
    "tasks_sha256": selection["tasks_sha256"],
    "gold_sha256": selection["gold_sha256"],
    "source_dir": str(source_dir),
    "launcher_sha256": file_sha256(source_dir / "run_validation_v4.sh"),
    "models": models,
    "generation": generation,
    "old_outputs_policy": "read_only_not_reused_as_v4_observations_or_responses",
}
if config_path.exists() and read_json(config_path) != config:
    raise ValueError("existing V4 config differs; do not overwrite or mix runs")
if provenance_path.exists() and read_json(provenance_path) != provenance:
    raise ValueError("existing V4 provenance differs; preserve it and investigate")
existing_manifest = run_root / "run_manifest.json"
if existing_manifest.exists():
    bound = read_json(existing_manifest)
    if bound.get("protocol_version") != PROTOCOL_VERSION or bound.get("config") != config:
        raise ValueError("existing run is not bound to this exact V4 validation config")
elif run_root.exists() and any(run_root.iterdir()):
    raise ValueError("unbound nonempty run root; preserve it and investigate before resuming")
evaluate.validate_evaluation_use(config, selection)
write_json_once(config_path, config)
validated = evaluate.configuration(config_path)
if validated != config:
    raise ValueError("config normalization changed inherited settings or frozen paths")
write_json_once(provenance_path, provenance)
print(f"V4_CONFIG_READY config={config_path} tasks={len(tasks)}", flush=True)
print(f"V4_CONFIG_SOURCE kind={source_kind} path={source_path}", flush=True)
print(f"V4_GENERATION observation={generation['observation_max_new_tokens']} answer={generation['max_new_tokens']}", flush=True)
for name, path in models.items():
    print(f"V4_READ_ONLY_MODEL arm={name} path={path}", flush=True)
PY

CONFIG="$WORK/config_validation10.json"
STATE=preflight
printf 'V4_STATE=PREFLIGHT config=%s\n' "$CONFIG"
bash "$SCRIPT_DIR/run_visual_transfer_gpu45.sh" preflight "$CONFIG"
STATE=validation
printf 'V4_STATE=VALIDATION_RUNNING config=%s\n' "$CONFIG"
bash "$SCRIPT_DIR/run_visual_transfer_gpu45.sh" all "$CONFIG"
STATE=complete
printf 'V4_STATE=COMPLETE summary=%s/runs_validation10/summary\n' "$WORK"
printf '%s\n' 'V4_TEST_NOT_STARTED: same-test repeats require a separate manual post_test_protocol_revision run.'
