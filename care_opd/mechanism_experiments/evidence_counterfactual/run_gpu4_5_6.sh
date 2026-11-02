#!/usr/bin/env bash
set -Eeuo pipefail

# One model per physical GPU.  Each model receives the same fixed request bank;
# generations are deterministic greedy and are rescored after all jobs finish.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
QZBC_ROOT="${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}"
SOURCE_REQUESTS="${SOURCE_REQUESTS:-${PROJECT_ROOT}/data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/runs/mechanism_experiments/evidence_counterfactual_v1}"
BASELINE_ROOT="${BASELINE_ROOT:-${PROJECT_ROOT}/runs/qzbc_baselines_3gpu}"
CARE_ROOT="${CARE_ROOT:-${PROJECT_ROOT}/runs/qzbc_base_model_3gpu}"
CLEAN_SFT_MODEL="${CLEAN_SFT_MODEL:-${BASELINE_ROOT}/warm_start/merged}"
GKD_MODEL="${GKD_MODEL:-${BASELINE_ROOT}/baselines/gkd_full/final_model}"
CARE_MODEL="${CARE_MODEL:-${CARE_ROOT}/final_model}"
SOURCE_SUITE="${SOURCE_SUITE:-${PROJECT_ROOT}/runs/public_eval/qzbc_grouped_identity_4gpu_v1/model_suite.json}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-4,5,6}"
GROUPS_PER_FAMILY="${GROUPS_PER_FAMILY:-50}"
SELECTION_SEED="${SELECTION_SEED:-20260813}"
GENERATION_SEED="${GENERATION_SEED:-20260814}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260813}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-10000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"

IFS=',' read -r -a GPU_ARRAY <<<"${PHYSICAL_GPUS}"
if [[ ${#GPU_ARRAY[@]} -ne 3 ]]; then
  printf 'PHYSICAL_GPUS must contain exactly three comma-separated GPUs\n' >&2
  exit 2
fi
for required in "${PYTHON_BIN}" "${QZBC_ROOT}" "${SOURCE_REQUESTS}" \
  "${CLEAN_SFT_MODEL}" "${GKD_MODEL}" "${CARE_MODEL}" "${SOURCE_SUITE}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing evidence-counterfactual input: %s\n' "${required}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" - "${SOURCE_SUITE}" "${CLEAN_SFT_MODEL}" "${GKD_MODEL}" "${CARE_MODEL}" <<'PY'
import json
import sys
from pathlib import Path

suite = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {"clean_sft": sys.argv[2], "gkd_full": sys.argv[3], "care_opd": sys.argv[4]}
entries = suite.get("models")
if not isinstance(entries, list) or len(str(suite.get("suite_fingerprint") or "")) != 64:
    raise SystemExit("invalid frozen model_suite.json")
for label, model_path in expected.items():
    matches = [row for row in entries if row.get("label") == label]
    if len(matches) != 1:
        raise SystemExit(f"suite does not contain {label!r} exactly once")
    recorded = str(Path(str(matches[0].get("canonical_path") or "")).resolve())
    requested = str(Path(model_path).resolve())
    if recorded != requested:
        raise SystemExit(f"checkpoint path mismatch for {label}: {requested} != {recorded}")
    if len(str(matches[0].get("model_fingerprint") or "")) != 64:
        raise SystemExit(f"invalid model fingerprint for {label}")
print(f"model_suite_fingerprint={suite['suite_fingerprint']}")
PY

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p -- "${OUT_ROOT}/logs"

BANK="${OUT_ROOT}/bank/counterfactual_requests.jsonl"
if [[ ! -s "${BANK}" ]]; then
  "${PYTHON_BIN}" -u -m \
    mechanism_experiments.evidence_counterfactual.build_counterfactual_bank \
    --source-requests "${SOURCE_REQUESTS}" \
    --qzbc-root "${QZBC_ROOT}" \
    --output-dir "${OUT_ROOT}/bank" \
    --groups-per-family "${GROUPS_PER_FAMILY}" \
    --selection-seed "${SELECTION_SEED}"
  "${PYTHON_BIN}" - "${OUT_ROOT}/bank/bank_manifest.json" "${SOURCE_SUITE}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
suite_path = Path(sys.argv[2]).resolve()
value = json.loads(path.read_text(encoding="utf-8"))
value["formal_model_suite"] = {
    "path": str(suite_path),
    "sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
    "suite_fingerprint": json.loads(suite_path.read_text(encoding="utf-8"))["suite_fingerprint"],
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
fi

labels=(clean_sft gkd_full care_opd)
models=("${CLEAN_SFT_MODEL}" "${GKD_MODEL}" "${CARE_MODEL}")
pids=()
for index in 0 1 2; do
  label="${labels[$index]}"
  model="${models[$index]}"
  gpu="${GPU_ARRAY[$index]}"
  model_out="${OUT_ROOT}/rollouts/${label}"
  mkdir -p -- "${model_out}"
  if [[ -s "${model_out}/summary.json" && -s "${model_out}/qzbc_rollouts.jsonl" ]]; then
    printf '[skip] complete rollout: %s\n' "${label}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PYTHON_BIN}" -u -m care_opd.qzbc_rollout \
    --requests "${BANK}" \
    --output-dir "${model_out}" \
    --model "${model}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --greedy \
    --seed "${GENERATION_SEED}" \
    >"${OUT_ROOT}/logs/${label}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  printf '[started] model=%s physical_gpu=%s pid=%s\n' "${label}" "${gpu}" "$!"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ ${status} -ne 0 ]]; then
  printf 'At least one rollout failed; inspect %s/logs\n' "${OUT_ROOT}" >&2
  exit ${status}
fi

"${PYTHON_BIN}" -u -m \
  mechanism_experiments.evidence_counterfactual.score_counterfactuals \
  --requests "${BANK}" \
  --rollouts "clean_sft=${OUT_ROOT}/rollouts/clean_sft/qzbc_rollouts.jsonl" \
  --rollouts "gkd_full=${OUT_ROOT}/rollouts/gkd_full/qzbc_rollouts.jsonl" \
  --rollouts "care_opd=${OUT_ROOT}/rollouts/care_opd/qzbc_rollouts.jsonl" \
  --output-dir "${OUT_ROOT}/results" \
  --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

printf 'Evidence counterfactual assay complete: %s\n' "${OUT_ROOT}/results/counterfactual_summary.md"
