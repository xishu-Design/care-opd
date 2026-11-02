#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

TEST_REQUESTS="${QZBC_DATA_ROOT}/compiled/test/qzbc_on_policy_requests.jsonl"
EVAL_ROOT="${QZBC_RUN_ROOT}/baseline_hf_evaluation"
MAX_EXAMPLES="${BASELINE_EVAL_MAX_EXAMPLES:--1}"
require_file "${TEST_REQUESTS}"
mkdir -p -- "${EVAL_ROOT}"

labels=(
  frozen
  clean_sft
  recovery_sft
  recovery_dpo
  gkd_full
  tropd_full
  care_opd
)
models=(
  "${STUDENT_MODEL}"
  "${QZBC_RUN_ROOT}/warm_start/merged"
  "${QZBC_RUN_ROOT}/baselines/recovery_sft/merged"
  "${QZBC_RUN_ROOT}/baselines/recovery_dpo/merged"
  "${QZBC_RUN_ROOT}/baselines/gkd_full/final_model"
  "${QZBC_RUN_ROOT}/baselines/tropd_full/final_model"
  "$(qzbc_final_model_path)"
)

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  model="${models[$index]}"
  if [[ "${model}" == /* && ! -f "${model}/config.json" ]]; then
    printf 'Skip missing baseline %s: %s\n' "${label}" "${model}"
    continue
  fi
  output="${EVAL_ROOT}/${label}"
  if [[ ! -f "${output}/summary.json" ]]; then
    run_qzbc_rollout \
      --requests "${TEST_REQUESTS}" \
      --output-dir "${output}" \
      --model "${model}" \
      --dtype bfloat16 \
      --max-new-tokens "${ROLLOUT_MAX_NEW_TOKENS}" \
      --greedy \
      --seed "${TRAIN_SEED}" \
      --max-examples "${MAX_EXAMPLES}"
  fi
done

export BASELINE_EVAL_ROOT="${EVAL_ROOT}"
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["BASELINE_EVAL_ROOT"])
rows = []
for summary_path in sorted(root.glob("*/summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    total = int(summary.get("rollouts", 0))
    passed = int(summary.get("passed", 0))
    rows.append({
        "baseline": summary_path.parent.name,
        "episodes": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "opd_failures": int(summary.get("opd_failures", 0)),
        "clean_controls": int(summary.get("clean_controls", 0)),
    })
(root / "baseline_metrics.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
with (root / "baseline_metrics.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["baseline"])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

printf 'HF baseline metrics: %s\n' "${EVAL_ROOT}/baseline_metrics.json"
