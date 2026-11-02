#!/usr/bin/env bash
set -Eeuo pipefail

# Four-system, doctor-role open-case evaluation on one physical GPU.  Models
# are loaded sequentially.  Case selection and blind mappings are frozen
# before model outputs or expert ratings are inspected.

umask 077

PHASE="${1:-all}"
case "$PHASE" in
  prepare|preflight|run|verify|review|score|all) ;;
  *) printf 'usage: %s {prepare|preflight|run|verify|review|score|all}\n' "$0" >&2; exit 2 ;;
esac

CARE_ROOT="${CARE_ROOT:-/opt/care-workspace/four_agent_expert_blind_bundle/care_opd}"
QZBC_ROOT="${QZBC_ROOT:-${CARE_ROOT}/competition_evidence/tcm_best4sdt_agent30/qzbc_full_product}"
MODEL_ROOT="${MODEL_ROOT:-/opt/care-data/four_agent_expert_blind/models}"
CARE_MODEL="${CARE_MODEL:-/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model}"
EVAL_PY="${EVAL_PY:-$(command -v python || true)}"
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
CARE_VLLM_BIN="${CARE_VLLM_BIN:-$VLLM_BIN}"
BASELINE_VLLM_BIN="${BASELINE_VLLM_BIN:-$VLLM_BIN}"
CARE_SERVER_BACKEND="${CARE_SERVER_BACKEND:-transformers}"
CARE_HF_PY="${CARE_HF_PY:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
BACKEND_PY="${BACKEND_PY:-${QZBC_ROOT}/.venv/bin/python}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-3}"
ACTIVE_GPU_PROCESS_CHECK_ACK="${ACTIVE_GPU_PROCESS_CHECK_ACK:-}"
SYSTEM_API_PROFILE="${SYSTEM_API_PROFILE:-local_product}"
case "$SYSTEM_API_PROFILE" in
  local_product|live_product) ;;
  *) printf 'unsupported SYSTEM_API_PROFILE: %s\n' "$SYSTEM_API_PROFILE" >&2; exit 2 ;;
esac
FULL_PRODUCT_ROOT="${FULL_PRODUCT_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
LIGHTRAG_API_BASE_URL="${LIGHTRAG_API_BASE_URL:-http://127.0.0.1:9621}"
LIGHTRAG_INPUT_DIR="${LIGHTRAG_INPUT_DIR:-${FULL_PRODUCT_ROOT}/runtime/lightrag/active/input}"
LIGHTRAG_WORKING_DIR="${LIGHTRAG_WORKING_DIR:-${FULL_PRODUCT_ROOT}/runtime/lightrag/active/current}"
LIGHTRAG_DOCS_PATH="${LIGHTRAG_DOCS_PATH:-${LIGHTRAG_INPUT_DIR}/guardtcm_lightrag_docs.jsonl}"
QA_KNOWLEDGE_INDEX_PATH="${QA_KNOWLEDGE_INDEX_PATH:-${FULL_PRODUCT_ROOT}/runtime/qa_knowledge/qa_index.db}"

TASK_ROOT="${TCM_BEST_TASK_ROOT:-${CARE_ROOT}/runs/public_eval/tcm_best4sdt_agent30_holdout_reference_free_fixed_seed20260827}"
TASKS="${TASK_ROOT}/tcm_best4sdt_agent30.tasks.private.jsonl"
SELECTION_MANIFEST="${TASK_ROOT}/tcm_best4sdt_agent30.selection_manifest.json"
SELECTION_OVERLAP_AUDIT="${TASK_ROOT}/tcm_best4sdt_agent30.overlap_audit.json"
if [[ "$SYSTEM_API_PROFILE" == "local_product" ]]; then
  DEFAULT_RUN_ROOT="${TASK_ROOT}/local_product_four_system_runs"
  DEFAULT_REVIEW_ROOT="${TASK_ROOT}/local_product_expert_blind_review"
else
  DEFAULT_RUN_ROOT="${TASK_ROOT}/full_product_four_system_runs"
  DEFAULT_REVIEW_ROOT="${TASK_ROOT}/full_product_expert_blind_review"
fi
RUN_ROOT="${TCM_BEST_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
REVIEW_ROOT="${TCM_BEST_REVIEW_ROOT:-$DEFAULT_REVIEW_ROOT}"
REVIEW_PRIVATE="${REVIEW_ROOT}/private"
REVIEW_PUBLIC="${REVIEW_ROOT}/public_safe"
SUMMARY_PRIVATE="${REVIEW_ROOT}/summary_private"
SUMMARY_PUBLIC="${REVIEW_ROOT}/summary_public"
REVIEWERS="${REVIEWERS:-expert_01,expert_02}"

RUNTIME_OVERLAP_AUDIT="${CARE_ROOT}/competition_evidence/tcm_best4sdt_agent30/audit_runtime_retrieval_overlap.py"
RUNTIME_OVERLAP_REPORT="${TASK_ROOT}/tcm_best4sdt_agent30.${SYSTEM_API_PROFILE}.runtime_retrieval_overlap_audit.private.json"
CASE_QUALITY_VALIDATOR="${CARE_ROOT}/competition_evidence/tcm_best4sdt_agent30/validate_case_reasoning_quality.py"
CASE_QUALITY_REPORT="${RUN_ROOT}/shennong_sanyan_full/case_reasoning_quality.private.json"
PREFERENCE_PREPARE="${CARE_ROOT}/competition_evidence/prepare_agent_preference_review.py"
PREFERENCE_SUMMARIZE="${CARE_ROOT}/competition_evidence/summarize_agent_preference_review.py"
REVIEW_RENDERER="${CARE_ROOT}/competition_evidence/tcm_best4sdt_agent30/render_review_reading_packet.py"
FOUR_RUNNER="${CARE_ROOT}/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh"
MODEL_SPEC="${CARE_ROOT}/competition_evidence/local_four_agent_server/model_suite.json"
SELECTION_LOCK="${MODEL_ROOT}/suite_selection_lock.private.json"
DOWNLOAD_MANIFEST="${MODEL_ROOT}/suite_download_manifest.private.json"
REVIEW_SECRET_FILE="${REVIEW_ROOT}/expert_blind_secret.private"

require_file() {
  [[ -s "$1" ]] || { printf 'missing required file: %s\n' "$1" >&2; exit 2; }
}

prepare_sample() {
  require_file "$EVAL_PY"
  require_file "$TASKS"
  require_file "$SELECTION_MANIFEST"
  require_file "$SELECTION_OVERLAP_AUDIT"
  "$EVAL_PY" - "$TASKS" "$SELECTION_MANIFEST" "$SELECTION_OVERLAP_AUDIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

TASKS_SHA256 = "8307262d4bca7cb2dc11188b23ad16a216e2360544b3b9d93b4ce4952e6211b9"
MANIFEST_SHA256 = "2e394611870089038f51ca9b4d9cc3006ce74397fee90e39e81766e21bc0638c"
OVERLAP_SHA256 = "f20a1dfdae0bf42f233b95d29baff2b75881eb2dc72510fc7a9ecf96f7b57b11"
FORBIDDEN_TASK_KEYS = {
    "answer",
    "diagnosis",
    "gold",
    "label",
    "output",
    "reference",
    "reference_answer",
    "reference_output",
    "中医疾病诊断",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


tasks_path, manifest_path, overlap_path = map(Path, sys.argv[1:])
actual = {
    "tasks": sha256(tasks_path),
    "manifest": sha256(manifest_path),
    "overlap": sha256(overlap_path),
}
expected = {
    "tasks": TASKS_SHA256,
    "manifest": MANIFEST_SHA256,
    "overlap": OVERLAP_SHA256,
}
if actual != expected:
    raise SystemExit(f"frozen reference-free sample checksum mismatch: {actual}")
rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
if len(rows) != 30 or len({row.get("task_id") for row in rows}) != 30:
    raise SystemExit("frozen task coverage is not exactly 30 unique tasks")
if manifest.get("status") != "FROZEN_BEFORE_ANY_MODEL_GENERATION":
    raise SystemExit("selection manifest is not pre-generation frozen")
protocol = manifest.get("task_protocol") or {}
if protocol.get("reference_answers_excluded_from_task_file") is not True:
    raise SystemExit("selection manifest does not attest reference exclusion")
if (manifest.get("artifacts") or {}).get("tasks_sha256") != TASKS_SHA256:
    raise SystemExit("selection manifest does not bind the frozen task bytes")
if overlap.get("status") != "COMPLETED_BEFORE_MODEL_GENERATION":
    raise SystemExit("selection overlap audit is not pre-generation complete")
for row in rows:
    bad = {key for key in walk_keys(row) if key.strip().lower() in FORBIDDEN_TASK_KEYS}
    if bad:
        raise SystemExit(f"forbidden answer/label field in frozen task {row.get('task_id')}: {sorted(bad)}")
    case_text = str(row.get("source_text_private") or "")
    user_message = str(row.get("user_message_private") or "")
    if hashlib.sha256(case_text.encode("utf-8")).hexdigest() != row.get("paired_case_material_sha256"):
        raise SystemExit(f"case material hash mismatch: {row.get('task_id')}")
    if hashlib.sha256(user_message.encode("utf-8")).hexdigest() != row.get("shared_payload_sha256"):
        raise SystemExit(f"shared payload hash mismatch: {row.get('task_id')}")
    if not user_message.endswith(case_text):
        raise SystemExit(f"case text is not bound to the user payload: {row.get('task_id')}")
reference_path = tasks_path.with_name("tcm_best4sdt_agent30.reference.private.jsonl")
if reference_path.exists():
    raise SystemExit(f"reference artifact must not be colocated with inference tasks: {reference_path}")
print("FROZEN_REFERENCE_FREE_TASKS_OK=30")
print("TASKS_SHA256=" + actual["tasks"])
PY
}

audit_runtime_retrieval_snapshots() {
  require_file "$RUNTIME_OVERLAP_AUDIT"
  require_file "$TASKS"
  local overlap_args=()
  local source
  if [[ "$SYSTEM_API_PROFILE" == "local_product" ]]; then
    local content_data_dir="${QZBC_ROOT}/backend/app/content/data"
    local local_sources=(
      "${content_data_dir}/herbs.json"
      "${content_data_dir}/formulas.json"
      "${content_data_dir}/contraindications.json"
      "${content_data_dir}/diet_precautions.json"
      "${content_data_dir}/public_education.json"
    )
    if [[ -s "$QA_KNOWLEDGE_INDEX_PATH" ]]; then
      local_sources+=("$QA_KNOWLEDGE_INDEX_PATH")
    else
      printf 'QA_KNOWLEDGE_INDEX=DISABLED_NOT_PRESENT path=%s\n' \
        "$QA_KNOWLEDGE_INDEX_PATH"
    fi
    for source in "${local_sources[@]}"; do
      require_file "$source"
      overlap_args+=(--overlap-source "$source")
    done
  else
    require_file "$LIGHTRAG_DOCS_PATH"
    require_file "$QA_KNOWLEDGE_INDEX_PATH"
    overlap_args+=(--overlap-source "$LIGHTRAG_DOCS_PATH")
    overlap_args+=(--overlap-source "$QA_KNOWLEDGE_INDEX_PATH")
  fi
  "$EVAL_PY" "$RUNTIME_OVERLAP_AUDIT" \
    --tasks "$TASKS" \
    --retrieval-profile "$SYSTEM_API_PROFILE" \
    "${overlap_args[@]}" \
    --output "$RUNTIME_OVERLAP_REPORT" \
    --expected-tasks 30
}

check_model_downloads() {
  require_file "$DOWNLOAD_MANIFEST"
  require_file "$SELECTION_LOCK"
  "$EVAL_PY" - "$DOWNLOAD_MANIFEST" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("status") == "COMPLETE" and x.get("model_count") == 3, x
print("MODEL_DOWNLOAD_COMPLETE=3")
PY
  for path in \
    "$MODEL_ROOT/huatuogpt_3_8b/config.json" \
    "$MODEL_ROOT/glm_4_9b_0414/config.json" \
    "$MODEL_ROOT/internlm3_8b_instruct/config.json"; do
    require_file "$path"
  done
}

load_or_create_review_secret() {
  install -d -m 700 "$REVIEW_ROOT"
  if [[ ! -s "$REVIEW_SECRET_FILE" ]]; then
    command -v openssl >/dev/null 2>&1 || {
      printf 'openssl is required to create the private blind secret\n' >&2
      exit 2
    }
    openssl rand -hex 32 > "$REVIEW_SECRET_FILE"
    chmod 600 "$REVIEW_SECRET_FILE"
  fi
  export EXPERT_BLIND_SECRET
  EXPERT_BLIND_SECRET="$(tr -d '\r\n' < "$REVIEW_SECRET_FILE")"
  [[ "${#EXPERT_BLIND_SECRET}" -ge 32 ]]
}

verify_case_reasoning_quality() {
  require_file "$EVAL_PY"
  require_file "$CASE_QUALITY_VALIDATOR"
  require_file "$TASKS"
  require_file "$SELECTION_MANIFEST"
  require_file "$RUN_ROOT/shennong_sanyan_full/responses.private.jsonl"
  "$EVAL_PY" "$CASE_QUALITY_VALIDATOR" \
    --tasks "$TASKS" \
    --responses "$RUN_ROOT/shennong_sanyan_full/responses.private.jsonl" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --output "$CASE_QUALITY_REPORT"
}

verify_review_prerequisites() {
  local execution_manifest="${RUN_ROOT}/four_agent_execution.private.json"
  local project_responses="${RUN_ROOT}/shennong_sanyan_full/responses.private.jsonl"
  require_file "$execution_manifest"
  require_file "$CASE_QUALITY_REPORT"
  require_file "$project_responses"
  require_file "$SELECTION_MANIFEST"
  "$EVAL_PY" - \
    "$TASKS" "$project_responses" "$SELECTION_MANIFEST" \
    "$execution_manifest" "$CASE_QUALITY_REPORT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


tasks, responses, selection_manifest, execution_path, quality_path = map(Path, sys.argv[1:])
execution = load(execution_path)
quality = load(quality_path)
if execution.get("schema_version") != "care_opd.competition.local_four_agent_execution.v2":
    raise SystemExit("four-system execution manifest uses an obsolete schema")
if execution.get("status") != "COMPLETE" or execution.get("ready_for_blind_review_packaging") is not True:
    raise SystemExit("four-system execution manifest is not ready for blind review")
if execution.get("tasks_sha256") != sha256(tasks):
    raise SystemExit("four-system execution manifest does not match frozen tasks")
if quality.get("schema_version") != "care_opd.competition.tcm_best4sdt_case_reasoning_quality.v4":
    raise SystemExit("project case-reasoning quality report uses an obsolete schema")
if quality.get("status") != "PASS" or quality.get("reference_answer_input") is not False:
    raise SystemExit("project selection-neutral execution audit did not pass")
counts = quality.get("counts") or {}
for name in (
    "success",
    "user_visible_response_present",
    "generated_response_present",
    "case_binding_verified",
    "raw_case_packet_verified",
    "selection_neutral_output_contract_verified",
    "row_audit_present",
):
    if int(counts.get(name) or 0) != 30:
        raise SystemExit(f"project execution/audit prerequisite is incomplete: {name}")
if quality.get("output_selection_policy") != "first_user_visible_output_frozen_content_neutral":
    raise SystemExit("project report does not freeze the first user-visible output")
if quality.get("content_review_policy") != "audit_only_expert_judged_no_content_triggered_regeneration":
    raise SystemExit("project report does not declare selection-neutral content review")
if quality.get("content_review_metrics_are_selection_neutral") is not True:
    raise SystemExit("project content metrics are not declared selection-neutral")
if quality.get("safety_review_metrics_are_selection_neutral") is not True:
    raise SystemExit("project safety metrics are not declared selection-neutral")
if quality.get("length_guidance_policy") != "advisory_only_not_an_eligibility_or_scoring_threshold":
    raise SystemExit("project case-reasoning report does not declare advisory-only length guidance")
if quality.get("tasks_sha256") != sha256(tasks):
    raise SystemExit("case-reasoning quality report does not match frozen tasks")
if quality.get("responses_sha256") != sha256(responses):
    raise SystemExit("case-reasoning quality report does not match project responses")
if quality.get("selection_manifest_verified") is not True:
    raise SystemExit("case-reasoning quality report did not verify the frozen selection manifest")
if quality.get("selection_manifest_sha256") != sha256(selection_manifest):
    raise SystemExit("case-reasoning quality report does not match the frozen selection manifest")
print("BLIND_REVIEW_PREREQUISITES_OK")
PY
}

prepare_review() {
  require_file "$TASKS"
  for system in \
    shennong_sanyan_full \
    huatuogpt3_local_medical_agent \
    glm4_local_role_model \
    internlm3_local_role_model; do
    require_file "$RUN_ROOT/$system/responses.private.jsonl"
  done
  verify_review_prerequisites
  if [[ -s "$REVIEW_PRIVATE/review_manifest.private.json" ]]; then
    printf 'SKIP_EXISTING_BLIND_REVIEW=%s\n' "$REVIEW_PRIVATE"
    return 0
  fi
  load_or_create_review_secret
  "$EVAL_PY" "$PREFERENCE_PREPARE" \
    --tasks "$TASKS" \
    --response "shennong_sanyan_full=$RUN_ROOT/shennong_sanyan_full/responses.private.jsonl" \
    --response "huatuogpt3_local_medical_agent=$RUN_ROOT/huatuogpt3_local_medical_agent/responses.private.jsonl" \
    --response "glm4_local_role_model=$RUN_ROOT/glm4_local_role_model/responses.private.jsonl" \
    --response "internlm3_local_role_model=$RUN_ROOT/internlm3_local_role_model/responses.private.jsonl" \
    --reviewer "$REVIEWERS" \
    --identity-term HuatuoGPT \
    --identity-term 华佗 \
    --identity-term InternLM \
    --identity-term 书生 \
    --identity-term GLM-4 \
    --identity-term 智谱 \
    --xlsx \
    --output-private "$REVIEW_PRIVATE" \
    --output-public "$REVIEW_PUBLIC"
  local index
  for index in $(seq 1 "$(tr ',' '\n' <<< "$REVIEWERS" | sed '/^[[:space:]]*$/d' | wc -l)"); do
    "$EVAL_PY" "$REVIEW_RENDERER" \
      --input "$REVIEW_PRIVATE/reviewer_${index}.private.csv" \
      --output "$REVIEW_PRIVATE/reviewer_${index}.reading.private.html"
  done
}

summarize_review() {
  require_file "$REVIEW_PRIVATE/review_manifest.private.json"
  require_file "$REVIEW_PRIVATE/blind_mapping.private.jsonl"
  local completed="${COMPLETED_RATINGS:-}"
  if [[ -z "$completed" && -n "${COMPLETED_RATING_1:-}" ]]; then
    completed="$COMPLETED_RATING_1"
    [[ -z "${COMPLETED_RATING_2:-}" ]] || completed+=",$COMPLETED_RATING_2"
  fi
  [[ -n "$completed" ]] || {
    printf 'set COMPLETED_RATINGS to comma-separated completed CSV/XLSX paths\n' >&2; exit 2;
  }
  local rating_args=()
  local rating_paths=()
  local rating_path
  IFS=',' read -r -a rating_paths <<< "$completed"
  for rating_path in "${rating_paths[@]}"; do
    rating_path="${rating_path#${rating_path%%[![:space:]]*}}"
    rating_path="${rating_path%${rating_path##*[![:space:]]}}"
    require_file "$rating_path"
    rating_args+=(--ratings "$rating_path")
  done
  "$EVAL_PY" "$PREFERENCE_SUMMARIZE" \
    --manifest "$REVIEW_PRIVATE/review_manifest.private.json" \
    --mapping "$REVIEW_PRIVATE/blind_mapping.private.jsonl" \
    "${rating_args[@]}" \
    --primary-agent shennong_sanyan_full \
    --bootstrap-replicates 10000 \
    --bootstrap-seed 20260827 \
    --output-private "$SUMMARY_PRIVATE" \
    --output-public "$SUMMARY_PUBLIC"
}

export CARE_ROOT QZBC_ROOT MODEL_ROOT CARE_MODEL EVAL_PY VLLM_BIN
export CARE_VLLM_BIN BASELINE_VLLM_BIN BACKEND_PY CARE_SERVER_BACKEND CARE_HF_PY
export PHYSICAL_GPUS ACTIVE_GPU_PROCESS_CHECK_ACK TASKS RUN_ROOT
export SYSTEM_API_PROFILE FULL_PRODUCT_ROOT OLLAMA_BASE_URL LIGHTRAG_API_BASE_URL
export LIGHTRAG_INPUT_DIR LIGHTRAG_WORKING_DIR LIGHTRAG_DOCS_PATH
export QA_KNOWLEDGE_INDEX_PATH LIGHTRAG_API_KEY
export EXPECTED_TASKS=30
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

case "$PHASE" in
  prepare)
    prepare_sample
    ;;
  preflight)
    prepare_sample
    audit_runtime_retrieval_snapshots
    check_model_downloads
    bash "$FOUR_RUNNER" preflight
    ;;
  run)
    prepare_sample
    audit_runtime_retrieval_snapshots
    check_model_downloads
    bash "$FOUR_RUNNER" run
    verify_case_reasoning_quality
    ;;
  verify)
    prepare_sample
    require_file "$TASKS"
    require_file "$RUNTIME_OVERLAP_REPORT"
    bash "$FOUR_RUNNER" verify
    verify_case_reasoning_quality
    ;;
  review)
    prepare_sample
    prepare_review
    ;;
  score)
    summarize_review
    ;;
  all)
    prepare_sample
    audit_runtime_retrieval_snapshots
    check_model_downloads
    bash "$FOUR_RUNNER" run
    verify_case_reasoning_quality
    prepare_review
    ;;
esac

printf 'TCM_BEST4SDT_AGENT30_PHASE=%s COMPLETE\n' "$PHASE"
