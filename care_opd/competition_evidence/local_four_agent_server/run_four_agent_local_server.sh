#!/usr/bin/env bash
set -Eeuo pipefail

# Sequentially run the complete Shennong Sanyan system and three local
# open-weight role-Agent baselines.  Keep this script and all outputs PRIVATE.

umask 077

PHASE="${1:-all}"
case "$PHASE" in
  preflight|download|run|verify|all) ;;
  *) printf 'usage: %s {preflight|download|run|verify|all}\n' "$0" >&2; exit 2 ;;
esac

CARE_ROOT="${CARE_ROOT:-/opt/care-workspace/care_opd}"
QZBC_ROOT="${QZBC_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main-private-eval}"
EVAL_PY="${EVAL_PY:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
BACKEND_PY="${BACKEND_PY:-${QZBC_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/vllm}"
# CARE-OPD is a Qwen3.5 checkpoint and may require a newer vLLM/Transformers
# stack than the three frozen comparison models.  Keep separate launchers so
# upgrading CARE inference cannot silently change the baseline environment.
CARE_VLLM_BIN="${CARE_VLLM_BIN:-$VLLM_BIN}"
BASELINE_VLLM_BIN="${BASELINE_VLLM_BIN:-$VLLM_BIN}"
CARE_SERVER_BACKEND="${CARE_SERVER_BACKEND:-vllm}"
CARE_HF_PY="${CARE_HF_PY:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-4}"
# The default remains fail-closed when another compute process is present.
# A time-critical operator may explicitly acknowledge the contention/OOM risk
# and bypass only that occupancy check; GPU existence and all study-contract
# checks still run.
ACTIVE_GPU_PROCESS_CHECK_ACK="${ACTIVE_GPU_PROCESS_CHECK_ACK:-}"
MODEL_ROOT="${MODEL_ROOT:-/opt/care-workspace/models/four_agent_expert_blind_local}"
CARE_MODEL="${CARE_MODEL:-/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model}"
TASKS="${TASKS:-${CARE_ROOT}/runs/private_real_thyroid_external/three_role_eval/paired_fact_n12/three_role_tasks.private.jsonl}"
RUN_ROOT="${RUN_ROOT:-${CARE_ROOT}/runs/private_real_thyroid_external/three_role_eval/four_agent_expert_blind_local}"
SCRIPT_ROOT="${CARE_ROOT}/competition_evidence"
LOCAL_ROOT="${SCRIPT_ROOT}/local_four_agent_server"
CARE_HF_SERVER="${LOCAL_ROOT}/serve_care_transformers.py"
MODEL_SPEC="${LOCAL_ROOT}/model_suite.json"
VLLM_PORT="${VLLM_PORT:-18100}"
BACKEND_PORT="${BACKEND_PORT:-18000}"
# Keep the comparison models on their frozen 8k protocol.  The complete CARE
# product receives a larger window because its prompt contains role context,
# retrieved evidence, Graphiti/Mem0 state and the requested 1.2k-token answer.
# MAX_MODEL_LEN remains a backwards-compatible alias for baseline launches.
BASELINE_MAX_MODEL_LEN="${BASELINE_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-8192}}"
CARE_MAX_MODEL_LEN="${CARE_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
# vLLM 0.11 enables the FlashInfer V1 sampler by default.  On the frozen
# server image its JIT sees incompatible nvcc/toolkit headers and aborts after
# the model weights have loaded.  Use vLLM's supported native sampler fallback;
# this leaves attention, model weights and generation parameters unchanged.
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
EXPECTED_TASKS="${EXPECTED_TASKS:-72}"
SYSTEM_API_PROFILE="${SYSTEM_API_PROFILE:-locked_evidence}"
FULL_PRODUCT_ROOT="${FULL_PRODUCT_ROOT:-/opt/care-workspace/qzbc_source/qzbc-main}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
LIGHTRAG_API_BASE_URL="${LIGHTRAG_API_BASE_URL:-http://127.0.0.1:9621}"
LIGHTRAG_INPUT_DIR="${LIGHTRAG_INPUT_DIR:-${FULL_PRODUCT_ROOT}/runtime/lightrag/active/input}"
LIGHTRAG_WORKING_DIR="${LIGHTRAG_WORKING_DIR:-${FULL_PRODUCT_ROOT}/runtime/lightrag/active/current}"
LIGHTRAG_DOCS_PATH="${LIGHTRAG_DOCS_PATH:-${LIGHTRAG_INPUT_DIR}/guardtcm_lightrag_docs.jsonl}"
QA_KNOWLEDGE_INDEX_PATH="${QA_KNOWLEDGE_INDEX_PATH:-${FULL_PRODUCT_ROOT}/runtime/qa_knowledge/qa_index.db}"
LOCAL_CONTENT_DATA_DIR="${LOCAL_CONTENT_DATA_DIR:-${QZBC_ROOT}/backend/app/content/data}"

IFS=',' read -r -a GPU_ARRAY <<< "$PHYSICAL_GPUS"
TP_SIZE="${#GPU_ARRAY[@]}"
if [[ "$TP_SIZE" -lt 1 ]]; then
  printf 'PHYSICAL_GPUS must contain at least one GPU index\n' >&2
  exit 2
fi

LOG_ROOT="${RUN_ROOT}/logs"
RUNTIME_ROOT="${RUN_ROOT}/runtime"
SECRETS_FILE="${RUNTIME_ROOT}/secrets.private.env"
DB_PATH="${RUNTIME_ROOT}/guardtcm_eval.private.db"
DRIVER_MANIFEST="${RUN_ROOT}/local_execution_driver.private.txt"
RUN_LOCK_FILE="${RUNTIME_ROOT}/four_agent_run.lock"
RUN_LOCK_OWNER_FILE="${RUNTIME_ROOT}/four_agent_run.lock.owner.private.txt"
VLLM_PID=""
BACKEND_PID=""

require_file() {
  [[ -s "$1" ]] || { printf 'missing required file: %s\n' "$1" >&2; exit 2; }
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "$1" >&2
    exit 2
  }
}

ensure_single_writer() {
  # Re-exec the run below a util-linux flock process.  The lock is owned by
  # flock rather than an inherited shell FD, so model/backend children cannot
  # accidentally keep it alive after the driver exits.
  if [[ "${FOUR_AGENT_RUN_LOCK_HELD:-}" == "1" ]]; then
    printf 'pid=%s\nphase=%s\nstarted_utc=%s\n' \
      "$$" "$PHASE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$RUN_LOCK_OWNER_FILE"
    chmod 600 "$RUN_LOCK_OWNER_FILE"
    return 0
  fi
  require_command flock
  install -d -m 700 "$RUN_ROOT" "$RUNTIME_ROOT"
  export FOUR_AGENT_RUN_LOCK_HELD=1
  exec flock --exclusive --nonblock --close --conflict-exit-code 73 \
    "$RUN_LOCK_FILE" bash "$0" "$PHASE"
}

stop_pid() {
  local pid="${1:-}"
  local label="${2:-process}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 1
    done
    printf '%s did not stop after SIGTERM; inspect PID %s manually\n' "$label" "$pid" >&2
    return 1
  fi
}

cleanup() {
  local status=$?
  stop_pid "$BACKEND_PID" "isolated backend" || true
  stop_pid "$VLLM_PID" "vLLM" || true
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_url() {
  local url="$1"
  local label="$2"
  local pid="$3"
  local attempts="${4:-180}"
  for _ in $(seq 1 "$attempts"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      printf '%s exited before becoming ready\n' "$label" >&2
      return 1
    fi
    if curl --fail --silent --show-error --max-time 3 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  printf 'timeout waiting for %s at %s\n' "$label" "$url" >&2
  return 1
}

output_complete() {
  local output_dir="$1"
  local expected_profile="${2:-}"
  local manifest="${output_dir}/three_role_execution.private.json"
  [[ -s "$manifest" ]] || return 1
  "$EVAL_PY" - "$manifest" "$EXPECTED_TASKS" "$expected_profile" <<'PY'
import json, sys
path, expected, expected_profile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
row = json.load(open(path, encoding="utf-8"))
ok = (
    row.get("status") == "COMPLETE"
    and row.get("success_count") == expected
    and row.get("failure_count") == 0
    and (not expected_profile or row.get("system_api_profile") == expected_profile)
)
raise SystemExit(0 if ok else 1)
PY
}

preflight_product_stack() {
  case "$SYSTEM_API_PROFILE" in
    live_product|local_product) ;;
    locked_evidence) return 0 ;;
    *) printf 'SYSTEM_API_PROFILE must be locked_evidence, live_product or local_product\n' >&2; exit 2 ;;
  esac
  (
    cd "${QZBC_ROOT}/backend"
    "$BACKEND_PY" - <<'PY'
import graphiti_core, mem0, qdrant_client, redislite
print("PRODUCT_PYTHON_STACK_OK")
PY
  )
  if [[ "$SYSTEM_API_PROFILE" == "local_product" ]]; then
    for name in sources herbs formulas contraindications diet_precautions public_education; do
      require_file "${LOCAL_CONTENT_DATA_DIR}/${name}.json"
    done
  else
    require_file "$QA_KNOWLEDGE_INDEX_PATH"
    require_file "$LIGHTRAG_DOCS_PATH"
  fi
  "$EVAL_PY" - "$SYSTEM_API_PROFILE" "$OLLAMA_BASE_URL" \
    "$LIGHTRAG_API_BASE_URL" "$LIGHTRAG_DOCS_PATH" \
    "$QA_KNOWLEDGE_INDEX_PATH" "$LOCAL_CONTENT_DATA_DIR" <<'PY'
import json, os, sqlite3, sys, urllib.request

profile = sys.argv[1]
ollama, lightrag = sys.argv[2].rstrip('/'), sys.argv[3].rstrip('/')
docs_path, qa_path, content_dir = sys.argv[4], sys.argv[5], sys.argv[6]
with urllib.request.urlopen(ollama + '/api/tags', timeout=10) as response:
    payload = json.load(response)
names = {str(row.get('name') or '') for row in payload.get('models', [])}
for required in ('qwen2.5:7b', 'bge-m3'):
    if required not in names and required + ':latest' not in names:
        raise SystemExit(f'missing Ollama model: {required}')
embed_request = urllib.request.Request(
    ollama + '/api/embed',
    data=json.dumps({'model': 'bge-m3', 'input': ['中医辨证论治']}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(embed_request, timeout=60) as response:
    embed_payload = json.load(response)
embeddings = embed_payload.get('embeddings') if isinstance(embed_payload, dict) else None
if not isinstance(embeddings, list) or not embeddings or len(embeddings[0]) != 1024:
    raise SystemExit('Ollama bge-m3 embedding probe did not return 1024 dimensions')
qa_count = 0
if os.path.isfile(qa_path) and os.path.getsize(qa_path) > 0:
    connection = sqlite3.connect(qa_path)
    try:
        qa_count = int(connection.execute('SELECT count(*) FROM qa_documents').fetchone()[0])
    finally:
        connection.close()
if profile == 'live_product' and qa_count < 1:
    raise SystemExit('live_product QA knowledge index contains no documents')

if profile == 'local_product':
    counts = {}
    for name in ('sources', 'herbs', 'formulas', 'contraindications', 'diet_precautions', 'public_education'):
        path = os.path.join(content_dir, name + '.json')
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
        items = payload.get('items') if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise SystemExit(f'local structured source has invalid items list: {path}')
        counts[name] = len(items)
    if counts['herbs'] < 1000 or counts['formulas'] < 4000:
        raise SystemExit(f'local structured corpus is unexpectedly incomplete: {counts}')
    if counts['contraindications'] < 1 or counts['diet_precautions'] < 1:
        raise SystemExit(f'local structured safety corpus is unexpectedly incomplete: {counts}')
    print(
        'LOCAL_PRODUCT_SERVICES_OK',
        f"structured_entries={sum(counts[name] for name in counts if name != 'sources')}",
        f'qa_docs={qa_count}',
        f'qa_enabled={str(qa_count > 0).lower()}',
        'embedding_dims=1024',
    )
    raise SystemExit(0)

headers = {}
if os.environ.get('LIGHTRAG_API_KEY', '').strip():
    headers['X-API-Key'] = os.environ['LIGHTRAG_API_KEY'].strip()
with urllib.request.urlopen(
    urllib.request.Request(lightrag + '/health', headers=headers), timeout=15
) as response:
    health = json.load(response)
status = str(health.get('status') or 'ok').lower()
if status not in {'ok', 'healthy', 'ready', 'success'}:
    raise SystemExit(f'LightRAG health is not ready: {status}')
with urllib.request.urlopen(
    urllib.request.Request(lightrag + '/openapi.json', headers=headers), timeout=15
) as response:
    openapi = json.load(response)
paths = set(openapi.get('paths', {}))
required = {'/health', '/query/data', '/documents/texts', '/graph/label/search', '/graphs'}
missing = sorted(required - paths)
if missing:
    raise SystemExit(f'LightRAG OpenAPI contract missing paths: {missing}')
docs = []
with open(docs_path, encoding='utf-8') as handle:
    for line in handle:
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict) and str(value.get('doc_id') or '').startswith('DOC_'):
                docs.append(value)
if not docs:
    raise SystemExit('LightRAG local document manifest is empty')
query_text = str(docs[0].get('title') or docs[0].get('text') or '中医辨证论治')[:200]
query_headers = {**headers, 'Content-Type': 'application/json'}
query_request = urllib.request.Request(
    lightrag + '/query/data',
    data=json.dumps({'query': query_text, 'mode': 'mix', 'top_k': 10}, ensure_ascii=False).encode('utf-8'),
    headers=query_headers,
    method='POST',
)
with urllib.request.urlopen(query_request, timeout=180) as response:
    query_payload = json.load(response)
data = query_payload.get('data') if isinstance(query_payload, dict) else None
data = data if isinstance(data, dict) else query_payload
chunks = data.get('chunks') if isinstance(data, dict) else None
references = data.get('references') if isinstance(data, dict) else None
if not isinstance(chunks, list) or not chunks or not isinstance(references, list):
    raise SystemExit('LightRAG query/data probe returned no structured chunks/references')
known_doc_ids = {str(item['doc_id']) for item in docs}
returned_doc_ids = {
    part.strip()
    for chunk in chunks if isinstance(chunk, dict)
    for part in str(chunk.get('file_path') or '').split('<SEP>')
    if part.strip().startswith('DOC_')
}
if not (known_doc_ids & returned_doc_ids):
    raise SystemExit('LightRAG query chunks do not map to the frozen local document manifest')
print(
    'FULL_PRODUCT_LOCAL_SERVICES_OK',
    f'lightrag_docs={len(docs)}',
    f'qa_docs={qa_count}',
    'embedding_dims=1024',
)
PY
}

preflight_product_task_routing() {
  case "$SYSTEM_API_PROFILE" in
    live_product|local_product) ;;
    *) return 0 ;;
  esac
  (
    cd "${QZBC_ROOT}/backend"
    "$BACKEND_PY" - "$TASKS" <<'PY'
import json
import sys

from app.services.memory_consolidation_service import MemoryConsolidationAgent

tasks = [
    json.loads(line)
    for line in open(sys.argv[1], encoding="utf-8")
    if line.strip()
]
agent = MemoryConsolidationAgent(None)
misrouted = []
for task in tasks:
    acknowledgement = agent.preference_acknowledgement(
        owner_role=str(task["role"]),
        user_message=str(task["user_message_private"]),
    )
    if acknowledgement:
        misrouted.append(str(task.get("task_id") or "unknown"))
if misrouted:
    raise SystemExit(
        "product task would be misrouted to memory preference policy: "
        + ",".join(misrouted)
    )
print(f"MEMORY_ROUTE_TASKS_OK={len(tasks)}")
PY
  )
}

preflight() {
  require_command curl
  require_command nvidia-smi
  require_file "$EVAL_PY"
  require_file "$BASELINE_VLLM_BIN"
  require_file "$TASKS"
  require_file "$MODEL_SPEC"
  require_file "${SCRIPT_ROOT}/execute_three_role_eval.py"
  require_file "${SCRIPT_ROOT}/three_role_eval_common.py"
  require_file "${SCRIPT_ROOT}/validate_local_four_agent_run.py"
  require_file "${QZBC_ROOT}/backend/app/main.py"
  require_file "${QZBC_ROOT}/backend/alembic.ini"
  require_file "$BACKEND_PY"
  require_file "${CARE_MODEL}/config.json"

  "$EVAL_PY" - "$TASKS" "$EXPECTED_TASKS" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
if len(rows) != expected:
    raise SystemExit(f"task count mismatch: expected={expected}, actual={len(rows)}")
if len({row.get("task_id") for row in rows}) != expected:
    raise SystemExit("task_id values are missing or duplicated")
print(f"TASKS_OK={len(rows)}")
PY

  nvidia-smi -i "$PHYSICAL_GPUS" \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader,nounits
  if [[ "$ACTIVE_GPU_PROCESS_CHECK_ACK" == "I_ACCEPT_GPU_CONTENTION_RISK" ]]; then
    printf 'ACTIVE_GPU_PROCESS_CHECK=SKIPPED_BY_OPERATOR_ACK gpu=%s\n' "$PHYSICAL_GPUS"
  else
    local active
    active="$(nvidia-smi -i "$PHYSICAL_GPUS" \
      --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
      | grep -Eiv 'ollama' || true)"
    if [[ -n "$active" ]]; then
      printf 'selected GPU set currently has active compute processes: %s\n' "$PHYSICAL_GPUS" >&2
      if [[ "$PHASE" == "run" || "$PHASE" == "all" ]]; then
        printf 'choose free cards with PHYSICAL_GPUS before inference, or explicitly set ACTIVE_GPU_PROCESS_CHECK_ACK=I_ACCEPT_GPU_CONTENTION_RISK\n' >&2
        exit 3
      fi
    fi
  fi
  df -h "$MODEL_ROOT" "$(dirname "$RUN_ROOT")" 2>/dev/null || true
  "$EVAL_PY" -c 'import huggingface_hub; print("EVAL_STACK_OK", huggingface_hub.__version__)'
  local vllm_serve_help
  printf 'BASELINE_VLLM=%s ' "$BASELINE_VLLM_BIN"
  "$BASELINE_VLLM_BIN" --version
  # vLLM 0.11 groups frontend flags behind --help=all.
  vllm_serve_help="$(
    "$BASELINE_VLLM_BIN" serve --help=all 2>&1 \
      || "$BASELINE_VLLM_BIN" serve --help 2>&1
  )"
  if ! grep -Fq -- '--disable-log-requests' <<< "$vllm_serve_help" \
    && ! grep -Fq -- '--enable-log-requests' <<< "$vllm_serve_help"; then
    printf 'baseline vLLM has no recognized request-logging privacy control\n' >&2
    exit 2
  fi

  case "$CARE_SERVER_BACKEND" in
    transformers)
      [[ "$TP_SIZE" -eq 1 ]] || {
        printf 'transformers CARE server requires exactly one physical GPU\n' >&2
        exit 2
      }
      require_file "$CARE_HF_PY"
      require_file "$CARE_HF_SERVER"
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "$CARE_HF_PY" - "$CARE_MODEL" <<'PY'
import sys
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(
    sys.argv[1], local_files_only=True, trust_remote_code=False
)
if getattr(config, "model_type", None) != "qwen3_5_text":
    raise SystemExit(f"unexpected CARE model type: {getattr(config, 'model_type', None)!r}")
if type(config) not in AutoModelForCausalLM._model_mapping:
    raise SystemExit(f"CARE config has no causal-LM mapping: {type(config)!r}")
print("CARE_TRANSFORMERS_STACK_OK", transformers.__version__)
PY
      ;;
    vllm)
      require_file "$CARE_VLLM_BIN"
      printf 'CARE_VLLM=%s ' "$CARE_VLLM_BIN"
      "$CARE_VLLM_BIN" --version
      vllm_serve_help="$(
        "$CARE_VLLM_BIN" serve --help=all 2>&1 \
          || "$CARE_VLLM_BIN" serve --help 2>&1
      )"
      if ! grep -Fq -- '--disable-log-requests' <<< "$vllm_serve_help" \
        && ! grep -Fq -- '--enable-log-requests' <<< "$vllm_serve_help"; then
        printf 'CARE vLLM has no recognized request-logging privacy control\n' >&2
        exit 2
      fi
      ;;
    *)
      printf 'CARE_SERVER_BACKEND must be transformers or vllm\n' >&2
      exit 2
      ;;
  esac
  (
    cd "${QZBC_ROOT}/backend"
    "$BACKEND_PY" -c \
      'import fastapi, sqlalchemy, uvicorn, alembic; import app.main; print("BACKEND_IMPORT_OK")'
  )
  preflight_product_task_routing
  preflight_product_stack
}

release_ollama_models() {
  case "$SYSTEM_API_PROFILE" in
    live_product|local_product) ;;
    *) return 0 ;;
  esac
  # Graphiti/Mem0 and optional QA reranking need the local Ollama models only
  # while the project arm is running.  Unload them before each vLLM baseline
  # reserves GPU memory; the service itself remains available to other jobs.
  curl --silent --show-error --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen2.5:7b","keep_alive":0}' \
    "${OLLAMA_BASE_URL%/}/api/generate" >/dev/null 2>&1 || true
  curl --silent --show-error --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"bge-m3","input":[""],"keep_alive":0}' \
    "${OLLAMA_BASE_URL%/}/api/embed" >/dev/null 2>&1 || true
  sleep 2
}

download_models() {
  install -d -m 700 "$MODEL_ROOT"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  "$EVAL_PY" "${LOCAL_ROOT}/download_model_suite.py" \
    --spec "$MODEL_SPEC" \
    --output-root "$MODEL_ROOT" \
    --max-workers "${HF_DOWNLOAD_WORKERS:-4}"
}

load_or_create_secrets() {
  install -d -m 700 "$RUNTIME_ROOT"
  if [[ ! -s "$SECRETS_FILE" ]]; then
    require_command openssl
    {
      printf 'JWT_SECRET_KEY=%s\n' "$(openssl rand -hex 32)"
      printf 'CARE_THREE_ROLE_BACKEND_EVAL_KEY=%s\n' "$(openssl rand -hex 32)"
      printf 'CARE_THREE_ROLE_PROVISION_SECRET=%s\n' "$(openssl rand -hex 32)"
    } > "$SECRETS_FILE"
    chmod 600 "$SECRETS_FILE"
  fi
  # The file contains only locally generated hexadecimal values and is PRIVATE.
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
  [[ "${#JWT_SECRET_KEY}" -ge 32 ]]
  [[ "${#CARE_THREE_ROLE_BACKEND_EVAL_KEY}" -ge 32 ]]
  [[ "${#CARE_THREE_ROLE_PROVISION_SECRET}" -ge 32 ]]
}

start_vllm() {
  local label="$1"
  local model_path="$2"
  local served_name="$3"
  local reasoning_parser="${4:-}"
  local trust_remote_code="${5:-false}"
  local vllm_bin="${6:-$BASELINE_VLLM_BIN}"
  local server_chat_template_kwargs="${7:-}"
  local max_model_len="${8:-$BASELINE_MAX_MODEL_LEN}"
  local log="${LOG_ROOT}/${label}.vllm.log"
  require_file "${model_path}/config.json"
  local args=(
    serve "$model_path"
    --host 127.0.0.1
    --port "$VLLM_PORT"
    --served-model-name "$served_name"
    --tensor-parallel-size "$TP_SIZE"
    --dtype bfloat16
    --max-model-len "$max_model_len"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs 1
    --seed 42
    --generation-config vllm
  )
  local vllm_serve_help
  vllm_serve_help="$(
    "$vllm_bin" serve --help=all 2>&1 || "$vllm_bin" serve --help 2>&1
  )"
  if grep -Fq -- '--enable-log-requests' <<< "$vllm_serve_help"; then
    # vLLM >=0.11 logs request content only when this opt-in flag is present.
    # Omitting it avoids the deprecated --disable-log-requests warning while
    # preserving the privacy-safe default.
    printf 'REQUEST_LOGGING label=%s control=enable-log-requests-omitted\n' \
      "$label" >> "$DRIVER_MANIFEST"
  elif grep -Fq -- '--disable-log-requests' <<< "$vllm_serve_help"; then
    args+=(--disable-log-requests)
  else
    printf '%s vLLM has no recognized request-logging privacy control\n' "$label" >&2
    exit 2
  fi
  if [[ -n "$server_chat_template_kwargs" ]]; then
    grep -Fq -- '--default-chat-template-kwargs' <<< "$vllm_serve_help" || {
      printf '%s vLLM does not support --default-chat-template-kwargs\n' "$label" >&2
      exit 2
    }
    args+=(--default-chat-template-kwargs "$server_chat_template_kwargs")
  fi
  if [[ "$trust_remote_code" == "true" ]]; then
    args+=(--trust-remote-code)
  fi
  if [[ -n "$reasoning_parser" ]]; then
    grep -Fq -- '--reasoning-parser' <<< "$vllm_serve_help" || {
      printf '%s vLLM does not support --reasoning-parser\n' "$label" >&2
      exit 2
    }
    # vLLM 0.11 accepts the parser option but rejects its former explicit
    # reasoning-enable switch.  The parser option alone is the compatible
    # contract, so the removed switch is never forwarded.
    args+=(--reasoning-parser "$reasoning_parser")
  fi
  printf 'START_VLLM label=%s model=%s served=%s gpus=%s binary=%s max_model_len=%s\n' \
    "$label" "$model_path" "$served_name" "$PHYSICAL_GPUS" "$vllm_bin" \
    "$max_model_len" >> "$DRIVER_MANIFEST"
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS" \
    VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
    "$vllm_bin" "${args[@]}" > "$log" 2>&1 &
  VLLM_PID=$!
  printf '%s\n' "$VLLM_PID" > "${LOG_ROOT}/${label}.vllm.pid"
  wait_url "http://127.0.0.1:${VLLM_PORT}/v1/models" "$label vLLM" "$VLLM_PID" 300
}

start_care_transformers() {
  local label="shennong_sanyan_full"
  local served_name="shennong-sanyan-care-opd"
  local log="${LOG_ROOT}/${label}.transformers.log"
  require_file "$CARE_HF_PY"
  require_file "$CARE_HF_SERVER"
  require_file "${CARE_MODEL}/config.json"
  printf 'START_TRANSFORMERS label=%s model=%s served=%s gpus=%s python=%s max_model_len=%s\n' \
    "$label" "$CARE_MODEL" "$served_name" "$PHYSICAL_GPUS" "$CARE_HF_PY" \
    "$CARE_MAX_MODEL_LEN" \
    >> "$DRIVER_MANIFEST"
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS" "$CARE_HF_PY" "$CARE_HF_SERVER" \
    --model "$CARE_MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --served-model-name "$served_name" \
    --max-model-len "$CARE_MAX_MODEL_LEN" \
    --seed 42 > "$log" 2>&1 &
  VLLM_PID=$!
  printf '%s\n' "$VLLM_PID" > "${LOG_ROOT}/${label}.transformers.pid"
  wait_url "http://127.0.0.1:${VLLM_PORT}/v1/models" \
    "$label Transformers server" "$VLLM_PID" 300
}

bootstrap_private_backend_db() {
  case "$DB_PATH" in
    "$RUNTIME_ROOT"/*.private.db) ;;
    *)
      printf 'refusing to bootstrap a database outside the isolated runtime: %s\n' "$DB_PATH" >&2
      return 2
      ;;
  esac
  (
    cd "${QZBC_ROOT}/backend"
    "$BACKEND_PY" - "$SYSTEM_API_PROFILE" "$LOCAL_CONTENT_DATA_DIR" <<'PY'
import sys
from pathlib import Path

from sqlalchemy import inspect

from app.content.services.content_import_service import (
    ContentImportService,
    content_db_counts,
)
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.init_db import init_db
from app.db.session import SessionLocal, engine

# Safe and idempotent only because the runner owns this explicit *.private.db.
# create_all preserves rows when an interrupted evaluation is resumed.
init_db()
inspector = inspect(engine)
problems = []
for table in Base.metadata.sorted_tables:
    name = table.name
    if not inspector.has_table(name):
        problems.append(f"missing table {name}")
        continue
    actual = {column["name"] for column in inspector.get_columns(name)}
    missing = {column.name for column in table.columns} - actual
    if missing:
        problems.append(f"{name}: missing columns {sorted(missing)}")
if problems:
    raise SystemExit("private DB schema mismatch: " + "; ".join(problems))
print(f"PRIVATE_DB_SCHEMA_OK tables={len(Base.metadata.tables)}")

profile, content_dir = sys.argv[1], Path(sys.argv[2]).resolve()
if profile == "local_product":
    if not content_dir.is_dir():
        raise SystemExit(f"missing frozen content directory: {content_dir}")
    with SessionLocal() as db:
        report = ContentImportService(db, content_dir).import_content()
        db.commit()
        counts = content_db_counts(db)
    if counts["knowledge_entries"] < 5000:
        raise SystemExit(
            "local_product structured corpus import is incomplete: " + str(counts)
        )
    print(
        "LOCAL_PRODUCT_CONTENT_IMPORT_OK",
        f"run_id={report['run_id']}",
        f"knowledge_entries={counts['knowledge_entries']}",
        f"knowledge_sources={counts['sources']}",
    )
PY
    # Revision 0001 materializes the current ORM metadata.  Replaying later
    # create-table migrations can therefore duplicate tables on a fresh DB.
    # Validate the current schema above, then mark this isolated DB at head.
    "$BACKEND_PY" -m alembic -c alembic.ini stamp head
    "$BACKEND_PY" -m alembic -c alembic.ini current
  )
}

start_private_backend() {
  local served_name="$1"
  local log="${LOG_ROOT}/shennong_sanyan_full.backend.log"
  export APP_ENV=competition_eval
  export DATABASE_URL="sqlite:///${DB_PATH}"
  export JWT_SECRET_KEY
  export CORS_ORIGINS='[]'

  if [[ "$SYSTEM_API_PROFILE" == "live_product" || "$SYSTEM_API_PROFILE" == "local_product" ]]; then
    export PRIVATE_EVAL_EVIDENCE_CONTROL_ENABLED=false
    unset PRIVATE_EVAL_ISOLATED_DATABASE_ACK PRIVATE_EVAL_EVIDENCE_CONTROL_SECRET || true
    export CASE_REASONING_EVIDENCE_CONTROL_ENABLED=true
    export CASE_REASONING_ISOLATED_DATABASE_ACK=I_CONFIRM_PRIVATE_ISOLATED_DATABASE
    export CASE_REASONING_EVIDENCE_CONTROL_SECRET="$CARE_THREE_ROLE_BACKEND_EVAL_KEY"
    export GRAPH_BACKEND=graphiti
    export MEMORY_BACKEND=mem0
    export GRAPHITI_DB_PATH="${RUNTIME_ROOT}/graphiti/guardtcm_eval.db"
    export GRAPHITI_DATABASE=guardtcm_eval
    export GRAPHITI_GROUP_PREFIX=guardtcm-eval
    export GRAPHITI_TIMEOUT_SECONDS=90
    if [[ "$SYSTEM_API_PROFILE" == "live_product" ]]; then
      export GRAPHITI_INGEST_CHAT_HISTORY=true
    else
      # local_product proves Graphiti through a synchronous, typed seed/write/
      # read contract.  Do not enqueue an additional LLM-heavy background
      # extraction for the neutral warm-up turn: it shares the single embedded
      # Graphiti runtime and can block the evaluated request for minutes.
      export GRAPHITI_INGEST_CHAT_HISTORY=false
    fi
    export MEM0_DIR="${RUNTIME_ROOT}/mem0/meta"
    export MEM0_QDRANT_PATH="${RUNTIME_ROOT}/mem0/qdrant"
    export MEM0_HISTORY_DB_PATH="${RUNTIME_ROOT}/mem0/history.db"
    export MEM0_COLLECTION_NAME=guardtcm_eval_memories
    export MEM0_EMBEDDING_DIMS=1024
    export MEM0_INGEST_CHAT_HISTORY=true
    export MEM0_SEARCH_TOP_K=8
    export MEM0_MIN_RELEVANCE_SCORE=0.55
    export CONVERSATION_CONTEXT_ENABLED=true
    export CONVERSATION_RECENT_TURNS=6
    export CONVERSATION_CONTEXT_MAX_CHARS=6000
    export CONVERSATION_QUESTION_MAX_CHARS=600
    export CONVERSATION_ANSWER_MAX_CHARS=1200
    export LANGGRAPH_TRACE_ENABLED=true
    if [[ -s "$QA_KNOWLEDGE_INDEX_PATH" ]]; then
      export QA_KNOWLEDGE_ENABLED=true
    else
      export QA_KNOWLEDGE_ENABLED=false
    fi
    export QA_KNOWLEDGE_INDEX_PATH
    export QA_KNOWLEDGE_CANDIDATE_LIMIT=30
    export QA_KNOWLEDGE_RERANK_ENABLED=true
    export QA_KNOWLEDGE_RERANK_LIMIT=8
    export OLLAMA_BASE_URL
    export OLLAMA_MODEL=qwen2.5:7b
    export OLLAMA_EMBEDDING_MODEL=bge-m3
    export OLLAMA_NUM_CTX=32768
    if [[ "$SYSTEM_API_PROFILE" == "live_product" ]]; then
      export KNOWLEDGE_BACKEND=lightrag
      export LIGHTRAG_ENABLED=true
      export LIGHTRAG_MODE=api
      export LIGHTRAG_API_BASE_URL
      export LIGHTRAG_API_KEY="${LIGHTRAG_API_KEY:-}"
      export LIGHTRAG_INPUT_DIR LIGHTRAG_WORKING_DIR
      export LIGHTRAG_TIMEOUT_SECONDS=60
      export LIGHTRAG_ALLOW_FALLBACK=false
      export LIGHTRAG_REINDEX_ON_START=false
    else
      export KNOWLEDGE_BACKEND=local_knowledge
      export LIGHTRAG_ENABLED=false
      export LIGHTRAG_ALLOW_FALLBACK=false
      export LIGHTRAG_REINDEX_ON_START=false
      unset LIGHTRAG_API_KEY || true
    fi
  else
    export CASE_REASONING_EVIDENCE_CONTROL_ENABLED=false
    unset CASE_REASONING_ISOLATED_DATABASE_ACK CASE_REASONING_EVIDENCE_CONTROL_SECRET || true
    export PRIVATE_EVAL_EVIDENCE_CONTROL_ENABLED=true
    export PRIVATE_EVAL_ISOLATED_DATABASE_ACK=I_CONFIRM_PRIVATE_ISOLATED_DATABASE
    export PRIVATE_EVAL_EVIDENCE_CONTROL_SECRET="$CARE_THREE_ROLE_BACKEND_EVAL_KEY"
    export KNOWLEDGE_BACKEND=local_knowledge
    export GRAPH_BACKEND=persistent_graph
    export MEMORY_BACKEND=persistent_memory
    export GRAPHITI_INGEST_CHAT_HISTORY=false
    export MEM0_INGEST_CHAT_HISTORY=false
    export CONVERSATION_CONTEXT_ENABLED=false
    export LANGGRAPH_TRACE_ENABLED=false
    export LIGHTRAG_ENABLED=false
    export LIGHTRAG_ALLOW_FALLBACK=false
    export QA_KNOWLEDGE_ENABLED=false
  fi
  export ALLOW_ADAPTER_FALLBACK=false
  export GRAPHITI_TELEMETRY_ENABLED=false
  export MEM0_TELEMETRY=false
  export LANGCHAIN_TRACING_V2=false
  export LANGSMITH_TRACING=false
  export DO_NOT_TRACK=1
  unset LANGSMITH_API_KEY || true

  export LLM_ENABLED=true
  export LLM_PROVIDER=openai_compatible
  export LLM_ALLOW_FALLBACK=false
  export LLM_TEMPERATURE=0
  export LLM_TOP_P=1.0
  export LLM_SEED=42
  export LLM_MAX_INPUT_TOKENS=6000
  export LLM_MAX_OUTPUT_TOKENS=1200
  export LLM_TIMEOUT_SECONDS=180
  export OLLAMA_HEALTH_TIMEOUT_SECONDS=10
  export OPENAI_COMPATIBLE_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
  export OPENAI_COMPATIBLE_API_KEY=local-private-eval
  export OPENAI_COMPATIBLE_MODEL="$served_name"

  bootstrap_private_backend_db > "$log" 2>&1
  (
    cd "${QZBC_ROOT}/backend"
    "$BACKEND_PY" -m uvicorn app.main:app \
      --host 127.0.0.1 --port "$BACKEND_PORT"
  ) >> "$log" 2>&1 &
  BACKEND_PID=$!
  printf '%s\n' "$BACKEND_PID" > "${LOG_ROOT}/shennong_sanyan_full.backend.pid"
  wait_url "http://127.0.0.1:${BACKEND_PORT}/health" \
    "isolated Shennong Sanyan backend" "$BACKEND_PID" 120
  wait_url "http://127.0.0.1:${BACKEND_PORT}/api/llm/health" \
    "isolated backend LLM" "$BACKEND_PID" 120
}

execute_project() {
  local output_dir="${RUN_ROOT}/shennong_sanyan_full"
  local output="${output_dir}/responses.private.jsonl"
  if output_complete "$output_dir" "$SYSTEM_API_PROFILE"; then
    printf 'SKIP_COMPLETE shennong_sanyan_full\n' >> "$DRIVER_MANIFEST"
    return 0
  fi
  if [[ "$CARE_SERVER_BACKEND" == "transformers" ]]; then
    start_care_transformers
  else
    start_vllm \
      shennong_sanyan_full "$CARE_MODEL" shennong-sanyan-care-opd "" true \
      "$CARE_VLLM_BIN" '{"enable_thinking":false}' "$CARE_MAX_MODEL_LEN"
  fi
  start_private_backend shennong-sanyan-care-opd
  local resume=()
  [[ -e "$output" ]] && resume=(--resume --retry-failures)
  local profile_ack=()
  if [[ "$SYSTEM_API_PROFILE" == "live_product" || "$SYSTEM_API_PROFILE" == "local_product" ]]; then
    profile_ack=(--full-product-stack-ack I_CONFIRM_LOCAL_RAG_GRAPH_MEMORY_EVAL)
  else
    profile_ack=(--external-memory-ack I_CONFIRM_EXTERNAL_MEMORY_AND_TELEMETRY_DISABLED)
  fi
  "$EVAL_PY" "${SCRIPT_ROOT}/execute_three_role_eval.py" \
    --input "$TASKS" \
    --output "$output" \
    --mode system_api \
    --system-api-profile "$SYSTEM_API_PROFILE" \
    --base-url "http://127.0.0.1:${BACKEND_PORT}" \
    --expected-backend-provider openai_compatible \
    --expected-backend-model shennong-sanyan-care-opd \
    --expected-backend-temperature 0 \
    --expected-backend-top-p 1.0 \
    --expected-backend-seed 42 \
    --expected-backend-max-input-tokens 6000 \
    --expected-backend-max-output-tokens 1200 \
    --timeout-seconds 900 \
    --retries 2 \
    --checkpoint-every 1 \
    --abort-on-first-failure \
    --provision-isolated-users \
    --execute \
    --isolated-eval-ack I_CONFIRM_PRIVATE_ISOLATED_EVAL \
    "${profile_ack[@]}" \
    "${resume[@]}"
  stop_pid "$BACKEND_PID" "isolated backend"
  BACKEND_PID=""
  stop_pid "$VLLM_PID" "project vLLM"
  VLLM_PID=""
  output_complete "$output_dir" "$SYSTEM_API_PROFILE" || {
    printf 'project Agent did not reach COMPLETE\n' >&2
    exit 4
  }
}

execute_baseline() {
  local system_id="$1"
  local model_path="$2"
  local served_name="$3"
  local reasoning_parser="${4:-}"
  local trust_remote_code="${5:-false}"
  local chat_template_kwargs="${6:-}"
  [[ -n "$chat_template_kwargs" ]] || chat_template_kwargs='{}'
  local output_dir="${RUN_ROOT}/${system_id}"
  local output="${output_dir}/responses.private.jsonl"
  if output_complete "$output_dir"; then
    printf 'SKIP_COMPLETE %s\n' "$system_id" >> "$DRIVER_MANIFEST"
    return 0
  fi
  start_vllm "$system_id" "$model_path" "$served_name" \
    "$reasoning_parser" "$trust_remote_code" "$BASELINE_VLLM_BIN"
  local resume=()
  [[ -e "$output" ]] && resume=(--resume --retry-failures)
  unset LOCAL_OPENAI_API_KEY || true
  "$EVAL_PY" "${SCRIPT_ROOT}/execute_three_role_eval.py" \
    --input "$TASKS" \
    --output "$output" \
    --mode openai_compatible \
    --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --model "$served_name" \
    --api-key-env LOCAL_OPENAI_API_KEY \
    --temperature 0 \
    --top-p 1.0 \
    --max-tokens 1200 \
    --generation-seed 42 \
    --timeout-seconds 240 \
    --retries 2 \
    --checkpoint-every 1 \
    --chat-template-kwargs-json "$chat_template_kwargs" \
    --execute \
    --isolated-eval-ack I_CONFIRM_PRIVATE_ISOLATED_EVAL \
    "${resume[@]}"
  stop_pid "$VLLM_PID" "$system_id vLLM"
  VLLM_PID=""
  output_complete "$output_dir" || {
    printf '%s did not reach COMPLETE\n' "$system_id" >&2
    exit 4
  }
}

run_models() {
  install -d -m 700 "$RUN_ROOT" "$LOG_ROOT" "$RUNTIME_ROOT"
  touch "$DRIVER_MANIFEST"
  chmod 600 "$DRIVER_MANIFEST"
  load_or_create_secrets
  execute_project
  release_ollama_models
  # HuatuoGPT-3-8B resolves to Qwen3ForCausalLM, but its released tokenizer
  # does not define the <think> start/end tokens required by vLLM's qwen3
  # reasoning parser.  The parser only segments server output and does not
  # alter weights, prompts or sampling.  Omit it for this baseline so the API
  # can start; keep the frozen selection record unchanged and record both the
  # frozen and effective values in the private driver manifest.
  printf '%s\n' \
    'RUNTIME_COMPATIBILITY label=huatuogpt3_local_medical_agent frozen_reasoning_parser=qwen3 effective_reasoning_parser=omitted reason=tokenizer_has_no_qwen3_think_delimiters' \
    >> "$DRIVER_MANIFEST"
  execute_baseline \
    huatuogpt3_local_medical_agent \
    "${MODEL_ROOT}/huatuogpt_3_8b" \
    huatuogpt3-local-medical-baseline "" false \
    '{"enable_thinking":false}'
  execute_baseline \
    glm4_local_role_model \
    "${MODEL_ROOT}/glm_4_9b_0414" \
    glm4-local-role-baseline "" false '{}'
  # The frozen InternLM3 snapshot ships the custom model classes required to
  # load its architecture.  vLLM rejects the snapshot before loading weights
  # unless trust_remote_code is enabled.  This executes only code already in
  # the immutable local snapshot selected by suite_selection_lock; it does not
  # fetch mutable code or change the selected model.  Record the compatibility
  # correction while leaving the frozen selection specification untouched.
  printf '%s\n' \
    'RUNTIME_COMPATIBILITY label=internlm3_local_role_model frozen_trust_remote_code=false effective_trust_remote_code=true reason=frozen_snapshot_requires_custom_model_code' \
    >> "$DRIVER_MANIFEST"
  execute_baseline \
    internlm3_local_role_model \
    "${MODEL_ROOT}/internlm3_8b_instruct" \
    internlm3-local-role-baseline "" true '{}'
}

verify_outputs() {
  "$EVAL_PY" "${SCRIPT_ROOT}/validate_local_four_agent_run.py" \
    --tasks "$TASKS" \
    --model-spec "$MODEL_SPEC" \
    --selection-lock "${MODEL_ROOT}/suite_selection_lock.private.json" \
    --run-root "$RUN_ROOT" \
    --project-system-api-profile "$SYSTEM_API_PROFILE" \
    --output "${RUN_ROOT}/four_agent_execution.private.json"
}

case "$PHASE" in
  run|all) ensure_single_writer ;;
esac

case "$PHASE" in
  preflight) preflight ;;
  download) preflight; download_models ;;
  run) preflight; run_models; verify_outputs ;;
  verify) verify_outputs ;;
  all) preflight; download_models; run_models; verify_outputs ;;
esac

trap - EXIT INT TERM
printf 'FOUR_AGENT_LOCAL_RUN_PHASE=%s COMPLETE\n' "$PHASE"
