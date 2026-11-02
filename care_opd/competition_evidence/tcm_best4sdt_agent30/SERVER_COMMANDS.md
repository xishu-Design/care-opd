# 服务器运行命令（`/opt/care-workspace`，物理 GPU 3）

> 本页仅保留需要完整 LightRAG 索引的可选增强轨，不是本轮30题正式主轨入口。
> 时间受限的正式 `local_product` 盲审请严格使用 `SERVER_COMMANDS_LOCAL_PRODUCT.md`，
> 两条轨的运行根、结果和汇报口径不得混合。

将下列两个文件手动上传到：

```text
/opt/care-workspace/care_opd/runs/public_eval/tcm_best4sdt_agent30_bundle/
```

文件名：

```text
TCM_BEST4SDT_AGENT30_FULL_PRODUCT_GPU3_FINAL.tar.gz
TCM_BEST4SDT_AGENT30_FULL_PRODUCT_GPU3_FINAL.tar.gz.sha256
```

然后执行：

```bash
set -Eeuo pipefail
umask 077

export FIX_DIR=/opt/care-workspace/care_opd/runs/public_eval/tcm_best4sdt_agent30_bundle
export FIX_NAME=TCM_BEST4SDT_AGENT30_FULL_PRODUCT_GPU3_FINAL.tar.gz
cd "$FIX_DIR"
sha256sum -c "$FIX_NAME.sha256"

export BUNDLE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle
export CARE_ROOT="$BUNDLE_ROOT/care_opd"
export QZBC_ROOT="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/qzbc_full_product"
export FULL_PRODUCT_ROOT=/opt/care-workspace/qzbc_source/qzbc-main
export MODEL_ROOT=/opt/care-data/four_agent_expert_blind/models
export CARE_MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model
install -d -m 700 "$CARE_ROOT" /opt/care-data/four_agent_expert_blind

# 先解压，使后续选取的 QZBC_ROOT 一定是本包冻结的完整后端快照。
tar -xzf "$FIX_DIR/$FIX_NAME" -C "$CARE_ROOT"
chmod 700 \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh" \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/serve_care_transformers.py" \
  "$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh"

export EVAL_PY=/opt/care-workspace/opdrevi_venv/bin/python
export VLLM_BIN=/opt/care-workspace/opdrevi_venv/bin/vllm
export BASELINE_VLLM_BIN="$VLLM_BIN"
export CARE_SERVER_BACKEND=transformers
export CARE_HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
test -x "$CARE_HF_PY"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$CARE_HF_PY" - "$CARE_MODEL" <<'PY'
import sys
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
assert config.model_type == "qwen3_5_text", config.model_type
assert type(config) in AutoModelForCausalLM._model_mapping, type(config)
assert torch.cuda.is_available()
print("CARE_HF_STACK_OK", transformers.__version__, torch.__version__)
PY

unset BACKEND_PY
for p in \
  "$FULL_PRODUCT_ROOT/.conda/bin/python" \
  "$BUNDLE_ROOT/qzbc_private_eval/.conda_eval/bin/python" \
  "$BUNDLE_ROOT/qzbc_private_eval/.venv/bin/python" \
  "$QZBC_ROOT/.conda_eval/bin/python" \
  "$QZBC_ROOT/.venv/bin/python" \
  /opt/care-data/qzbc_backend_venv/bin/python \
  "$EVAL_PY"; do
  if test -x "$p" && "$p" -c 'import fastapi, sqlalchemy, uvicorn, alembic, graphiti_core, mem0, qdrant_client, redislite' >/dev/null 2>&1; then
    export BACKEND_PY="$p"
    break
  fi
done
test -n "${BACKEND_PY:-}" || { echo "未找到包含Graphiti/Mem0的QZBC后端环境" >&2; exit 2; }

# 启动完整项目依赖的本地 AI 服务；已有健康实例时不重复启动。
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_BIN="$FULL_PRODUCT_ROOT/runtime/ollama/bin/ollama"
test -x "$OLLAMA_BIN" || export OLLAMA_BIN="$(command -v ollama)"
test -n "$OLLAMA_BIN" || { echo "未找到Ollama" >&2; exit 2; }

if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  nohup env OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_CONTEXT_LENGTH=32768 \
    OLLAMA_NUM_PARALLEL=1 "$OLLAMA_BIN" serve \
    > /opt/care-data/four_agent_expert_blind/ollama_full_product.log 2>&1 </dev/null &
  echo $! > /opt/care-data/four_agent_expert_blind/ollama_full_product.pid
fi
for _ in $(seq 1 60); do
  curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR>1 {print $1}' | grep -Eq '^qwen2\.5:7b(:latest)?$|^qwen2\.5:7b$'
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR>1 {print $1}' | grep -Eq '^bge-m3(:latest)?$|^bge-m3$'

export LIGHTRAG_ENV="$FULL_PRODUCT_ROOT/services/lightrag/.env"
if ! test -s "$LIGHTRAG_ENV"; then
  test -s "$FULL_PRODUCT_ROOT/services/lightrag/.env.example" || {
    echo "缺少LightRAG配置模板" >&2
    exit 2
  }
  install -m 600 "$FULL_PRODUCT_ROOT/services/lightrag/.env.example" "$LIGHTRAG_ENV"
fi
test -x "$FULL_PRODUCT_ROOT/services/lightrag/.venv/bin/lightrag-server" || {
  echo "缺少LightRAG运行环境；请先执行 $FULL_PRODUCT_ROOT/scripts/setup_local_ai.sh" >&2
  exit 2
}
set -a
source "$LIGHTRAG_ENV"
set +a
export LIGHTRAG_API_BASE_URL="http://127.0.0.1:${PORT:-9621}"
export LIGHTRAG_API_KEY="${LIGHTRAG_API_KEY:-}"
LIGHTRAG_SERVICE_DIR="$FULL_PRODUCT_ROOT/services/lightrag"
if [[ "${INPUT_DIR:-}" = /* ]]; then
  export LIGHTRAG_INPUT_DIR="$INPUT_DIR"
else
  export LIGHTRAG_INPUT_DIR="$(readlink -m "$LIGHTRAG_SERVICE_DIR/${INPUT_DIR:-../../runtime/lightrag/active/input}")"
fi
if [[ "${WORKING_DIR:-}" = /* ]]; then
  export LIGHTRAG_WORKING_DIR="$WORKING_DIR"
else
  export LIGHTRAG_WORKING_DIR="$(readlink -m "$LIGHTRAG_SERVICE_DIR/${WORKING_DIR:-../../runtime/lightrag/active/current}")"
fi
export LIGHTRAG_DOCS_PATH="$LIGHTRAG_INPUT_DIR/guardtcm_lightrag_docs.jsonl"
test -s "$LIGHTRAG_DOCS_PATH" || {
  echo "缺少已冻结的LightRAG文档映射；请先完成QZBC本地AI栈构建与激活" >&2
  exit 2
}
LIGHTRAG_CURL_ARGS=()
test -z "$LIGHTRAG_API_KEY" || LIGHTRAG_CURL_ARGS=(-H "X-API-Key: $LIGHTRAG_API_KEY")
if ! curl -fsS "${LIGHTRAG_CURL_ARGS[@]}" "$LIGHTRAG_API_BASE_URL/health" >/dev/null 2>&1; then
  nohup bash "$FULL_PRODUCT_ROOT/services/lightrag/start_lightrag.sh" \
    > /opt/care-data/four_agent_expert_blind/lightrag_full_product.log 2>&1 </dev/null &
  echo $! > /opt/care-data/four_agent_expert_blind/lightrag_full_product.pid
fi
for _ in $(seq 1 90); do
  curl -fsS "${LIGHTRAG_CURL_ARGS[@]}" "$LIGHTRAG_API_BASE_URL/health" >/dev/null 2>&1 && break
  sleep 2
done
if ! curl -fsS "${LIGHTRAG_CURL_ARGS[@]}" "$LIGHTRAG_API_BASE_URL/health" >/dev/null; then
  echo "LightRAG未成功启动，停止后续实验；服务日志如下：" >&2
  tail -n 120 /opt/care-data/four_agent_expert_blind/lightrag_full_product.log >&2 || true
  exit 2
fi
echo "LIGHTRAG_OK"

# 若服务器没有既有的 active 快照，则从本包冻结的 QZBC 内容数据构建完整文档映射，
# 分批写入当前 LightRAG，并在查询与图谱验证通过后才允许继续。该过程不读取30例任务；
# 首次索引耗时较长，但带状态文件，命令中断后原样重跑即可续接已完成批次。
LIGHTRAG_BOOTSTRAP="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/bootstrap_full_product_lightrag.py"
test -s "$LIGHTRAG_BOOTSTRAP"
"$BACKEND_PY" "$LIGHTRAG_BOOTSTRAP" \
  --qzbc-root "$QZBC_ROOT" \
  --docs-path "$LIGHTRAG_DOCS_PATH" \
  --working-dir "$LIGHTRAG_WORKING_DIR" \
  --api-base-url "$LIGHTRAG_API_BASE_URL" \
  --batch-size 128

export QA_KNOWLEDGE_INDEX_PATH="$FULL_PRODUCT_ROOT/runtime/qa_knowledge/qa_index.db"
if ! test -s "$QA_KNOWLEDGE_INDEX_PATH"; then
  (
    cd "$FULL_PRODUCT_ROOT/backend"
    "$BACKEND_PY" scripts/build_qa_knowledge_index.py
  )
fi
test -s "$QA_KNOWLEDGE_INDEX_PATH"

export PHYSICAL_GPUS=3
export HF_HUB_DISABLE_XET=1
export TCM_BEST_TASK_ROOT="$CARE_ROOT/runs/public_eval/tcm_best4sdt_agent30_fixed_seed20260825"
export TCM_BEST_RUN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_full_product_four_system_runs
export TCM_BEST_REVIEW_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_full_product_expert_blind
export TCM_BEST_DATA="$CARE_ROOT/public_eval/tcm_best4sdt/data/TCM_SDT.json"
export TCM_BEST_OVERLAP_SOURCE="$QZBC_ROOT/backend/app/content/data/learning_cases.json"
export REVIEWERS=expert_01,expert_02
export TCM_BEST_RUNNER="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh"

bash "$TCM_BEST_RUNNER" prepare
sha256sum "$TCM_BEST_TASK_ROOT/tcm_best4sdt_agent30.tasks.private.jsonl"
```

默认冻结两位专家，以便计算一致性；若本轮确实只有一位专家，必须在生成盲表之前改为
`export REVIEWERS=expert_01`，汇总时 `COMPLETED_RATINGS` 也只填这一份。盲表生成后不得
再改变专家列表。

冻结任务文件预期 SHA-256：

```text
01f802a2973fa615f007d2305d59e1ad0a893013614fef1f11fd7a946b8f92d5
```

卡3空闲后启动：

新启动器会读取本机 `vllm serve --help`：vLLM 0.11 只传
`--reasoning-parser`，不会误传不支持的 `--enable-reasoning`；存在新版
`--enable-log-requests` 时保持该显式开启项缺省，从而关闭请求内容日志，也不再触发
`--disable-log-requests` 的弃用警告。

```bash
nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits

RUN_LOG=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_full_product_gpu3.nohup.log
RUN_PID=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_full_product_gpu3.pid

# 不再单独运行 preflight；all 内部已包含同一套完整预检，避免 Ollama 探针
# 首次加载模型后被第二次显卡空闲检查误判为外部占用。
nohup bash "$TCM_BEST_RUNNER" all >"$RUN_LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$RUN_PID"
sleep 3

if kill -0 "$pid" 2>/dev/null; then
  printf '已启动：PID=%s\n日志=%s\n' "$pid" "$RUN_LOG"
  tail -F "$RUN_LOG"
else
  printf '进程在启动阶段退出，最后 120 行日志如下：\n' >&2
  tail -n 120 "$RUN_LOG" >&2
fi
```

脚本依次生成 `30 × 4 = 120` 份回答，随后生成两位专家各自独立的 CSV；若环境已有
openpyxl，也会生成 XLSX。输出位置：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_full_product_four_system_runs/
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_full_product_expert_blind/private/
```

将每位专家自己的 `reviewer_N.private.xlsx`（若没有 XLSX 则用 CSV）及同编号
`reviewer_N.reading.private.html` 分别发送；HTML 便于逐病例纵向阅读，评分仍填回
CSV/XLSX。不要发送 `blind_mapping.private.jsonl` 或 `expert_blind_secret.private`。

专家填完并回传到服务器后，指定实际路径：

```bash
export COMPLETED_RATINGS=/opt/care-data/four_agent_expert_blind/completed/reviewer_1.completed.xlsx,/opt/care-data/four_agent_expert_blind/completed/reviewer_2.completed.xlsx
bash "$TCM_BEST_RUNNER" score
```

汇总结果位于：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_full_product_expert_blind/summary_public/
```

推理完成后回传包：

```bash
RETURN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_full_product_return
test ! -e "$RETURN_ROOT" || { echo "回传目录已存在，请先核对" >&2; exit 2; }
install -d -m 700 "$RETURN_ROOT"
cp -a "$TCM_BEST_TASK_ROOT" "$RETURN_ROOT/frozen_task"
cp -a "$TCM_BEST_RUN_ROOT" "$RETURN_ROOT/four_system_runs"
cp -a "$TCM_BEST_REVIEW_ROOT" "$RETURN_ROOT/expert_blind_review"
cp -a "$MODEL_ROOT/suite_selection_lock.private.json" "$RETURN_ROOT/"
cp -a "$CARE_ROOT/competition_evidence/local_four_agent_server/model_suite.json" "$RETURN_ROOT/"
tar -czf /opt/care-data/four_agent_expert_blind/TCM_BEST4SDT_AGENT30_FULL_PRODUCT_RESULTS.PRIVATE.tar.gz \
  -C /opt/care-data/four_agent_expert_blind tcm_best4sdt_agent30_full_product_return
sha256sum /opt/care-data/four_agent_expert_blind/TCM_BEST4SDT_AGENT30_FULL_PRODUCT_RESULTS.PRIVATE.tar.gz \
  > /opt/care-data/four_agent_expert_blind/TCM_BEST4SDT_AGENT30_FULL_PRODUCT_RESULTS.PRIVATE.tar.gz.sha256
```

若专家尚未完成，回传推理与空白盲审表即可；不要预填结果。
