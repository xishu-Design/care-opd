# TCM-BEST4SDT 最终留出 30 例四系统盲审：GPU 6 完整运行命令

旧 30 例及其输出仅作开发审计。本轮使用与旧样本病例 ID、规范化诊断哈希均不重叠的
30 例留出集，神农三言完整项目和三个对照系统全部从头运行。内容质量不触发重生成；
只有在尚未形成真实模型输出的供应商或传输失败时才允许基础设施重试。

## 1. 上传、校验、解压

将压缩包和 `.sha256` 上传到：

```text
/opt/care-workspace/care_opd/runs/public_eval/tcm_best4sdt_agent30_bundle/
```

```bash
set +e
set +u
set +o pipefail
umask 077

export FIX_DIR=/opt/care-workspace/care_opd/runs/public_eval/tcm_best4sdt_agent30_bundle
export FIX_NAME=TCM_BEST4SDT_AGENT30_HOLDOUT_GPU6_FROZEN_CANARY_XLSX_REVIEW_FIX_FINAL.tar.gz
cd "$FIX_DIR"
sha256sum -c "$FIX_NAME.sha256" || exit 2

export BUNDLE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle
export CARE_ROOT="$BUNDLE_ROOT/care_opd"
install -d -m 700 "$CARE_ROOT" /opt/care-data/four_agent_expert_blind
tar -xzf "$FIX_DIR/$FIX_NAME" -C "$CARE_ROOT" || exit 2

chmod 700 \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh" \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/serve_care_transformers.py" \
  "$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh" \
  "$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu6.sh"
```

## 2. 冻结 GPU 6 运行环境

```bash
export BUNDLE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle
export CARE_ROOT="$BUNDLE_ROOT/care_opd"
export QZBC_ROOT="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/qzbc_full_product"
export FULL_PRODUCT_ROOT=/opt/care-workspace/qzbc_source/qzbc-main
export MODEL_ROOT=/opt/care-data/four_agent_expert_blind/models
export CARE_MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model

export EVAL_PY=/opt/care-workspace/opdrevi_venv/bin/python
export VLLM_BIN=/opt/care-workspace/opdrevi_venv/bin/vllm
export BASELINE_VLLM_BIN="$VLLM_BIN"
export CARE_SERVER_BACKEND=transformers
export CARE_HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export BACKEND_PY=/opt/care-workspace/four_agent_expert_blind_bundle/qzbc_private_eval/.conda_eval/bin/python

export SYSTEM_API_PROFILE=local_product
export PHYSICAL_GPUS=6
# 按操作者要求不因 GPU 6 已存在计算进程而拦截；任务、模型、服务和哈希预检仍保留。
export ACTIVE_GPU_PROCESS_CHECK_ACK=I_ACCEPT_GPU_CONTENTION_RISK
export CARE_MAX_MODEL_LEN=32768
export BASELINE_MAX_MODEL_LEN=8192
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_DISABLE_XET=1
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_HOST=127.0.0.1:11434
export QA_KNOWLEDGE_INDEX_PATH="$FULL_PRODUCT_ROOT/runtime/qa_knowledge/qa_index.db"

export TCM_BEST_TASK_ROOT="$CARE_ROOT/runs/public_eval/tcm_best4sdt_agent30_holdout_reference_free_fixed_seed20260827"
export TCM_BEST_RUN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6_four_system_runs
export TCM_BEST_REVIEW_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6_expert_blind
export REVIEWERS=expert_01,expert_02
export TCM_BEST_RUNNER="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu6.sh"

test -x "$EVAL_PY" || exit 2
test -x "$VLLM_BIN" || exit 2
test -x "$CARE_HF_PY" || exit 2
test -x "$BACKEND_PY" || exit 2
test -x "$TCM_BEST_RUNNER" || exit 2
test -s "$CARE_MODEL/config.json" || exit 2
"$BACKEND_PY" -c 'import fastapi, sqlalchemy, uvicorn, alembic, graphiti_core, mem0, qdrant_client, redislite; print("BACKEND_STACK_OK")' || exit 2
```

QA 索引不存在时会被明确记录为未启用；不要创建空数据库。本轮 `local_product` 不依赖
LightRAG。

## 3. 确认 Graphiti/Mem0 使用的 Ollama 服务

```bash
export OLLAMA_BIN="$(command -v ollama)"
test -x "$OLLAMA_BIN" || { echo '未找到 Ollama' >&2; exit 2; }

if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  nohup env OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_CONTEXT_LENGTH=32768 \
    OLLAMA_NUM_PARALLEL=1 "$OLLAMA_BIN" serve \
    >/opt/care-data/four_agent_expert_blind/ollama_holdout_gpu6.log 2>&1 </dev/null &
  echo $! >/opt/care-data/four_agent_expert_blind/ollama_holdout_gpu6.pid
fi
for _ in $(seq 1 60); do
  curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null || exit 2
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR>1 {print $1}' | grep -Eq '^qwen2\.5:7b(:latest)?$|^qwen2\.5:7b$' || exit 2
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR>1 {print $1}' | grep -Eq '^bge-m3(:latest)?$|^bge-m3$' || exit 2
```

## 4. 预检并在 GPU 6 后台启动

```bash
# 在启动块再次显式锁定卡号，避免切换终端后继承到旧值。
export PHYSICAL_GPUS=6
export ACTIVE_GPU_PROCESS_CHECK_ACK=I_ACCEPT_GPU_CONTENTION_RISK

if test -e "$TCM_BEST_RUN_ROOT" || test -e "$TCM_BEST_REVIEW_ROOT"; then
  echo 'GPU 6 最终留出运行根或盲审根已存在，请人工核对并换一个全新路径' >&2
  exit 2
fi

bash "$TCM_BEST_RUNNER" prepare || exit 2
bash "$TCM_BEST_RUNNER" preflight || exit 2

export RUN_LOG=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6.nohup.log
export RUN_PID=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6.pid

nohup bash "$TCM_BEST_RUNNER" all >"$RUN_LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$RUN_PID"
disown "$pid" 2>/dev/null || true

sleep 3
ps -fp "$pid" || true
tail -n 120 "$RUN_LOG"
```

持续查看：

```bash
tail -F /opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6.nohup.log
```

## 5. 完成后核验并生成专家盲审表

```bash
export PHYSICAL_GPUS=6
export TCM_BEST_RUNNER="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu6.sh"

# 单独记录核验和盲审表生成日志。这里不要使用 `|| exit 2`，否则任一阶段
# 返回非零值都会直接关闭当前交互终端，反而看不到真正的失败原因。
set +e
VERIFY_LOG=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6.verify.log
REVIEW_LOG=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6.review.log

bash "$TCM_BEST_RUNNER" verify >"$VERIFY_LOG" 2>&1
verify_rc=$?
echo "VERIFY_RC=$verify_rc LOG=$VERIFY_LOG"
tail -n 120 "$VERIFY_LOG"

if (( verify_rc == 0 )); then
  if "$EVAL_PY" -c 'import openpyxl; print("OPENPYXL_OK", openpyxl.__version__)'; then
    bash "$TCM_BEST_RUNNER" review >"$REVIEW_LOG" 2>&1
    review_rc=$?
    echo "REVIEW_RC=$review_rc LOG=$REVIEW_LOG"
    tail -n 160 "$REVIEW_LOG"
  else
    review_rc=126
    echo "缺少 openpyxl，尚未冻结盲审包；先安装 openpyxl，再重新执行本节。"
  fi
else
  review_rc=125
  echo "核验未通过，未生成盲审表；终端保持打开，请先检查核验日志。"
fi

find "$TCM_BEST_RUN_ROOT" -maxdepth 3 -type f -printf '%s %p\n' | sort
find "$TCM_BEST_REVIEW_ROOT" -maxdepth 3 -type f -printf '%s %p\n' | sort
```

四系统输出：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6_four_system_runs/
```

专家表格、阅读版 HTML 和私有盲映射：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu6_expert_blind/private/
```

专家完成前不得读取 `blind_mapping.private.jsonl`，也不要提前运行 `score`。
