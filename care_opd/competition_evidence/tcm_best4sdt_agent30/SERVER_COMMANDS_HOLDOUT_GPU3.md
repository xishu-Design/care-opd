# TCM-BEST4SDT 最终留出 30 例四系统盲审：GPU 3 完整运行命令

本轮是正式留出实验。旧 30 例及其全部输出只保留为开发审计；新样本在推理前同时排除
旧样本的病例 ID 与规范化诊断哈希。神农三言完整项目、HuatuoGPT-3、GLM-4-9B 与
InternLM3-8B 均使用同一份新题面从头生成，不复制任何旧回答。内容质量只交由专家盲审，
脚本仅在尚无真实模型输出的供应商/传输失败时重试。

## 1. 校验并解压

将压缩包和 `.sha256` 上传到：

```text
/opt/care-workspace/care_opd/runs/public_eval/tcm_best4sdt_agent30_bundle/
```

执行：

```bash
set +e
set +u
set +o pipefail
umask 077

export FIX_DIR=/opt/care-workspace/care_opd/runs/public_eval/tcm_best4sdt_agent30_bundle
export FIX_NAME=TCM_BEST4SDT_AGENT30_HOLDOUT_LOCAL_PRODUCT_GPU3_FAIR_BLIND_FINAL.tar.gz
cd "$FIX_DIR"
sha256sum -c "$FIX_NAME.sha256" || exit 2

export BUNDLE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle
export CARE_ROOT="$BUNDLE_ROOT/care_opd"
install -d -m 700 "$CARE_ROOT" /opt/care-data/four_agent_expert_blind
tar -xzf "$FIX_DIR/$FIX_NAME" -C "$CARE_ROOT" || exit 2

chmod 700 \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh" \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/serve_care_transformers.py" \
  "$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh"
```

## 2. 冻结环境

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
export PHYSICAL_GPUS=3
# 按操作者要求跳过“GPU 3 已有计算进程”拦截；其余任务、模型、服务和哈希预检仍保留。
export ACTIVE_GPU_PROCESS_CHECK_ACK=I_ACCEPT_GPU_CONTENTION_RISK
export CARE_MAX_MODEL_LEN=32768
export BASELINE_MAX_MODEL_LEN=8192
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_DISABLE_XET=1
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_HOST=127.0.0.1:11434
export QA_KNOWLEDGE_INDEX_PATH="$FULL_PRODUCT_ROOT/runtime/qa_knowledge/qa_index.db"

export TCM_BEST_TASK_ROOT="$CARE_ROOT/runs/public_eval/tcm_best4sdt_agent30_holdout_reference_free_fixed_seed20260827"
export TCM_BEST_RUN_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_four_system_runs
export TCM_BEST_REVIEW_ROOT=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_expert_blind
export REVIEWERS=expert_01,expert_02
export TCM_BEST_RUNNER="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh"

test -x "$EVAL_PY" || exit 2
test -x "$VLLM_BIN" || exit 2
test -x "$CARE_HF_PY" || exit 2
test -x "$BACKEND_PY" || exit 2
test -x "$TCM_BEST_RUNNER" || exit 2
test -s "$CARE_MODEL/config.json" || exit 2
"$BACKEND_PY" -c 'import fastapi, sqlalchemy, uvicorn, alembic, graphiti_core, mem0, qdrant_client, redislite; print("BACKEND_STACK_OK")' || exit 2
```

`QA_KNOWLEDGE_INDEX_PATH` 是可选增强；不存在时 runner 会明确记录未启用，不应创建空库。
本轮 `local_product` 不等待也不调用 LightRAG。

## 3. 确认 Ollama 的 Graphiti/Mem0 依赖

```bash
export OLLAMA_BIN="$(command -v ollama)"
test -x "$OLLAMA_BIN" || { echo '未找到 Ollama' >&2; exit 2; }

if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  nohup env OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_CONTEXT_LENGTH=32768 \
    OLLAMA_NUM_PARALLEL=1 "$OLLAMA_BIN" serve \
    >/opt/care-data/four_agent_expert_blind/ollama_holdout.log 2>&1 </dev/null &
  echo $! >/opt/care-data/four_agent_expert_blind/ollama_holdout.pid
fi
for _ in $(seq 1 60); do
  curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null || exit 2
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR>1 {print $1}' | grep -Eq '^qwen2\.5:7b(:latest)?$|^qwen2\.5:7b$' || exit 2
OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR>1 {print $1}' | grep -Eq '^bge-m3(:latest)?$|^bge-m3$' || exit 2
```

## 4. 预检并启动四系统完整重跑

新根必须不存在，防止误接旧结果：

```bash
if test -e "$TCM_BEST_RUN_ROOT" || test -e "$TCM_BEST_REVIEW_ROOT"; then
  echo '最终留出运行根或盲审根已存在，请人工核对并换一个全新路径' >&2
  exit 2
fi

bash "$TCM_BEST_RUNNER" prepare || exit 2
bash "$TCM_BEST_RUNNER" preflight || exit 2

export RUN_LOG=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu3.nohup.log
export RUN_PID=/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu3.pid

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
tail -F /opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_gpu3.nohup.log
```

## 5. 完成后核验并生成盲审表

```bash
export TCM_BEST_RUNNER="$CARE_ROOT/competition_evidence/tcm_best4sdt_agent30/run_tcm_best4sdt_agent30_gpu3.sh"
bash "$TCM_BEST_RUNNER" verify || exit 2
bash "$TCM_BEST_RUNNER" review || exit 2

find "$TCM_BEST_RUN_ROOT" -maxdepth 3 -type f -printf '%s %p\n' | sort
find "$TCM_BEST_REVIEW_ROOT" -maxdepth 3 -type f -printf '%s %p\n' | sort
```

四系统原始结果位于：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_four_system_runs/
```

专家 `.xlsx`、阅读版 HTML、盲映射和复核清单位于：

```text
/opt/care-data/four_agent_expert_blind/tcm_best4sdt_agent30_holdout_final_expert_blind/private/
```

只有专家填写完成后，才设置 `COMPLETED_RATINGS` 并运行 `score`；不得在揭盲前读取
`blind_mapping.private.jsonl`。
