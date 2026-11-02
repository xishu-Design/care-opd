# 服务器执行命令（物理 GPU 3，免下载 CARE 推理环境）

以下假定最终增量包及其 `.sha256` 已上传到
`/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_agent30_bundle/`。现有四系统程序包位于
`/opt/care-workspace/four_agent_expert_blind_bundle`；模型缓存和本轮输出继续放在空间充足的
`/opt/care-data`，避免写满根分区。CARE 模型复用服务器上已经验证过的
`qzbc_baseline_gpu4` Transformers 环境，三个冻结对照模型继续使用原 vLLM 环境；
不再安装新版 vLLM、Torch 或 CUDA 依赖。

```bash
export FIX_DIR=/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_agent30_bundle
export FIX_NAME=TCMEVAL_PA_AGENT30_GPU3_NO_DOWNLOAD_FINAL.tar.gz
cd "$FIX_DIR"
sha256sum -c "$FIX_NAME.sha256"

unset BUNDLE_ROOT
for d in \
  /opt/care-workspace/four_agent_expert_blind_bundle \
  /opt/care-data/four_agent_expert_blind \
  /opt/care-data/four_agent_expert_blind_bundle; do
  if test -d "$d/care_opd" -a -d "$d/qzbc_private_eval"; then
    export BUNDLE_ROOT="$d"
    break
  fi
done
test -n "${BUNDLE_ROOT:-}" || { echo "未找到现有四系统解包目录" >&2; exit 2; }

export CARE_ROOT="$BUNDLE_ROOT/care_opd"
export QZBC_ROOT="$BUNDLE_ROOT/qzbc_private_eval"

unset MODEL_ROOT
for d in \
  /opt/care-data/four_agent_expert_blind/models \
  /opt/care-data/models/four_agent_expert_blind_local \
  "$BUNDLE_ROOT/models"; do
  if test -d "$d/huatuogpt_3_8b"; then
    export MODEL_ROOT="$d"
    break
  fi
done
test -n "${MODEL_ROOT:-}" || { echo "未找到已有模型下载目录" >&2; exit 2; }

unset CARE_MODEL
for d in \
  /opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model \
  /opt/care-data/care_opd/runs/qzbc_base_model_3gpu/final_model; do
  if test -s "$d/config.json"; then
    export CARE_MODEL="$d"
    break
  fi
done
test -n "${CARE_MODEL:-}" || { echo "未找到 CARE-OPD final_model" >&2; exit 2; }

export EVAL_PY=/opt/care-workspace/opdrevi_venv/bin/python
export VLLM_BIN=/opt/care-workspace/opdrevi_venv/bin/vllm
export BASELINE_VLLM_BIN="$VLLM_BIN"
export CARE_SERVER_BACKEND=transformers
export CARE_HF_PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
test -x "$EVAL_PY" -a -x "$BASELINE_VLLM_BIN" -a -x "$CARE_HF_PY"

# 只做本地离线兼容性检查，不访问 Hugging Face，也不下载任何文件。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$CARE_HF_PY" - "$CARE_MODEL" <<'PY'
import sys, torch, transformers
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(
    sys.argv[1], local_files_only=True, trust_remote_code=False
)
assert config.model_type == "qwen3_5_text", config.model_type
assert type(config) in AutoModelForCausalLM._model_mapping, type(config)
print(
    "CARE_TRANSFORMERS_STACK_OK",
    transformers.__version__, torch.__version__, torch.version.cuda,
)
PY

unset BACKEND_PY
for p in \
  "$QZBC_ROOT/.conda_eval/bin/python" \
  "$QZBC_ROOT/.venv/bin/python" \
  /opt/care-data/qzbc_backend_venv/bin/python \
  /opt/care-workspace/qzbc_source/qzbc-main/.conda/bin/python \
  "$EVAL_PY"; do
  if test -x "$p" && "$p" -c 'import fastapi, sqlalchemy, uvicorn, alembic' >/dev/null 2>&1; then
    export BACKEND_PY="$p"
    break
  fi
done
test -n "${BACKEND_PY:-}" || { echo "未找到通过 fastapi/sqlalchemy/uvicorn/alembic 检查的后端环境" >&2; exit 2; }

(
  cd "$QZBC_ROOT/backend"
  "$BACKEND_PY" -c \
    'import fastapi, sqlalchemy, uvicorn, alembic; import app.main; print("BACKEND_IMPORT_OK")'
)

tar -xzf "$FIX_DIR/$FIX_NAME" -C "$CARE_ROOT"
chmod 700 \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh" \
  "$CARE_ROOT/competition_evidence/tcmeval_pa_agent30/run_tcmeval_pa_agent30_gpu3.sh" \
  "$CARE_ROOT/competition_evidence/local_four_agent_server/serve_care_transformers.py"

export PHYSICAL_GPUS=3
export HF_HUB_DISABLE_XET=1
export TCMEVAL_TASK_ROOT="$CARE_ROOT/runs/public_eval/tcmeval_pa_agent30_fixed_seed20260825"
export TCMEVAL_RUN_ROOT=/opt/care-data/four_agent_expert_blind/tcmeval_pa_agent30_four_system_runs
export TCMEVAL_SCORE_ROOT=/opt/care-data/four_agent_expert_blind/tcmeval_pa_agent30_scored
export TCMEVAL_DATA="$CARE_ROOT/public_eval/tcmeval_pa/data/TCMEval-PA.xlsx"
export TCM30_RUNNER="$CARE_ROOT/competition_evidence/tcmeval_pa_agent30/run_tcmeval_pa_agent30_gpu3.sh"
export FOUR_SYSTEM_RUNNER="$CARE_ROOT/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh"

bash "$TCM30_RUNNER" prepare
sha256sum "$TCMEVAL_TASK_ROOT/tcmeval_pa_agent30.tasks.jsonl"
```

冻结任务文件的期望 SHA-256 为：

```text
084034f8ea82db2d4caf32631489e78fea78050ea4365aaa9eb4fc32c205abdf
```

三个对照模型的下载必须先完成。检查：

```bash
"$EVAL_PY" - "$MODEL_ROOT/suite_download_manifest.private.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("status") == "COMPLETE" and x.get("model_count") == 3, x
print("MODEL_DOWNLOAD_COMPLETE=3")
PY
```

若该检查失败，先确认是否已有下载进程：

```bash
pgrep -af 'download_model_suite.py|run_four_agent_local_server.sh download' || true
```

若能看到已有进程，不要启动第二个下载进程，继续查看原下载日志。仅当确认没有上述进程时，
使用下面的完整命令续传；Hugging Face 会复用已有缓存和半成品：

```bash
DOWNLOAD_LOG=/opt/care-data/four_agent_expert_blind/four_agent_local_download.resume.nohup.log
nohup env \
  CARE_ROOT="$CARE_ROOT" \
  QZBC_ROOT="$QZBC_ROOT" \
  EVAL_PY="$EVAL_PY" \
  VLLM_BIN="$VLLM_BIN" \
  BASELINE_VLLM_BIN="$BASELINE_VLLM_BIN" \
  CARE_SERVER_BACKEND="$CARE_SERVER_BACKEND" \
  CARE_HF_PY="$CARE_HF_PY" \
  BACKEND_PY="$BACKEND_PY" \
  PHYSICAL_GPUS=3 \
  CARE_MODEL="$CARE_MODEL" \
  MODEL_ROOT="$MODEL_ROOT" \
  TASKS="$TCMEVAL_TASK_ROOT/tcmeval_pa_agent30.tasks.jsonl" \
  EXPECTED_TASKS=30 \
  RUN_ROOT="$TCMEVAL_RUN_ROOT" \
  HF_HUB_DISABLE_XET=1 \
  bash "$FOUR_SYSTEM_RUNNER" download > "$DOWNLOAD_LOG" 2>&1 </dev/null &
echo $! > /opt/care-data/four_agent_expert_blind/four_agent_local_download.resume.pid
tail -F "$DOWNLOAD_LOG"
```

下载日志出现 `FOUR_AGENT_LOCAL_RUN_PHASE=download COMPLETE` 后，再执行上面的下载完成检查。
下载完成后，确认卡3空闲并启动：

```bash
nvidia-smi -i 3 --query-gpu=index,name,memory.total,memory.free \
  --format=csv,noheader,nounits
nvidia-smi -i 3 --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits

bash "$TCM30_RUNNER" preflight

RUN_LOG=/opt/care-data/four_agent_expert_blind/tcmeval_pa_agent30_gpu3.no_download.nohup.log
nohup bash "$TCM30_RUNNER" all > "$RUN_LOG" 2>&1 </dev/null &
echo $! > /opt/care-data/four_agent_expert_blind/tcmeval_pa_agent30_gpu3.no_download.pid
tail -F "$RUN_LOG"
```

脚本在物理卡3上依次运行四套系统，共生成 `30 × 4 = 120` 份回答；下层进程看到的
设备是逻辑 `cuda:0`。本项目完整 Agent 的日志出现
`CARE_TRANSFORMERS_SERVER_LISTENING` 即表示免下载推理服务已就绪。不要终止卡3上不属于
本轮的进程。

完成后查看：

```bash
test -s "$TCMEVAL_RUN_ROOT/four_agent_execution.private.json"
test -s "$TCMEVAL_SCORE_ROOT/summary.public.json"
cat "$TCMEVAL_SCORE_ROOT/REPORT.md"
```

回传结果：

```bash
RETURN_ROOT=/opt/care-data/four_agent_expert_blind/tcmeval_pa_agent30_return
test ! -e "$RETURN_ROOT" || { echo "回传目录已存在，请先核对" >&2; exit 2; }
install -d -m 700 "$RETURN_ROOT"
cp -a "$TCMEVAL_TASK_ROOT/tcmeval_pa_agent30.tasks.jsonl" "$RETURN_ROOT/"
cp -a "$TCMEVAL_TASK_ROOT/tcmeval_pa_agent30.selection_manifest.json" "$RETURN_ROOT/"
install -d -m 700 "$RETURN_ROOT/four_system_runs"
for system in \
  shennong_sanyan_full \
  huatuogpt3_local_medical_agent \
  glm4_local_role_model \
  internlm3_local_role_model; do
  cp -a "$TCMEVAL_RUN_ROOT/$system" "$RETURN_ROOT/four_system_runs/$system"
done
cp -a "$TCMEVAL_RUN_ROOT/four_agent_execution.private.json" \
  "$RETURN_ROOT/four_system_runs/"
cp -a "$TCMEVAL_SCORE_ROOT" "$RETURN_ROOT/scored"
cp -a "$MODEL_ROOT/suite_selection_lock.private.json" "$RETURN_ROOT/"
cp -a "$CARE_ROOT/competition_evidence/local_four_agent_server/model_suite.json" \
  "$RETURN_ROOT/"
tar -czf /opt/care-data/four_agent_expert_blind/TCMEVAL_PA_AGENT30_RESULTS.PRIVATE.tar.gz \
  -C /opt/care-data/four_agent_expert_blind tcmeval_pa_agent30_return
sha256sum /opt/care-data/four_agent_expert_blind/TCMEVAL_PA_AGENT30_RESULTS.PRIVATE.tar.gz \
  > /opt/care-data/four_agent_expert_blind/TCMEVAL_PA_AGENT30_RESULTS.PRIVATE.tar.gz.sha256
```

该回传包不包含 `runtime/`、SQLite 数据库、临时密钥或运行日志。
