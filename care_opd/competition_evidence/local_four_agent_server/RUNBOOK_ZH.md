# 四方本地推理与专家盲评：服务器完整指令

## 1. 实验口径

本轮比较四个回答来源：

1. **神农三言完整项目**：三角色中医药智能体系统与 CARE-OPD 后训练模型的完整组合；
2. `FreedomIntelligence/HuatuoGPT-3-8B`：中文医疗领域公开权重模型基线；
3. `zai-org/GLM-4-9B-0414`：中文通用公开权重模型基线；
4. `internlm/internlm3-8b-instruct`：另一独立中文模型生态的公开权重基线。

三个公开模型都在受控服务器上通过 loopback OpenAI-compatible 接口运行，
收到完全相同的冻结角色提示与病例材料。它们没有复制神农三言完整的工具、
角色路由和业务编排，因此正式材料应称为“本地公开权重模型基线”，不要写成
三个与神农三言结构完全等价的完整 Agent。实验可以支持完整项目的匿名专家偏好，
不能单独归因 CARE-OPD，也不能写成临床疗效。

冻结任务为 12 个真实病例族 × 2 个证据条件 × 3 个角色，共 72 题；四方应得到
288 份唯一冻结回答。真实记录由数据提供方提供，已去标识化并结构化；所有推理均在
受控服务器本地完成，不发送到第三方 API。

基线在任何回答生成前由脚本解析并锁定 Hugging Face 完整 commit SHA。三者统一：

- 官方聊天模板；
- BF16、8K 上下文；
- `temperature=0`、`top_p=1`、`max_tokens=1200`、seed 42；
- 单请求并发、最多 2 次冻结重试；
- 不做 best-of-N，不按结果选择性重跑；
- HuatuoGPT-3 关闭 thinking，专家只看最终回答。

HuatuoGPT-3 是中文医疗模型，不是中医专用模型。其 Hugging Face 模型卡元数据声明
Apache-2.0，但仓库文件树可能没有单独的 LICENSE 文件；归档时保留锁定模型卡与 SHA，
按此准确口径记录。

## 2. Windows 本机生成 PRIVATE 服务器包

在 PowerShell 执行：

```powershell
Set-Location /opt/care-workspace\care_opd
$Py = "..\.hf_upload_venv\Scripts\python.exe"
$Tasks = "runs\private_real_thyroid_external\three_role_eval\paired_fact_n12\three_role_tasks.private.jsonl"
$BundleDir = "runs\private_real_thyroid_external\three_role_eval\server_bundle_local_four_agent"
New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null
$Bundle = "$BundleDir\FOUR_AGENT_LOCAL_BLIND.PRIVATE.tar.gz"

if (Test-Path $Bundle) {
    throw "目标包已存在；请保留旧包并使用新的空目录，不要覆盖：$Bundle"
}

& $Py competition_evidence\local_four_agent_server\build_private_server_bundle.py `
  --care-root . `
  --qzbc-root ..\qzbc_source\qzbc-main `
  --tasks $Tasks `
  --output $Bundle

Get-FileHash $Bundle -Algorithm SHA256
Get-Content "$Bundle.sha256" -Encoding ASCII
```

该压缩包包含去标识化真实病例任务，属于 PRIVATE 服务器输入，不得上传 GitHub、
Hugging Face、公共对象存储或竞赛提交包。

把包和校验文件传至服务器：

```powershell
scp $Bundle "$Bundle.sha256" root@gpu-a800-094:/opt/care-workspace/
```

## 3. 服务器首次准备

手动登录后执行：

```bash
cd /opt/care-workspace
sha256sum -c FOUR_AGENT_LOCAL_BLIND.PRIVATE.tar.gz.sha256

BUNDLE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle
test ! -e "$BUNDLE_ROOT" || {
  echo "目标目录已存在。若是同一次中断任务请直接复用；若是新实验请换新目录，禁止覆盖。" >&2
  exit 2
}
tar -xzf FOUR_AGENT_LOCAL_BLIND.PRIVATE.tar.gz -C /opt/care-workspace
chmod -R go-rwx "$BUNDLE_ROOT"
```

确认硬件、磁盘和现有环境：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
  --format=csv,noheader,nounits
df -h /opt/care-workspace /opt/care-data 2>/dev/null || true

EVAL_PY=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
VLLM_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/vllm
"$EVAL_PY" -c 'import torch, vllm, huggingface_hub; print(torch.__version__, vllm.__version__, huggingface_hub.__version__)'
"$VLLM_BIN" serve --help | grep -- '--disable-log-requests'
```

不要修改既有训练环境。若上面缺少 `huggingface_hub` 或 vLLM，创建本轮独立环境：

```bash
python3.12 -m venv /opt/care-workspace/venvs/four_agent_blind_vllm
/opt/care-workspace/venvs/four_agent_blind_vllm/bin/python -m pip install --upgrade pip
/opt/care-workspace/venvs/four_agent_blind_vllm/bin/python -m pip install \
  'huggingface_hub>=0.34,<2' 'vllm>=0.9.0'
export EVAL_PY=/opt/care-workspace/venvs/four_agent_blind_vllm/bin/python
export VLLM_BIN=/opt/care-workspace/venvs/four_agent_blind_vllm/bin/vllm
```

优先复用已经验证过的 qzbc Python 3.12 环境：

```bash
if test -x /opt/care-workspace/qzbc_source/qzbc-main/.conda/bin/python; then
  BACKEND_PY=/opt/care-workspace/qzbc_source/qzbc-main/.conda/bin/python
else
  python3.12 -m venv "$BUNDLE_ROOT/qzbc_private_eval/.venv"
  BACKEND_PY="$BUNDLE_ROOT/qzbc_private_eval/.venv/bin/python"
  "$BACKEND_PY" -m pip install --upgrade pip
  "$BACKEND_PY" -m pip install -r "$BUNDLE_ROOT/qzbc_private_eval/requirements.txt"
fi
"$BACKEND_PY" -c 'import fastapi, sqlalchemy, uvicorn; print("BACKEND_STACK_OK")'
```

设置本轮固定路径。`PHYSICAL_GPUS` 必须替换成现场空闲卡；8–9B BF16 与 4B CARE
模型在 A800 80GB 上均可单卡运行，因此默认只占一张卡并依次加载：

```bash
export BUNDLE_ROOT=/opt/care-workspace/four_agent_expert_blind_bundle
export CARE_ROOT="$BUNDLE_ROOT/care_opd"
export QZBC_ROOT="$BUNDLE_ROOT/qzbc_private_eval"
export EVAL_PY="${EVAL_PY:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python}"
export VLLM_BIN="${VLLM_BIN:-/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/vllm}"
export BACKEND_PY
export PHYSICAL_GPUS=4
export CARE_MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model
export MODEL_ROOT=/opt/care-workspace/models/four_agent_expert_blind_local
export RUN_ROOT="$CARE_ROOT/runs/private_real_thyroid_external/three_role_eval/four_agent_expert_blind_local"
export HF_HUB_DISABLE_XET=1

SCRIPT="$CARE_ROOT/competition_evidence/local_four_agent_server/run_four_agent_local_server.sh"
chmod 700 "$SCRIPT"
bash "$SCRIPT" preflight
```

若提示 GPU 占用，先重新查看 `nvidia-smi` 并把 `PHYSICAL_GPUS` 换成空闲卡。不要终止
不属于本轮的进程。

## 4. 下载并锁定三个公开模型

三个仓库均为公开下载，通常不需要 HF token。若服务器已有只读 token，只通过环境变量
`HF_TOKEN` 传入，不要写进脚本或日志。

```bash
DOWNLOAD_LOG=/opt/care-workspace/four_agent_local_download.nohup.log
nohup env \
  BUNDLE_ROOT="$BUNDLE_ROOT" CARE_ROOT="$CARE_ROOT" QZBC_ROOT="$QZBC_ROOT" \
  EVAL_PY="$EVAL_PY" VLLM_BIN="$VLLM_BIN" BACKEND_PY="$BACKEND_PY" \
  PHYSICAL_GPUS="$PHYSICAL_GPUS" CARE_MODEL="$CARE_MODEL" \
  MODEL_ROOT="$MODEL_ROOT" RUN_ROOT="$RUN_ROOT" HF_HUB_DISABLE_XET=1 \
  bash "$SCRIPT" download > "$DOWNLOAD_LOG" 2>&1 </dev/null &
echo $! > /opt/care-workspace/four_agent_local_download.pid
tail -F "$DOWNLOAD_LOG"
```

下载中断时用完全相同的命令重跑即可续传。不要删除半成品，也不要更改
`model_suite.json`。完成后必须存在：

```bash
test -s "$MODEL_ROOT/suite_selection_lock.private.json"
test -s "$MODEL_ROOT/suite_download_manifest.private.json"
"$EVAL_PY" - "$MODEL_ROOT/suite_download_manifest.private.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
assert x['status'] == 'COMPLETE' and x['model_count'] == 3, x
print('MODEL_DOWNLOAD_COMPLETE=3')
PY
```

`suite_selection_lock.private.json` 记录三个仓库在生成前解析出的完整 commit SHA；正式
材料中的模型版本以该文件为准，而不是笼统写 `main`。

## 5. 依次生成四套回答

```bash
RUN_LOG=/opt/care-workspace/four_agent_local_run.nohup.log
nohup env \
  BUNDLE_ROOT="$BUNDLE_ROOT" CARE_ROOT="$CARE_ROOT" QZBC_ROOT="$QZBC_ROOT" \
  EVAL_PY="$EVAL_PY" VLLM_BIN="$VLLM_BIN" BACKEND_PY="$BACKEND_PY" \
  PHYSICAL_GPUS="$PHYSICAL_GPUS" CARE_MODEL="$CARE_MODEL" \
  MODEL_ROOT="$MODEL_ROOT" RUN_ROOT="$RUN_ROOT" HF_HUB_DISABLE_XET=1 \
  bash "$SCRIPT" run > "$RUN_LOG" 2>&1 </dev/null &
echo $! > /opt/care-workspace/four_agent_local_run.pid
tail -F "$RUN_LOG"
```

脚本按以下顺序运行并释放同一张 GPU：

1. CARE-OPD vLLM → 隔离 qzbc 后端 → 神农三言完整项目；
2. HuatuoGPT-3-8B 本地模型基线；
3. GLM-4-9B-0414 本地模型基线；
4. InternLM3-8B-Instruct 本地模型基线。

qzbc 使用独立 SQLite，关闭 Graphiti/Mem0 对话写入、会话上下文、LightRAG、QA
知识、LangSmith 与回退；PRIVATE Evidence Control 对 clean 注入一条锁定病例材料，对
证据不可用条件强制零检索，并在逐答中保留审计字段。vLLM 绑定 `127.0.0.1` 且关闭
请求日志。

若 SSH 中断或某一模型 OOM/超时，修复环境后原命令重跑；脚本会验证已有逐题哈希，
跳过已成功题，只按冻结规则重试失败题。不得因答案内容不好看而删除或重新生成。

查看阶段状态：

```bash
find "$RUN_ROOT" -name three_role_execution.private.json -print
tail -n 80 "$RUN_LOG"
nvidia-smi
```

## 6. 完成门

```bash
bash "$SCRIPT" verify
"$EVAL_PY" - "$RUN_ROOT/four_agent_execution.private.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding='utf-8'))
assert x['status'] == 'COMPLETE', x
assert x['task_count_per_system'] == 72, x
assert x['total_response_count'] == 288, x
assert x['ready_for_blind_review_packaging'] is True, x
assert all(s['status'] == 'COMPLETE' and s['response_count'] == 72 for s in x['systems'])
print('FOUR_SYSTEM_RESPONSES_COMPLETE=288')
PY
```

四方任一缺题、失败、空回答、输入哈希不一致或模型选择锁不一致，完成门都会失败。

## 7. 只回传盲评所需 PRIVATE 结果

在服务器创建新的只读回传目录，不带数据库、密钥、日志或模型权重：

```bash
RETURN_STAGE=/opt/care-workspace/four_agent_expert_blind_return
test ! -e "$RETURN_STAGE" || {
  echo "回传目录已存在；请核对旧目录，不要覆盖。" >&2
  exit 2
}
install -d -m 700 "$RETURN_STAGE"
for system in \
  shennong_sanyan_full \
  huatuogpt3_local_medical_agent \
  glm4_local_role_model \
  internlm3_local_role_model; do
  cp -a "$RUN_ROOT/$system" "$RETURN_STAGE/$system"
done
cp -a "$RUN_ROOT/four_agent_execution.private.json" "$RETURN_STAGE/"
cp -a "$MODEL_ROOT/suite_selection_lock.private.json" "$RETURN_STAGE/"
cp -a "$CARE_ROOT/competition_evidence/local_four_agent_server/model_suite.json" "$RETURN_STAGE/"

(cd "$RETURN_STAGE" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "$RETURN_STAGE/SHA256SUMS.private"
tar -czf /opt/care-workspace/FOUR_AGENT_LOCAL_RESULTS.PRIVATE.tar.gz \
  -C /opt/care-workspace four_agent_expert_blind_return
sha256sum /opt/care-workspace/FOUR_AGENT_LOCAL_RESULTS.PRIVATE.tar.gz \
  > /opt/care-workspace/FOUR_AGENT_LOCAL_RESULTS.PRIVATE.tar.gz.sha256
```

回到 Windows PowerShell：

```powershell
Set-Location /opt/care-workspace\care_opd
$ReturnRoot = "runs\private_real_thyroid_external\three_role_eval\returned_local_four_agent"
New-Item -ItemType Directory -Force -Path $ReturnRoot | Out-Null
scp root@gpu-a800-094:/opt/care-workspace/FOUR_AGENT_LOCAL_RESULTS.PRIVATE.tar.gz `
  root@gpu-a800-094:/opt/care-workspace/FOUR_AGENT_LOCAL_RESULTS.PRIVATE.tar.gz.sha256 `
  "$ReturnRoot\"

Set-Location $ReturnRoot
Get-FileHash .\FOUR_AGENT_LOCAL_RESULTS.PRIVATE.tar.gz -Algorithm SHA256
tar -xzf .\FOUR_AGENT_LOCAL_RESULTS.PRIVATE.tar.gz
```

## 8. 生成两位专家各自独立的盲表

以下命令仍在 Windows 本地执行。盲态密钥只存在当前 PowerShell 会话，不写入聊天、
截图或提交包：

```powershell
Set-Location /opt/care-workspace\care_opd
$Py = "..\.hf_upload_venv\Scripts\python.exe"
$Tasks = "runs\private_real_thyroid_external\three_role_eval\paired_fact_n12\three_role_tasks.private.jsonl"
$Returned = "runs\private_real_thyroid_external\three_role_eval\returned_local_four_agent\four_agent_expert_blind_return"
$ReviewPrivate = "runs\private_real_thyroid_external\three_role_eval\expert_preference_local\private"
$ReviewProtocol = "runs\private_real_thyroid_external\three_role_eval\expert_preference_local\public_protocol"

$BlindBytes = New-Object byte[] 48
$BlindRng = [Security.Cryptography.RandomNumberGenerator]::Create()
$BlindRng.GetBytes($BlindBytes)
$env:EXPERT_BLIND_SECRET = [Convert]::ToBase64String($BlindBytes)
$BlindRng.Dispose()

& $Py competition_evidence\prepare_agent_preference_review.py `
  --tasks $Tasks `
  --response "shennong_sanyan_full=$Returned\shennong_sanyan_full\responses.private.jsonl" `
  --response "huatuogpt3_local_medical_agent=$Returned\huatuogpt3_local_medical_agent\responses.private.jsonl" `
  --response "glm4_local_role_model=$Returned\glm4_local_role_model\responses.private.jsonl" `
  --response "internlm3_local_role_model=$Returned\internlm3_local_role_model\responses.private.jsonl" `
  --reviewer expert_01 `
  --reviewer expert_02 `
  --identity-term HuatuoGPT `
  --identity-term GLM-4 `
  --identity-term InternLM `
  --identity-term 神农三言 `
  --identity-term CARE-OPD `
  --blind-secret-env EXPERT_BLIND_SECRET `
  --xlsx `
  --output-private $ReviewPrivate `
  --output-public $ReviewProtocol
```

身份泄露扫描必须为零。若模型在正文自报名称，命令会阻断；不要删题、改医学内容或
只重跑该题，应保留原始回答并为四方统一建立仅遮蔽身份词的展示层后重新生成盲表。

只把 `reviewer_1.private.xlsx` 发给专家 1，把 `reviewer_2.private.xlsx` 发给专家 2。
绝不能发送：

- `blind_mapping.private.jsonl`；
- `review_manifest.private.json`；
- selection lock 中的系统名；
- 另一位专家的表。

专家可选 `A/B/C/D/TIE/NONE/SKIP`；非跳过题必须填写 `unsafe_candidates`，没有不安全
回答时填 `NONE`。被标记不安全的答案不能同时成为最佳；四个都不可接受时选择
`NONE`，不能强迫产生赢家。

## 9. 专家评分完成后的汇总

先保存专家原件并计算 SHA-256，再运行：

```powershell
& $Py competition_evidence\summarize_agent_preference_review.py `
  --manifest "$ReviewPrivate\review_manifest.private.json" `
  --mapping "$ReviewPrivate\blind_mapping.private.jsonl" `
  --ratings "<expert_01完成表路径>" `
  --ratings "<expert_02完成表路径>" `
  --primary-agent shennong_sanyan_full `
  --bootstrap-replicates 10000 `
  --bootstrap-seed 20260825 `
  --output-private "runs\private_real_thyroid_external\three_role_eval\expert_preference_local\results_private" `
  --output-public "runs\private_real_thyroid_external\three_role_eval\expert_preference_local\results_public"
```

主指标是安全合格专家首选份额，95% 区间按真实病例 `family` 整组 bootstrap；同时完整
报告医生、患者、学生、证据可用/不可用、并列、均不可接受、不安全标记和技术失败。
只有公开汇总状态为 `FINAL` 才能进入正文。

## 10. 官方模型与部署依据

- HuatuoGPT-3-8B：<https://huggingface.co/FreedomIntelligence/HuatuoGPT-3-8B>
- GLM-4-9B-0414：<https://huggingface.co/zai-org/GLM-4-9B-0414>
- InternLM3-8B-Instruct：<https://huggingface.co/internlm/internlm3-8b-instruct>
- vLLM 支持模型：<https://docs.vllm.ai/en/latest/models/supported_models/>
- vLLM 在线服务与生成配置：<https://docs.vllm.ai/en/latest/serving/online_serving/>

模型页和政策可能变化；执行时的完整 commit SHA、模型卡元数据、下载审计与响应哈希才是
本轮正式证据。
