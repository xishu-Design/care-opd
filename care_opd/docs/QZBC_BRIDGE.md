# qzbc 底座替换与同场景对照

该工具不修改 qzbc。它读取 qzbc 的真实 provider、环境变量和模型实验路由，生成独立部署目录，再让同一个 qzbc `LLMExperimentRunner` 依次运行 base 与 trained 模型。该路径是 LLM provider/model 集成对照，不是完整 Agent Graph A/B。

qzbc 当前契约为：

- Ollama：`LLM_PROVIDER=ollama`、`OLLAMA_BASE_URL`、`OLLAMA_MODEL`；
- OpenAI-compatible：`OPENAI_COMPATIBLE_BASE_URL`、`OPENAI_COMPATIBLE_MODEL`；
- 模型对照入口：`POST /api/llm/experiments/run`；
- `full` 是 12 个固定场景，覆盖医生审方、患者科普、学生解析和练习题生成；
- 对照时必须设置 `LLM_ALLOW_FALLBACK=false`，否则 provider 失败会静默落到模板。

## 1. 生成部署包

```bash
bash scripts/qzbc_model_ab.sh prepare \
  --base-checkpoint /root/models/Qwen3.5-4B \
  --trained-checkpoint /opt/care-workspace/care_opd/runs/qzbc_base_model/final_model \
  --qzbc-root /opt/care-workspace/qzbc_source/qzbc-main \
  --output-dir runs/qzbc_ab
```

工具会验证两者均为完整的 merged Hugging Face checkpoint、架构一致，并记录 qzbc 源码契约指纹。它只向 `runs/qzbc_ab` 写文件。

## 2. 注册到 Ollama

Ollama 官方目前没有把 Qwen 列入直接导入 Safetensors 的保证列表，因此工具不假定 `FROM <HF目录>` 可用，而是生成 HF → GGUF → Ollama 的显式流程：

```bash
LLAMA_CPP_DIR=/root/llama.cpp \
  bash runs/qzbc_ab/ollama/convert_hf_to_gguf.sh

bash runs/qzbc_ab/ollama/register_models.sh
```

注册脚本最后直接检查 Ollama `/api/tags` 中是否存在两个完整模型名，不使用 qzbc 内硬编码 `qwen2.5:7b` 的旧检查脚本。相关格式见 [Ollama 模型导入文档](https://docs.ollama.com/import)。

## 3. 启动 qzbc

```bash
cd /root/qzbc-main
set -a
source /root/care_opd/runs/qzbc_ab/qzbc.ollama.env
set +a
bash scripts/dev_backend.sh
```

qzbc 会缓存 settings，因此更改 `.env` 后要重启后端。A/B 运行本身不需要重启：qzbc 的实验路由会在同一 provider 和同一场景上临时覆盖模型名。

LightRAG 服务有独立的 `LLM_MODEL` 配置；本工具不改变它，因此比较只归因于 qzbc 后端生成模型。

## 4. 同场景 A/B

使用演示医生账号登录：

```bash
QZBC_PASSWORD=password123 \
  bash runs/qzbc_ab/compare_ollama.sh
```

或提前提供 token：

```bash
QZBC_TOKEN=... bash runs/qzbc_ab/compare_ollama.sh
```

输出：

- `comparison.json`：两个原始 qzbc report、逐 case 配对结果、指标差值和 exact McNemar 检验；
- `comparison.cases.csv`：逐场景 base/trained 成败、fallback 与延迟；
- 任一模型出现 fallback 时默认立即失败，不产生可误解的正式对照结果。

这 12 个场景是 qzbc 自带的模型/provider 集成测试，不等同于真实临床
验证，也不能替代独立 test split 的 qzbc-native 配对评测；它们不执行
Agent Graph、工具、检索或记忆链路。

若要检查完整 Agent 部署路径，请先运行
`scripts/07_launch_qzbc_agent_ab_backends.sh`，再用
`care_opd.qzbc_agent_ab` 对端口 `8001/8002` 做双后端配对。不能在同一
qzbc 进程中做这项检查，因为 qzbc 会缓存启动时的 `OLLAMA_MODEL`。

## Heldout 原生边界对照

若已有 `qzbc_dataset` 产生的 heldout
`qzbc_on_policy_requests.jsonl`，可绕过 qzbc 的固定 12 例，直接在两模型上
复用完全相同的 qzbc `student_messages -> GeneratedResponse` 请求：

```bash
python -m care_opd.qzbc_native_eval \
  --requests data/qzbc_formal/compiled/test/qzbc_on_policy_requests.jsonl \
  --base-model qwen3.5:4b \
  --trained-model qwen3.5-4b-care-opd:latest \
  --output runs/qzbc_ab/native-heldout.json
```

该命令使用 qzbc Ollama provider 相同的结构化输出 schema 和默认生成参数，
逐例调用 `assess_response`，输出 pass-rate 差、失败类型、paired
wins/losses/ties、exact McNemar p 值，以及 `.cases.csv`。它不启动或替换
qzbc Agent，也不修改 qzbc。

## 可选 HF serving

`openai_compatible/` 同时生成两个 `vllm serve` 脚本和对应 qzbc 环境文件。qzbc 的 OpenAI-compatible provider 需要 URL 以 `/v1` 结尾；启动参数依据 [vLLM serve 文档](https://docs.vllm.ai/en/latest/cli/serve/)。
