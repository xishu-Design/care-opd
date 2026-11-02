# qzbc CARE-OPD 基础模型后训练

本目录训练的不是一个新 Agent，而是一个可直接替换 qzbc 底座的
`Qwen3.5-4B`。

部署时保持 `qzbc_source/qzbc-main` 的 LangGraph、MCP 工具、LightRAG、
Graphiti、Mem0、规则引擎和后处理全部不变，只替换：

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5-4b-care-opd:latest
```

最终产物是合并后的独立 Hugging Face checkpoint；推理时不依赖
CARE-OPD、PEFT、训练 verifier 或额外 Agent。

> 数据由 qzbc 已审校、带来源记录的药材/方剂/配伍/饮食/教学资产与
> 无真实患者信息的受控上下文组成；不会读取患者数据库、Graphiti 或
> Mem0。它只用于比赛研究与工程验证，不构成诊疗、处方或用药建议。

## 训练边界

qzbc 的工具路由由宿主程序确定，基础模型不选择工具。真实边界是：

```text
原 qzbc Graph/工具
        │
        ▼
qzbc 原生 system + 多轮消息 + Evidence Packet
        │
        ▼
Qwen3.5-4B
        │
        ▼
GeneratedResponse JSON
  title / summary / sections(source_ids)
  warnings / next_actions / disclaimer
        │
        ▼
原 qzbc Safety Guard、规则覆盖与业务响应
```

因此 OPD 学习的是底座在以下输入上的生成恢复能力：

- 错误但标为可信的历史回答；
- Evidence Packet 的证据源不可用；
- Evidence Packet 内证据冲突；
- 引用不存在的 `source_id`；
- 角色越权、无依据确定性结论或结构化输出失败。

训练时，当前学生在 qzbc 原生消息上在线生成；宿主 verifier 定位失败所
影响的 `summary/sections/source_ids/warnings/next_actions` 字段；教师只在
这些字段对应 token 上提供稠密分布，并按风险加权，同时保持其余正确
内容。每轮更新后重新采样：

```text
student rollout → host verification → localized OPD → merge → recollect
```

## 目录中真正需要的入口

- `qzbc_seed_builder.py`：覆盖 qzbc 8 种原生任务的 train/dev/test seed。
- `qzbc_assets.py`：按可见性、复核状态和训练许可加载 qzbc 资产，并按
  完整知识条目隔离 train/dev/test。
- `qzbc_dataset.py`：读取 qzbc 当前源码中的真实 prompt，编译 clean 与
  三类恢复样本。
- `qzbc_rollout.py`：从当前基础模型在线采样，产生 OPD 和 DPO 数据。
- `train_opd.py`：同 tokenizer 的逐 token OPD、因果字段 mask、风险权重、
  教师信任 mask 和 clean retention。
- `export_model.py`：合并 LoRA，导出无训练时依赖的完整模型。
- `qzbc_native_eval.py`：在同一 held-out qzbc 原生请求上配对比较模型。
- `qzbc_bridge.py`：调用 qzbc 内置的 LLM provider/model 实验接口。
- `qzbc_agent_ab.py`：经两个未修改 qzbc 后端的真实 Agent Graph 做部署
  A/B，并核验实际模型名和 fallback。

旧的 `CareRecoveryEnv` 代码只保留为早期机制探针和回归测试，不进入
本流程的数据、模型输入或部署。

## 服务器正式训练

假设目录为：

```text
/opt/care-workspace/care_opd
/opt/care-workspace/qzbc_source/qzbc-main
```

安装：

```bash
cd /opt/care-workspace/care_opd
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev,quant]'
```

正式训练使用 Hugging Face 权重，而不是 Ollama 中的 GGUF。若不提前
下载，Transformers 会在首次运行时自动下载：

```bash
export STUDENT_MODEL=Qwen/Qwen3.5-4B
export TEACHER_MODEL=Qwen/Qwen3.5-9B
```

在 GPU 1 后台运行完整的数据生成、clean warm-start 和三轮 OPD：

```bash
cd /opt/care-workspace/care_opd
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
ROLLOUT_NPROC=3 \
ROLLOUT_DEVICES=0,1,2 \
QZBC_ROOT=/opt/care-workspace/qzbc_source/qzbc-main \
bash scripts/launch_nohup.sh
```

查看状态：

```bash
tail -f /opt/care-workspace/care_opd/runs/qzbc_base_model/train.nohup.log
cat /opt/care-workspace/care_opd/runs/qzbc_base_model/FINAL_MODEL.txt
```

后台日志会持续输出单行进度，不依赖交互式终端：

```text
[progress] stage=sft:clean_sft status=accumulating step=3/138 pct=2.17% batch=121/4400 accum=25/32 loss=...
[progress] stage=sft:clean_sft status=running step=4/138 pct=2.90% elapsed=... eta=... loss=... lr=...
[progress] stage=rollout status=running step=317/12000 pct=2.64% elapsed=... eta=... passed=... failed=...
```

训练阶段在每个 microbatch 后报告当前位置，每个 optimizer step 后更新
百分比和 ETA；三卡训练只由全局 rank 0 输出。`metrics.jsonl` 仍作为
机器可读的低频指标记录保留。

默认正式规模为 12,000 个 clean seed、四种输入状态和三轮各 12,000 次
当前策略采样。可以用环境变量调整：

```bash
QZBC_SEED_TOTAL=12000
ROLLOUTS_PER_ROUND=12000
ONLINE_ROUNDS=3
NPROC_PER_NODE=1
ROLLOUT_NPROC=1
```

断点后重跑同一命令即可：已有完整阶段会复用；未完成的训练阶段默认以
`--resume_from_checkpoint auto` 恢复最新 checkpoint。

seed manifest 会保留 qzbc 资产哈希、`source_id/source_record_id`、许可
元数据和知识条目切分检查。正式数据说明及公开权重前的许可复核要求见
`docs/DATA.md`。

## 最相关 baseline

运行 baseline 是可选的，不会阻塞正式模型训练：

```bash
CUDA_VISIBLE_DEVICES=1 \
QZBC_ROOT=/opt/care-workspace/qzbc_source/qzbc-main \
bash scripts/02_train_baselines.sh
bash scripts/06_evaluate_hf_baselines.sh
```

脚本在完全相同的 qzbc 数据、学生初始化和采样预算上训练：

1. Frozen `Qwen3.5-4B`
2. qzbc clean SFT
3. offline Recovery-SFT
4. Recovery-DPO
5. full-response GKD/OPD
6. trust-region full-response OPD
7. 本方法：字段局部化、风险加权、带 retention 的多轮 OPD

不能把相关论文中不同模型和数据集上的数字直接抄作本项目 baseline；
比赛表格应使用这些脚本在同一 qzbc 测试集上重跑的结果。具体对应关系
见 `docs/BASELINES.md`。

## 注册并替换 qzbc 底座

训练完成后，用当前 `llama.cpp` 转 GGUF 并注册：

```bash
cd /opt/care-workspace/care_opd
source .venv/bin/activate

LLAMA_CPP_DIR=/opt/care-workspace/llama.cpp \
bash scripts/05_register_final_ollama.sh
```

然后在 qzbc 原有 `.env` 中保留你的 embedding、LightRAG、Graphiti 和
Mem0 设置，只改生成模型：

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5-4b-care-opd:latest
LLM_ALLOW_FALLBACK=false

OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
KNOWLEDGE_BACKEND=lightrag
LIGHTRAG_ENABLED=true
LIGHTRAG_MODE=api
LIGHTRAG_API_BASE_URL=http://127.0.0.1:9621
GRAPH_BACKEND=graphiti
MEMORY_BACKEND=mem0
GRAPHITI_DB_PATH=runtime/graphiti/guardtcm.db
MEM0_QDRANT_PATH=runtime/mem0/qdrant
```

`OLLAMA_MODEL` 也被 qzbc 的部分 Graphiti/Mem0 适配器和候选排序复用，
因此切换后必须重启 qzbc 后端，并检查这些适配器的 health。LightRAG 的
生成模型是独立配置，不会被本脚本改动。

## 模型主评测与完整 Agent 部署检查

先确保 Ollama 中同时存在：

```text
qwen3.5:4b
qwen3.5-4b-care-opd:latest
```

主评测使用独立 test split 和 qzbc 完全相同的消息/JSON schema。qzbc
自带的 12 场景接口直接调用 `LLMExperimentRunner/LLMService`，可作为
provider/model 集成对照，但不经过完整 Agent Graph：

```bash
QZBC_PASSWORD=password123 \
QZBC_URL=http://127.0.0.1:8000 \
bash scripts/04_evaluate_formal.sh
```

输出位于：

```text
runs/qzbc_base_model/evaluation/native_heldout.json
runs/qzbc_base_model/evaluation/qzbc_llm_experiment_ab.json
```

若还要验证“同一 Agent、只换基础模型”，必须启动两个 qzbc 进程，因为
qzbc 会在进程启动时缓存 `OLLAMA_MODEL`。脚本会使用相同源码和等价的
演示数据，同时隔离 SQLite、Graphiti 与 Mem0 的可写状态：

```bash
bash scripts/07_launch_qzbc_agent_ab_backends.sh

QZBC_BASE_AGENT_URL=http://127.0.0.1:8001 \
QZBC_TRAINED_AGENT_URL=http://127.0.0.1:8002 \
QZBC_PASSWORD=password123 \
SKIP_QZBC_LLM_EXPERIMENT_AB=1 \
bash scripts/04_evaluate_formal.sh
```

完整 Agent 输出为
`runs/qzbc_base_model/evaluation/qzbc_full_agent_ab.json`。它会强制
检查 `RoleAIChatGraph/PatientEducationGraph` 的节点路径、
`llm_result.model`、schema 以及 LLM 自身的 degraded/fallback；工具或
空检索库造成的 Agent 降级单独报告，不会伪装成模型加载失败。它仍只是
部署集成检查；方法效果以 held-out 主评测为准，不能表述为临床有效性。
