# 服务器运行

## 1. 安装

```bash
cd /opt/care-workspace/care_opd
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev,quant]'
```

训练使用 HF checkpoint。Ollama 的 `qwen3.5:4b` 只用于部署与 A/B，
不能直接作为训练权重。

## 2. GPU 1 后台正式训练

```bash
cd /opt/care-workspace/care_opd
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
ROLLOUT_NPROC=3 \
ROLLOUT_DEVICES=0,1,2 \
QZBC_ROOT=/opt/care-workspace/qzbc_source/qzbc-main \
STUDENT_MODEL=Qwen/Qwen3.5-4B \
TEACHER_MODEL=Qwen/Qwen3.5-9B \
bash scripts/launch_nohup.sh
```

该命令会先从 qzbc 当前仓库生成正式数据。它不读取患者 SQLite、
Graphiti 或 Mem0；知识资产的来源、许可元数据及 train/dev/test
条目隔离结果写入 seed manifest。详见 `docs/DATA.md`。

监控：

```bash
tail -f /opt/care-workspace/care_opd/runs/qzbc_base_model/train.nohup.log
cat /opt/care-workspace/care_opd/runs/qzbc_base_model/train.pid
nvidia-smi -i 0,1,2
```

完成标志：

```bash
cat /opt/care-workspace/care_opd/runs/qzbc_base_model/FINAL_MODEL.txt
test -f /opt/care-workspace/care_opd/runs/qzbc_base_model/final_model/config.json
test -f /opt/care-workspace/care_opd/runs/qzbc_base_model/final_model/export_manifest.json
```

## 3. 分阶段运行

```bash
bash scripts/00_validate.sh
bash scripts/01_build_formal_data.sh
bash scripts/03_train_care_opd.sh
```

相关 baseline 单独运行：

```bash
bash scripts/02_train_baselines.sh
```

默认规模：

```bash
QZBC_SEED_TOTAL=12000
ROLLOUTS_PER_ROUND=12000
ONLINE_ROUNDS=3
NPROC_PER_NODE=1
ROLLOUT_NPROC=1
```

小规模流程检查可临时设置：

```bash
QZBC_SEED_TOTAL=80 \
ROLLOUTS_PER_ROUND=56 \
ONLINE_ROUNDS=1 \
bash scripts/run_all_formal.sh
```

这只是流程检查，不能作为比赛结果。

## 4. 断点与显存

阶段产物完整时脚本会复用；训练过程中断时，正式脚本已经传入：

```bash
--resume_from_checkpoint auto
```

`auto` 表示有 checkpoint 就恢复，没有就从该阶段起点开始。

单张 A800 80GB 默认使用 BF16 LoRA，并以 768 个 response token 的
selected logits 单块计算，避免把同一个 4B/9B 前向重复几十次。若显存
仍不足，可在
`configs/qzbc_formal.yaml` 中启用 4-bit QLoRA，并降低
`response_chunk_size`、`max_length`、`max_new_tokens` 或每轮采样数；
不要通过换成 tokenizer 不一致的教师而
破坏教师/学生 tokenizer 一致性。

## 5. Ollama 注册

```bash
LLAMA_CPP_DIR=/opt/care-workspace/llama.cpp \
QZBC_GGUF_OUTTYPE=q8_0 \
bash scripts/05_register_final_ollama.sh
```

若要做严格量化公平的主表，应使用 `qzbc_model_ab.sh prepare` 将 base
HF 与 trained HF 用完全相同 outtype 转换；直接拿已有
`qwen3.5:4b` 与 q8_0 trained 模型比较只适合工程检查。

## 6. qzbc 替换

保留原后端配置，只改：

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5-4b-care-opd:latest
```

评测时另设：

```dotenv
LLM_ALLOW_FALLBACK=false
```

修改后重启 qzbc。因为 qzbc 也可能把 `OLLAMA_MODEL` 用于 Graphiti、
Mem0 和候选排序，需同时检查这些适配器 health。

## 7. 可选：完整 Agent 双后端 A/B

先确保 base 和 trained 两个 Ollama tag 都存在，然后运行：

```bash
cd /opt/care-workspace/care_opd
source .venv/bin/activate

QZBC_ROOT=/opt/care-workspace/qzbc_source/qzbc-main \
BASE_OLLAMA_MODEL=qwen3.5:4b \
TRAINED_OLLAMA_MODEL=qwen3.5-4b-care-opd:latest \
bash scripts/07_launch_qzbc_agent_ab_backends.sh
```

该脚本启动端口 `8001/8002` 上的两个 qzbc 进程，使用相同配置但隔离
SQLite、Graphiti 和 Mem0 写状态。随后：

```bash
QZBC_BASE_AGENT_URL=http://127.0.0.1:8001 \
QZBC_TRAINED_AGENT_URL=http://127.0.0.1:8002 \
QZBC_PASSWORD=password123 \
SKIP_QZBC_LLM_EXPERIMENT_AB=1 \
bash scripts/04_evaluate_formal.sh
```

PID、日志和隔离数据库位于
`runs/qzbc_base_model/agent_ab_runtime_<timestamp>/`。此步骤是部署路径
检查，不是训练前置条件。
