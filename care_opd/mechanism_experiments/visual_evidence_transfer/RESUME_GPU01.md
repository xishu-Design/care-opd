# 原结果保留：多模态 GPU0 续跑，QZBC GPU1 续跑条件核查

本入口只新增恢复调度，不修改冻结推理器、提示词、评分器或原始 manifest。
用户要求 QZBC 也是续跑，不授权全量重算。不要执行新根全量重跑命令。

## 多模态 GPU0

把 `resume_visual_on_gpu.py` 上传为 `/opt/care-workspace/resume_visual_on_gpu.py`。
脚本必须放在原冻结代码目录之外，通过 PYTHONPATH 调用原 V4 模块。

```bash
(
set -e
SRC=/opt/care-workspace/care_opd_visual_v4/mechanism_experiments/visual_evidence_transfer
ENTRY=/opt/care-workspace/resume_visual_on_gpu.py
PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
WORK=/opt/care-data/care_opd_tmm_visual_gpu45_v4
CONFIG="$WORK/config_validation10.json"
test -s "$ENTRY"
test -s "$CONFIG"
test -x "$PY"

env PYTHONPATH="$SRC" "$PY" -B "$ENTRY" preflight \
  --source-dir "$SRC" --config "$CONFIG" --gpu 0

mkdir -p "$WORK/logs"
LOG="$WORK/logs/resume_care_gpu0_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTHONPATH="$SRC" "$PY" -B -u "$ENTRY" resume \
  --source-dir "$SRC" --config "$CONFIG" --gpu 0 \
  >"$LOG" 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" >"${LOG%.log}.pid"
disown "$run_pid" 2>/dev/null || true
printf 'PID=%s\nLOG=%s\n' "$run_pid" "$LOG"
sleep 3
ps -fp "$run_pid" || true
tail -n 80 "$LOG"
)
```

预期 `care_saved=5 care_missing=21`，以实际校验值为准。已完成四个对照臂和共享观察必须完整。
可重复执行，完整行按原协议验证后跳过，包括空答、拒答或截断，不按回答好坏筛选重试。
不检查其他 GPU 进程、不清卡、不杀进程；仅保留本实验独占写锁与环境一致性检查。
若 `.execution.lock` 残留，先核对其 PID，不自动删除。换卡无法保证解除主机内存或 cgroup 限制。

输出仍在 `.../runs_validation10/arms/care_opd/responses.jsonl`，汇总仍在 `.../runs_validation10/summary/`。
额外审计 `.../runs_validation10/runtime/device_migrations.jsonl` 记录原 GPU5、实际 GPU0、
环境、代码指纹、保留输出哈希、待补 task IDs、开始/完成状态。原始 manifest 不改写。
论文/报告应披露迁卡，不能把混合设备时延当成同一物理卡上的严格对照。

## QZBC GPU1：先检查持久化结果，不能从日志计数假装恢复

当前本地 `qzbc_rollout.py` 的单 worker 在内存累计 candidates，完成所有请求后才发布三个 JSONL。
它没有逐条 checkpoint；`RUN_MODE=resume` 在 wrapper 中只适用于真实外测，不适用于 QZBC。
以下仅列文件和行数，不读取或展示请求/临床内容、不启动推理、不修改旧目录。

```bash
ROOT=/opt/care-data/care_opd_tropd_backfill_gpu67/gpu6_qzbc/qzbc_grouped_tropd_vs_care_v1
for model in tropd_full care_opd; do
  printf '\nMODEL=%s\n' "$model"
  OUT="$ROOT/formal/$model"
  if test -d "$OUT"; then
    find "$OUT" -maxdepth 3 -type f -printf '%P\t%s bytes\n'
    find "$OUT" -maxdepth 3 -type f \
      \( -name '*.jsonl' -o -name '*.tmp' \) -exec wc -l {} +
  else
    echo MODEL_DIRECTORY=ABSENT
  fi
done
grep -nE 'def _generate_candidates|candidates.append|_atomic_jsonl|resume|checkpoint' \
  /opt/care-workspace/care_opd/src/care_opd/qzbc_rollout.py
```

若没有已持久化候选，死进程内的3537条无法恢复；仅日志的 passed/failed 数不能重建模型响应。
已有 COMPLETE.json 的模型可用原协议锁核验并保留，不需重新生成。
旧锁还包含 GPU6，不能直接改 PHYSICAL_GPU=1 冒用原锁。
任何无法恢复而需重算的部分，应明确向用户说明并取得同意，不自动转换成全量重跑。

## 本地验证范围

本次仅运行 CPU 合成测试和源码核查，未连接服务器、未运行真实 GPU 推理。
