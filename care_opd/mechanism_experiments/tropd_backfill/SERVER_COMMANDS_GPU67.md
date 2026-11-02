# 物理卡 6、7 并行补测 TROPD

这只是执行调度调整，不改变数据、生成参数、模型、评分和统计规则。
不是把同一个 4B 模型切到两张卡上。

| 物理卡 | 顺序执行内容 | 新输出根 |
| --- | --- | --- |
| 6 | QZBC，TROPD 然后 CARE | `.../gpu6_qzbc/` |
| 7 | Structured 的 TROPD/CARE，然后 Raw 的 TROPD/CARE | `.../gpu7_real/` |

两个 lane 并行、同一 lane 内串行；每个模型进程只看到一张逻辑卡 `cuda:0`。
同协议两模型仍在同一物理卡上评测。QZBC 工作量较大，卡 7 提前完成是正常现象，
不能承诺全程均衡或两倍加速。总工作量仍为 9,352 次请求，无重训。
脚本不检查 GPU 是否已有计算进程，不清卡、不终止其他任务；共享显存不足仍可能 OOM。

## 1. 部署本次 GPU67 调度脚本

将 `CARE_OPD_TMM_VISUAL_TROPD_GPU67_V2.tar.gz` 和同名 `.sha256` 上传到 `/opt/care-workspace`。
如果此前单卡任务已经启动，先让其完成或由操作者处理；本脚本不会自动迁移或终止它。
本包包含原单卡评测器与新增调度器，不覆盖旧结果，但会更新同路径的脚本文件，
所以不要在旧任务运行中解压。

```bash
(
  set -e
  cd /opt/care-workspace
  sha256sum -c CARE_OPD_TMM_VISUAL_TROPD_GPU67_V2.tar.gz.sha256
  test -d /opt/care-workspace/care_opd
  tar -xzf CARE_OPD_TMM_VISUAL_TROPD_GPU67_V2.tar.gz -C /opt/care-workspace/care_opd
)
```

## 2. 路径与启动

模型位置不同就修改 `TROPD_MODEL` 或 `CARE_MODEL`。须使用合并后的完整 HF 权重。
新调度器固定物理卡 6、7，无需设置 `PHYSICAL_GPU=6,7`（旧脚本不接受这个值）。

```bash
export CARE_ROOT=/opt/care-workspace/care_opd
export QZBC_ROOT=/opt/care-workspace/qzbc_source/qzbc-main
export PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export TROPD_MODEL="$CARE_ROOT/runs/qzbc_baselines_3gpu/baselines/tropd_full/final_model"
export CARE_MODEL="$CARE_ROOT/runs/qzbc_base_model_3gpu/final_model"
export PARALLEL_ROOT=/opt/care-data/care_opd_tropd_backfill_gpu67
export REAL_RUN_MODE=fresh
unset DRY_RUN

RUNNER="$CARE_ROOT/mechanism_experiments/tropd_backfill/run_tropd_backfill_gpu67.sh"
RUN_LOG="$PARALLEL_ROOT/driver.nohup.log"

if bash "$RUNNER" preflight; then
  nohup bash "$RUNNER" all >>"$RUN_LOG" 2>&1 </dev/null &
  run_pid=$!
  printf '%s\n' "$run_pid" >"$PARALLEL_ROOT/driver.pid"
  disown "$run_pid" 2>/dev/null || true
  echo "PID=$run_pid LOG=$RUN_LOG"
else
  echo '预检失败，未启动；终端保持打开，请查看上方错误。'
fi
```

`all` 会再次预检后才启动两个 lane。预检会核对软件和模型架构/数据及哈希，不是检查卡是否空闲。
调度器清除继承的 `QZBC_OUT_ROOT` / `STRUCTURED_OUT` / `RAW_OUT` 等输出路径覆盖，
防止此前终端变量把两个 lane 写到同一个目录；数据路径覆盖仍沿用原单卡脚本定义。

## 3. 看日志

```bash
tail -F /opt/care-data/care_opd_tropd_backfill_gpu67/driver.nohup.log
```

两个卡的详细日志（另开终端，Ctrl+C 只结束 tail）：

```bash
tail -F /opt/care-data/care_opd_tropd_backfill_gpu67/gpu6_qzbc.log /opt/care-data/care_opd_tropd_backfill_gpu67/gpu7_real.log
```

主日志只有 `GPU67_RESULT gpu6_rc=0 gpu7_rc=0` 之后才输出 `GPU67_COMPLETE`。
一条 lane 失败时，另一条不会被杀掉，已完成结果保留，但总任务不能视作全部完成。

## 4. 汇总表位置

```text
/opt/care-data/care_opd_tropd_backfill_gpu67/gpu6_qzbc/qzbc_grouped_tropd_vs_care_v1/formal/suite_summary.csv
/opt/care-data/care_opd_tropd_backfill_gpu67/gpu7_real/real_thyroid_v5_structured/tropd_vs_care_v1/public/model_metrics.csv
/opt/care-data/care_opd_tropd_backfill_gpu67/gpu7_real/real_thyroid_v5_raw/tropd_vs_care_v1/public/model_metrics.csv
```

两条 lane 使用独立锁和模型清单。旧 `/opt/care-data/care_opd_tropd_backfill/` 不被复用或覆盖。
完整协议和恢复限制仍见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)：已完成输出核验后跳过；
QZBC partial 不支持逐行恢复，须保留旧目录并选新输出位置；真实轨 partial 可在同合同下
用原单卡脚本 `RUN_MODE=resume` 对相应轨恢复，`BACKFILL_ROOT` 指向本次 `gpu7_real`。
不要对尚未开始的真实轨使用 resume。

新增视觉迁移实验不由此调度器自动启动，仍用 `visual_evidence_transfer/SERVER_COMMANDS.md`。
要复用卡 6、7，应先确认对应 lane 已完成，避免同时加载额外模型。

## 5. 仅查看计划，不推理

```bash
DRY_RUN=1 bash /opt/care-workspace/care_opd/mechanism_experiments/tropd_backfill/run_tropd_backfill_gpu67.sh all
```
