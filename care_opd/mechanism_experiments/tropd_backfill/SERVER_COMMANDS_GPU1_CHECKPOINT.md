# GPU1 QZBC 重启（逐条保存，后续可续跑）

旧 GPU6 中断时只有进程锁，前3537条未持久化，无法恢复。本次经用户确认重新执行
TROP-D 4200条与 CARE-OPD 4200条，共8400条；正式统计仍为同一3900条 family-clean视图。
不重训、不下载、不重做真实外测、不修改原GPU0多模态结果。

## 部署

上传 `CARE_OPD_QZBC_GPU1_CHECKPOINT_V1.tar.gz` 和同名 `.sha256` 到 `/opt/care-workspace`。
部署仅更新 QZBC 收集器、backfill包装脚本和协议守卫，并新增GPU1入口与说明/测试。
不要在另一个 QZBC 收集器仍运行时部署。以下备份三个将被替换的文件，不删除旧结果。

```bash
(
set -e
cd /opt/care-workspace
sha256sum -c CARE_OPD_QZBC_GPU1_CHECKPOINT_V1.tar.gz.sha256
if pgrep -af '[c]are_opd.qzbc_rollout'; then
  echo '仍有QZBC进程，先核对；未更新脚本，也不会自动结束进程。' >&2
  exit 2
fi
CARE_ROOT=/opt/care-workspace/care_opd
BACKUP=/opt/care-workspace/qzbc_code_before_checkpoint_$(date +%Y%m%d_%H%M%S)_$$.tar.gz
tar -czf "$BACKUP" -C "$CARE_ROOT" \
  src/care_opd/qzbc_rollout.py \
  mechanism_experiments/tropd_backfill/run_tropd_backfill.sh \
  mechanism_experiments/tropd_backfill/protocol_guard.py
tar -xzf CARE_OPD_QZBC_GPU1_CHECKPOINT_V1.tar.gz -C "$CARE_ROOT"
echo "CODE_BACKUP=$BACKUP"
bash "$CARE_ROOT/mechanism_experiments/tropd_backfill/run_qzbc_gpu1_checkpoint.sh" start
)
```

所有失败退出都在子shell中，不会关闭交互终端。没有GPU占用检查、清卡或kill。
进程重入锁、代码/模型/数据/软件环境校验仍保留。共享主机内存不足等外部问题不由换卡解决。

## 看进度

```bash
bash /opt/care-workspace/care_opd/mechanism_experiments/tropd_backfill/run_qzbc_gpu1_checkpoint.sh status
```

持续跟踪（Ctrl+C只结束tail）：

```bash
bash /opt/care-workspace/care_opd/mechanism_experiments/tropd_backfill/run_qzbc_gpu1_checkpoint.sh follow
```

## 再次中断后的续跑

不重新解压、不改配置、不换目录，重复以下命令即可。旧checkpoint和代码不符时会拒绝运行，
不会通过覆盖旧锁来伪造续跑。若PID已被其他进程复用，入口会停下供人工核对，不自动kill。

```bash
bash /opt/care-workspace/care_opd/mechanism_experiments/tropd_backfill/run_qzbc_gpu1_checkpoint.sh start
```

新根固定为 `/opt/care-data/care_opd_tropd_backfill_gpu1_checkpoint_v1`。
其中 `qzbc_grouped_tropd_vs_care_v1/formal/{tropd_full,care_opd}/` 每条生成后保存
`checkpoint_candidates.jsonl` 并fsync，另有 `checkpoint_manifest.json` 与
`checkpoint_progress.json` 绑定协议与进度。拒答、错误格式和失败答案也保留，不按内容重试。
损坏或截断的checkpoint会中止并保留现场，不擅自截尾后重生成。

一个模型完整发布并通过验证后才生成 `COMPLETE.json`，再次启动会验证后跳过该模型。
最终汇总：

```text
/opt/care-data/care_opd_tropd_backfill_gpu1_checkpoint_v1/qzbc_grouped_tropd_vs_care_v1/formal/suite_summary.csv
```

这是新一次执行，不是恢复GPU6已丢失的3537条；旧GPU6目录与已完成GPU7真实外测原样保留。
协议锁披露新GPU1、checkpoint策略和代码身份，数据/权重/生成参数/评分规则不因恢复机制改变。
