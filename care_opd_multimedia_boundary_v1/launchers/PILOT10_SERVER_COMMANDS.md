# GPU1：明确改为 10 组开发性预实验

用户已确认将原定 24 组改为 10 组。原因是固定 ChartQA validation 数据在当前筛查、精确问题配对及三图不复用规则下，原确定性搜索只找到 10 组；这不是最大可行组数的精确证明。变更发生在本轮模型输出生成前，不按模型表现选样。

保留 seed=20260917、六种条件、四个模型、两个确定性基线、共享视觉前端及首输出冻结规则。实际配对要求问题原文在 strip() 后完全相同；问题不做语义改写或模糊匹配。10 组 × 6 条件 = 每臂 60 条任务，四模型共 240 条输出，两个确定性基线另计 120 条。不把 60 条相关请求当作 60 个独立样本。

不改原核心包，不重新下载数据，不清理任何已有文件。先显式准备 10 组，核验后再调用已有 root helper。原 runner 遇到有效 selection_manifest.json 会复用，因此不会重新执行其默认的 --blocks 24。后续报告以清单的 10/60 为准，不沿用原 README 中的 24/144。

## 准备并后台启动

直接在服务器 Bash 中执行整段。括号内失败只退出子 shell，不关闭当前交互终端。需要此前已上传的 root helper 及其校验文件。

```bash
(
set -euo pipefail
umask 077

PKG=/opt/care-workspace/care_opd_multimedia_boundary_v1
WORK=/opt/care-workspace/care_opd_multimedia_boundary_gpu1_v1_diskfix
DATA_ROOT=/opt/care-workspace/care_opd_multimedia_boundary_chartqa_val_v1
PY=/opt/care-workspace/venvs/care_boundary_data/bin/python
CONTROL=/opt/care-workspace/care_opd_boundary_gpu1_control

cd /opt/care-workspace
sha256sum -c run_boundary_root_gpu1.sh.sha256
test -x "$PY"
test -s "$DATA_ROOT/download_manifest.json"
for target in "$WORK" "$DATA_ROOT" "$CONTROL"; do
  case "$(readlink -m -- "$target")" in
    /data|/data/*) echo "路径指向已满的 /data，停止：$target" >&2; exit 2 ;;
  esac
done
available=$(df -Pk /opt/care-workspace | awk 'NR==2 {print $4}')
free_inodes=$(df -Pi /opt/care-workspace | awk 'NR==2 {print $4}')
if (( available < 8*1024*1024 || free_inodes < 10000 )); then
  echo '根分区空间不足，未启动；不会自动删除文件。' >&2
  exit 2
fi

mkdir -p "$CONTROL" "$CONTROL/cache/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$CONTROL/cache/tmp" TMP="$CONTROL/cache/tmp" TEMP="$CONTROL/cache/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PKG"

exec 9>"$CONTROL/launcher.lock"
flock -n 9 || { echo '本实验已有启动器运行，未重复启动。' >&2; exit 2; }
PREP_LOG="$CONTROL/prepare_pilot10_$(date +%Y%m%d_%H%M%S).log"
(
  printf '%s\n' \
    'PROTOCOL_CHANGE=24_to_10_blocks_user_approved_before_generation' \
    'EVALUATION_USE=development_pilot' \
    'SEED=20260917 CONDITIONS=6 TASKS_PER_ARM=60' \
    'PAIRING=exact_stripped_question_disjoint_labels_unique_RGB_pixels' \
    'CORE_CODE_MODELS_SCORING_UNCHANGED=true'
  "$PY" -B "$PKG/check_package.py"
  "$PY" -B -c 'import sys; from pathlib import Path; from boundary_fetch import _verify_existing; m=_verify_existing(Path(sys.argv[1]), "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"); print("PINNED_DATA_OK", m["resolved_revision"])' "$DATA_ROOT"

  if ! test -s "$WORK/selection_manifest.json"; then
    "$PY" -B "$PKG/boundary_prepare.py" \
      --source "$DATA_ROOT" --format chartqa-parquet \
      --work "$WORK" --blocks 10 --seed 20260917
  fi

  "$PY" -B -c 'import json,sys; from pathlib import Path; from boundary_run import load_prepared; tasks,m=load_prepared(Path(sys.argv[1])); assert m["blocks"]==m["block_count"]==10, "Wrong block count"; assert m["task_count"]==len(tasks)==60, "Wrong task count"; assert m["seed"]==20260917; assert m["evaluation_use"]=="development_pilot"; assert m["source_audit"]["download_manifest"]["resolved_revision"]=="b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"; print("PILOT10_READY blocks=10 tasks=60"); print(json.dumps(m["files"],sort_keys=True))' "$WORK"
) 2>&1 | tee "$PREP_LOG"

flock -u 9
exec 9>&-

LOG="$CONTROL/run_pilot10_$(date +%Y%m%d_%H%M%S).log"
nohup bash /opt/care-workspace/run_boundary_root_gpu1.sh all >"$LOG" 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" >"$CONTROL/run.pid"
printf '%s\n' "$LOG" >"$CONTROL/latest_log.path"
disown "$run_pid" 2>/dev/null || true
printf 'PID=%s\nLOG=%s\nPREP_LOG=%s\n' "$run_pid" "$LOG" "$PREP_LOG"
sleep 3
ps -fp "$run_pid" || true
tail -n 80 "$LOG"
)
```

## 进度与日志

```bash
tail -F "$(cat /opt/care-workspace/care_opd_boundary_gpu1_control/latest_log.path)"
```

```bash
bash /opt/care-workspace/run_boundary_root_gpu1.sh status
df -h /opt/care-workspace
```

预期任务总数为 blocks=10 tasks=60。status 只反映已保存数量，不代表科学指标达标。

结果根为 /opt/care-workspace/care_opd_multimedia_boundary_gpu1_v1_diskfix，汇总在 summary/<评分指纹>/。程序完成四模型和两基线后自动评分。

本说明不代表服务器任务已启动，也不代表 GPU 运行已通过；命令需用户在服务器实际执行。不得为了通过准备阶段删除已有非空 WORK 或修改冻结清单；如报目录非空或计数不符，应保留现场。
