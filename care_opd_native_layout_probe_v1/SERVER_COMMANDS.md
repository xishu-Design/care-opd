# 卡 1：冻结缓存上的证据布局开发对照

以下路径沿用既有服务器布局。请先将压缩包及 `.sha256` 上传到 `$HOME/lcx`。这是 **72 次文本生成、0 次视觉生成、0 次训练** 的默认小对照，不覆盖旧负结果。

## 1. 解压并静态检查（不占 GPU）

整段放在子 shell 中，检查失败只退出该段，不关闭交互终端。不要 source 启动脚本。

```bash
(
set -euo pipefail
cd "$HOME/lcx"
sha256sum -c CARE_OPD_NATIVE_LAYOUT_PROBE_GPU1_V1.tar.gz.sha256
PKG="$HOME/lcx/care_opd_native_layout_probe_v1"
if test -e "$PKG"; then
  echo "包目录已存在；不要覆盖正在使用的脚本。已解压过可直接进行下一步。" >&2
  exit 2
fi
install -d -m 700 "$PKG"
tar -xzf CARE_OPD_NATIVE_LAYOUT_PROBE_GPU1_V1.tar.gz -C "$PKG"

export HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export SOURCE_WORK="$HOME/lcx/care_opd_multimedia_boundary_gpu1_v1_diskfix"
export WORK="$HOME/lcx/care_opd_native_layout_probe_gpu1_v1"
export PHYSICAL_GPU=1
export BLOCKS=3
test -s "$SOURCE_WORK/observations.jsonl"
test -s "$SOURCE_WORK/run_manifest.json"
bash "$PKG/run_probe_gpu1.sh" selftest
bash "$PKG/run_probe_gpu1.sh" prepare
bash "$PKG/run_probe_gpu1.sh" check
)
```

如果原结果后来搬了位置，只修改 `SOURCE_WORK` 为真实的完整原目录。不要指向 summary 子目录。首次准备完成会显示 `tasks=18 new_text_generations=72 new_visual=0`。

## 2. 后台运行，日志与 PID 写入新目录

本段重新给出所有变量，换终端也不会因变量丢失启动空路径。没有 GPU 空闲进程检查，不会杀任何已有任务；共享显存不足仍会 OOM。

```bash
(
set -euo pipefail
PKG="$HOME/lcx/care_opd_native_layout_probe_v1"
export HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export SOURCE_WORK="$HOME/lcx/care_opd_multimedia_boundary_gpu1_v1_diskfix"
export WORK="$HOME/lcx/care_opd_native_layout_probe_gpu1_v1"
export PHYSICAL_GPU=1
export BLOCKS=3

bash "$PKG/run_probe_gpu1.sh" check
# 完整模型 SHA256 校验后再启动，不要求 VLM checkpoint 存在。
FULL_HASH=1 bash "$PKG/run_probe_gpu1.sh" preflight

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$WORK/run_gpu1.$STAMP.log"
PID_FILE="$WORK/run_gpu1.pid"
if test -s "$PID_FILE"; then
  previous_pid=$(head -n 1 "$PID_FILE")
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "PID 文件对应进程仍在，先人工核对，避免重复启动：$previous_pid" >&2
    exit 2
  fi
fi
nohup bash "$PKG/run_probe_gpu1.sh" run >"$LOG" 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" >"$PID_FILE"
printf '%s\n' "$LOG" >"$WORK/latest_log.txt"
disown "$run_pid" 2>/dev/null || true
echo "PID=$run_pid LOG=$LOG"
sleep 3
ps -fp "$run_pid" || true
tail -n 80 "$LOG"
)
```

这里的 PID 检查只防止本实验重复启动，不是检查 GPU 上是否有其他进程。若确需共享卡，不需要改脚本。

## 3. 查看进度/汇总

```bash
export HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export WORK="$HOME/lcx/care_opd_native_layout_probe_gpu1_v1"
PKG="$HOME/lcx/care_opd_native_layout_probe_v1"
bash "$PKG/run_probe_gpu1.sh" status
tail -n 100 "$(cat "$WORK/latest_log.txt")"
find "$WORK/summary" -maxdepth 2 -type f -print 2>/dev/null
```

运行完成会自动评分。若只需重新生成同一份汇总，可运行：

```bash
export HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export WORK="$HOME/lcx/care_opd_native_layout_probe_gpu1_v1"
bash "$HOME/lcx/care_opd_native_layout_probe_v1/run_probe_gpu1.sh" score
```

因基础设施错误退出后，确认没有旧运行进程再重复第 2 段即可，已封存的输出会被跳过。不要为了改善内容删除响应文件；不要手工移除仍被占用的锁。
