# /data 已满、尚未准备任务时的 GPU1 启动入口

适用状态：服务器已有原始 `care_opd_multimedia_boundary_v1` 代码包，但尚无本轮数据或任务清单。不是迁移已有任务，不删除旧文件，不复制模型。

上传 `run_boundary_root_gpu1.sh` 和其 `.sha256` 到 `/opt/care-workspace/`。原始 tar 不需要重新上传或修改。

```bash
(
set -e
cd /opt/care-workspace
sha256sum -c run_boundary_root_gpu1.sh.sha256
CONTROL=/opt/care-workspace/care_opd_boundary_gpu1_control
mkdir -p "$CONTROL"
LOG="$CONTROL/run_$(date +%Y%m%d_%H%M%S).log"
nohup bash /opt/care-workspace/run_boundary_root_gpu1.sh all >"$LOG" 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" > "$CONTROL/run.pid"
printf '%s\n' "$LOG" > "$CONTROL/latest_log.path"
disown "$run_pid" 2>/dev/null || true
echo "PID=$run_pid LOG=$LOG"
sleep 3
ps -fp "$run_pid" || true
tail -n 80 "$LOG"
)
```

`all` 创建 `/root` 下独立 CPU 数据环境，再执行自测、下载 validation 数据、准备任务、初始化、GPU 预检、四臂生成和评分。数据、任务、日志及常见运行缓存均设到 `/root`。GPU 环境和 checkpoint 仍从原路径读取，不重装 Torch、不复制或下载模型。

```bash
tail -F "$(cat /opt/care-workspace/care_opd_boundary_gpu1_control/latest_log.path)"
```

```bash
bash /opt/care-workspace/run_boundary_root_gpu1.sh status
df -h /opt/care-workspace
```

准备前 `status` 显示 `DATA_PREPARATION_NOT_COMPLETE` 是阶段提示，是否出错以日志为准。先出现依赖安装/数据准备而非 GPU 生成是正常顺序。

该启动器默认临时绕过失效代理（`IGNORE_PROXY=1`）；服务器必须使用可用代理时，启动命令改成 `nohup env IGNORE_PROXY=0 bash ...`。阶段性空间检查要求根分区至少有 8 GiB 和 10000 个空闲 inode，但不能保证其他并行任务不会继续消耗空间。不要通过降低空间门槛来解决满盘。

后台退出时先看日志。已准备成功的同协议任务再次执行 `all` 会复用；完整配置已具备时 `resume` 可只补推理及评分。不要修改冻结的任务、代码、模型和已完成首输出。重复启动会被启动器自身锁阻止，不检查或终止其他 GPU 任务。

结果根目录为 `/opt/care-workspace/care_opd_multimedia_boundary_gpu1_v1_diskfix`，汇总位于其 `summary/<评分指纹>/`。

本辅助脚本仅完成本地 Bash 语法和路径检查，原核心实验包测试状态不变；真实联网下载与 GPU 执行须在服务器验收。
