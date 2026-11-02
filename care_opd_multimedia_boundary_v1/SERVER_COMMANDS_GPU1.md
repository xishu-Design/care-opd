# 多媒体证据边界 V1：服务器完整指令

默认物理卡 **1**，所有模型串行占用此卡。不会检查该卡是否有其他计算进程，不会清卡或杀进程；如显存不足会保留错误并停止。程序仍保留同一结果目录的并发写入锁，避免重复启动破坏结果。

## 1. 上传并解压

把以下两个文件上传到 `/opt/care-workspace/`：

- `CARE_OPD_MULTIMEDIA_BOUNDARY_GPU1_V1.tar.gz`
- `CARE_OPD_MULTIMEDIA_BOUNDARY_GPU1_V1.tar.gz.sha256`

以下各段括号创建子 shell，检查失败也不会退出当前交互终端。不需要记住旧会话的环境变量。

```bash
(
set -e
cd /opt/care-workspace
sha256sum -c CARE_OPD_MULTIMEDIA_BOUNDARY_GPU1_V1.tar.gz.sha256
PKG=/opt/care-workspace/care_opd_multimedia_boundary_v1
if test -e "$PKG"; then
  echo "目录已存在，不覆盖：$PKG。已解压过则跳过本段，直接核验。" >&2
  exit 2
fi
install -d -m 700 "$PKG"
tar -xzf CARE_OPD_MULTIMEDIA_BOUNDARY_GPU1_V1.tar.gz -C "$PKG"
/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python -B "$PKG/check_package.py"
)
```

包内已包含所需旧版 HF 推理辅助代码，不需要复制旧脚本、不需要设置 PYTHONPATH。

## 2. 准备独立 CPU 数据环境与程序自测

只在新的数据环境中安装 Pillow、pyarrow 和下载库，不升级已经验证过的 Torch/Transformers GPU 环境。

```bash
(
set -e
DATA_ENV=/opt/care-data/venvs/care_boundary_data
if ! test -x "$DATA_ENV/bin/python"; then
  /opt/care-workspace/opdrevi_venv/bin/python -m venv "$DATA_ENV"
fi
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$DATA_ENV/bin/python" -m pip install \
  -r /opt/care-workspace/care_opd_multimedia_boundary_v1/requirements_data.txt
bash /opt/care-workspace/care_opd_multimedia_boundary_v1/run_boundary_gpu1.sh selftest
)
```

上面 pip 命令临时绕过此前失效的代理；若服务器必须经可用代理联网，去掉该行的 `env -u ...`，直接运行 Python 即可。不要在 GPU 环境重新安装依赖。

## 3. 准备数据、冻结配置与预检

默认输出根为 `/opt/care-data/care_opd_multimedia_boundary_gpu1_v1`。新数据独立保存，不动上一轮结果。首次需联网下载 ChartQA validation，生成时强制本地离线。

```bash
(
set -e
RUNNER=/opt/care-workspace/care_opd_multimedia_boundary_v1/run_boundary_gpu1.sh
IGNORE_PROXY=1 bash "$RUNNER" prepare
bash "$RUNNER" init
bash "$RUNNER" preflight
)
```

预期依次出现数据准备成功、`CONFIG_READY`/`CONFIG_EXISTS` 和 `PREFLIGHT_PASS`。初次会完整计算五个 checkpoint 哈希，耗时取决于磁盘吞吐，不等于卡住。若 validation 中凑不齐 24 个合规配对块，会明确失败；不要复制数据或根据模型得分改样本。

`init` 优先读取上一轮 followup 的模型路径与指纹。如果服务器路径不同，可在首次 init 前显式指定已有配置和视觉快照：

```bash
SOURCE_CONFIG=/opt/care-data/care_opd_tmm_visual_followup_gpu1/experiment.json \
VLM_MODEL=/实际路径/Qwen2.5-VL-7B-Instruct快照目录 \
bash /opt/care-workspace/care_opd_multimedia_boundary_v1/run_boundary_gpu1.sh init
```

这是可选替代命令，不要原样运行带“实际路径”的示例。默认会从 `/opt/care-data/model_cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/refs/main` 或唯一 snapshots 子目录定位 VLM；找不到就报错，不把不存在的本地路径当 Hub repo 下载。

## 4. 建议先试跑一个配对块

```bash
bash /opt/care-workspace/care_opd_multimedia_boundary_v1/run_boundary_gpu1.sh smoke
```

这会先完成一个块的 6 个条件及所有模型。仅检查真实模型能否加载、输入是否超预算、缓存和输出是否正常写盘；不要按答案好坏挑选、重抽、改提示。试跑首输出在后续全量中直接保留。试跑可以跳过，但不能与全量同时运行。

## 5. 后台运行全部并自动汇总

完成前面准备与预检后运行下面这一整段。无需再次 export 旧 RUNNER 或 TCM 变量。

```bash
(
set -e
RUNNER=/opt/care-workspace/care_opd_multimedia_boundary_v1/run_boundary_gpu1.sh
WORK=/opt/care-data/care_opd_multimedia_boundary_gpu1_v1
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="/opt/care-data/care_opd_multimedia_boundary_gpu1_v1.${STAMP}.log"
nohup env PHYSICAL_GPU=1 WORK="$WORK" bash "$RUNNER" run \
  >"$LOG" 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" > /opt/care-data/care_opd_multimedia_boundary_gpu1_v1.pid
printf '%s\n' "$LOG" > /opt/care-data/care_opd_multimedia_boundary_gpu1_v1.latest_log.path
disown "$run_pid" 2>/dev/null || true
echo "PID=$run_pid"
echo "LOG=$LOG"
sleep 3
ps -fp "$run_pid" || true
tail -n 80 "$LOG"
)
```

四个模型为 `clean_sft/gkd_full/tropd_full/care_opd`，不是重训。共 144 请求/模型；两个规则基线无需 GPU。视觉前端先生成共享观察，释放后再依次加载文本模型，不会把五个模型同时放进 GPU。

## 6. 看进度与日志

```bash
bash /opt/care-workspace/care_opd_multimedia_boundary_v1/run_boundary_gpu1.sh status
tail -n 120 "$(cat /opt/care-data/care_opd_multimedia_boundary_gpu1_v1.latest_log.path)"
ps -fp "$(cat /opt/care-data/care_opd_multimedia_boundary_gpu1_v1.pid)" || true
```

实时看日志：

```bash
tail -F "$(cat /opt/care-data/care_opd_multimedia_boundary_gpu1_v1.latest_log.path)"
```

`Ctrl+C` 只结束 tail，不终止后台实验。`status` 只报告已保存条数，144/144 不代表回答全正确。

## 7. 中断后续跑

先确认旧进程已结束，再重复第 5 段即可；`run` 和 `resume` 都按本目录已保存记录补缺。也可前台运行：

```bash
bash /opt/care-workspace/care_opd_multimedia_boundary_v1/run_boundary_gpu1.sh resume
```

遇到 `.boundary.lock` 不要直接删除，先读锁里的主机/PID，并核对是否仍有该实验进程。确认是异常退出留下的锁后，将这一特定文件改名留档再续跑；不批量删除任何日志、响应或 GPU 进程。JSONL 行尾不完整会停止而不是自动丢行，以免首输出被悄悄改写。若代码、权重或协议变了，应使用新的 WORK，不能拼接两轮结果。

## 8. 单独汇总与取回结果

全量完成后自动评分；只重跑 CPU 评分也可以：

```bash
bash /opt/care-workspace/care_opd_multimedia_boundary_v1/run_boundary_gpu1.sh score
find /opt/care-data/care_opd_multimedia_boundary_gpu1_v1/summary -maxdepth 2 -type f -printf '%p\n' | sort
```

主要文件在 `summary/<评分指纹>/`：

- `summary.md`：主表与解释边界；
- `metrics.csv`、`metrics.json`：正常有效回答、换图双正确、局部失效保护、越界与拒答等指标；
- `paired_effects.csv`：整块配对 bootstrap 差值与区间；
- `frontend_metrics.csv`：前端识别和攻击目标暴露诊断；
- `*_strict_rows.private.csv`、`*_semantic_rows.private.csv`：逐条评分审计；
- `scoring_manifest.json`：该次评分绑定的输入哈希。

原始输出在 `responses/<模型名>/responses.jsonl`；共享观察在 `observations.jsonl`。评分不需要 GPU，也不会重生成任何回答。没收齐全部六臂时拒绝产生最终汇总。

请整体保留本轮 WORK 和旧实验目录。主结果要同时看有效回答与边界表现，不能只选择 CARE-OPD 最有利的条件；该轮仍是开发性 pilot，不声称已经形成正式论文结论。
