# 多模态视觉证据迁移：物理 GPU 4 / 5 完整运行指令

本实验只做推理，不重训，不依赖 LightRAG。它与 GPU 6 / 7 的 QZBC、真实医疗文本补测相互独立。

执行顺序固定为先在 **GPU 4** 生成一份共享视觉观察；观察全部完成并校验后，再并行运行两条分支。

| 物理卡 | 共享观察完成后的任务顺序 |
| --- | --- |
| GPU 4 | VLM 直接回答 → GKD-Full |
| GPU 5 | Clean-SFT → TROPD-Full → CARE-OPD |

两条分支共用同一批任务和同一份视觉观察，每张卡一次只加载一个模型。脚本不检查空卡、不清卡、不终止其他进程；共享卡显存不足仍可能 OOM。这里不是单个模型的双卡张量并行。

所有命令在服务器 **Bash** 中运行。带括号的准备块在子 shell 中失败，不会关闭当前交互终端。正式结果保存在全新的 `/opt/care-data/care_opd_tmm_visual_gpu45_v3`，不覆盖旧实验结果。

## 1. 上传和安装脚本包

将 `CARE_OPD_TMM_VISUAL_GPU45_V3_1.tar.gz` 和同名 `.sha256` 上传到 `/opt/care-workspace`。包仅包含 `visual_evidence_transfer` 目录源码，不包含模型、测试数据，也不改动正在运行的 `tropd_backfill` 目录。

```bash
(
  set -e
  cd /opt/care-workspace
  sha256sum -c CARE_OPD_TMM_VISUAL_GPU45_V3_1.tar.gz.sha256
  test -d /opt/care-workspace/care_opd
  tar -xzf CARE_OPD_TMM_VISUAL_GPU45_V3_1.tar.gz -C /opt/care-workspace/care_opd
  test -s /opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer/run_visual_transfer_gpu45.sh
)
```

确认校验和解压成功，再执行后续步骤。不要在本组实验运行期间再次解压覆盖源码。

## 2. 路径及现有环境

```bash
export CARE_ROOT=/opt/care-workspace/care_opd
export EXP_DIR="$CARE_ROOT/mechanism_experiments/visual_evidence_transfer"
export WORK=/opt/care-data/care_opd_tmm_visual_gpu45_v3
export HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export HF_HOME=/opt/care-data/hf_cache
export HF_HUB_CACHE=/opt/care-data/hf_cache/hub
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

export VLM_MODEL=/opt/care-data/models/Qwen2.5-VL-7B-Instruct
export CLEAN_MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged
export GKD_MODEL=/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/gkd_full/final_model
export TROP_MODEL=/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/baselines/tropd_full/final_model
export CARE_MODEL=/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/final_model

umask 077
install -d -m 700 "$WORK" "$WORK/logs"
"$HF_PY" -c 'import torch, transformers, PIL, accelerate, huggingface_hub; from transformers import Qwen2_5_VLForConditionalGeneration; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print("qwen3_5_text", CONFIG_MAPPING["qwen3_5_text"]); print("STACK", torch.__version__, transformers.__version__)'
```

这些模型路径是已有项目的标准路径，不表示文件一定已经存在。四个文本模型必须是对应的**完整合并检查点**，不是仅包含 LoRA adapter 的目录。若模型实际在别处，请在生成配置前修改相应变量。不要用别的方法检查点替代缺失模型。

本组沿用已验证的 Python 环境，另需原生 Qwen2.5-VL 支持；不依赖 vLLM，不要求升级 torch、Transformers 或原训练环境。上面的导入检查若失败，先处理明确缺少的依赖，不要继续后台启动。

## 3. 准备 SLAKE 和视觉模型

下载固定版本的 SLAKE 数据及图像，记录来源和哈希，不构建知识索引：

```bash
env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$HF_PY" "$EXP_DIR/fetch_assets.py" slake --dest "$WORK/data/slake"
```

若已有完整的 Qwen2.5-VL-7B-Instruct 本地模型，可以直接把 `VLM_MODEL` 指向它，不重复下载。若尚未下载，执行：

```bash
env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$HF_PY" "$EXP_DIR/fetch_assets.py" vlm --repo Qwen/Qwen2.5-VL-7B-Instruct --dest "$VLM_MODEL"
```

第二条是独立的大模型下载，不是小脚本下载；缓存与模型副本都会占磁盘。中断后保留 `/opt/care-data/hf_cache` 缓存，再运行同一命令。下载器不会覆盖来源不明的既有目录；如果指定目录已存在但不完整，先核对其内容，并为本次下载另选一个新的模型目录，不要直接删除旧目录。

推理阶段由运行脚本强制使用离线模型路径，不会自动下载其他模型。

## 4. 生成验证集 10 题族并预检

每个题族包含正常图像和缺失图像两种状态；满足固定配对规则的题族另外包含替换图像状态。因此 10 题族不等于 10 次生成。验证集只用于检查实现、环境和运行完整性，不按模型得分选择样本或调整提示。

```bash
(
  set -e
  "$HF_PY" "$EXP_DIR/prepare_slake.py" \
    --dataset-root "$WORK/data/slake" \
    --output-dir "$WORK/tasks_validation10" \
    --split validation --max-items 10 --seed 20260915

  "$HF_PY" "$EXP_DIR/evaluate.py" init-config \
    --output "$WORK/config_validation10.json" \
    --tasks-dir "$WORK/tasks_validation10" \
    --run-root "$WORK/runs_validation10" \
    --vlm "$VLM_MODEL" \
    --clean-sft "$CLEAN_MODEL" \
    --gkd-full "$GKD_MODEL" \
    --tropd-full "$TROP_MODEL" \
    --care-opd "$CARE_MODEL"

  env HF_PY="$HF_PY" bash "$EXP_DIR/run_visual_transfer_gpu45.sh" preflight "$WORK/config_validation10.json"
)
```

初次预检会读取全部检查点文件计算哈希，日志停留在 `HASH_MODEL` 时可能仍在读取权重。预检不是 GPU 推理完成，也不承诺共享卡剩余显存充足。

配置与运行身份会冻结，不能在运行后改模型、提示、采样参数或任务集继续混用原运行根。

## 5. 后台启动验证集

本段明确设置所有启动路径，重新开终端也能执行。日志使用时间戳，每次启动生成独立日志和 PID 文件，不覆盖之前日志。

```bash
(
  HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
  RUNNER=/opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer/run_visual_transfer_gpu45.sh
  WORK=/opt/care-data/care_opd_tmm_visual_gpu45_v3
  CONFIG="$WORK/config_validation10.json"
  umask 077
  install -d -m 700 "$WORK/logs"
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
  RUN_LOG="$WORK/logs/validation10_${RUN_STAMP}.nohup.log"
  RUN_PID="$WORK/logs/validation10_${RUN_STAMP}.pid"

  if env HF_PY="$HF_PY" bash "$RUNNER" preflight "$CONFIG"; then
    nohup env HF_PY="$HF_PY" bash "$RUNNER" all "$CONFIG" >>"$RUN_LOG" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$RUN_PID"
    disown "$pid" 2>/dev/null || true
    echo "PID=$pid"
    echo "LOG=$RUN_LOG"
    echo "PID_FILE=$RUN_PID"
    sleep 3
    ps -fp "$pid" || true
    tail -n 80 "$RUN_LOG"
  else
    echo '验证集预检失败，未启动；请保留上方错误。终端保持打开。'
  fi
)
```

检查验证集是否完成：

```bash
/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_validation10/summary/metrics.json')
if p.is_file():
    print('VALIDATION_STATUS=' + str(json.loads(p.read_text(encoding='utf-8')).get('status')))
else:
    print('验证集汇总尚未生成，请查看验证集启动日志。')
PY
```

`COMPLETE` 只表示全部冻结任务已返回可统计输出，不要求准确率、格式通过率或任务成功率达到某个分数。已有的低分、空文本或格式错误输出照常保留，不因分数低而重新生成。

## 6. 冻结正式测试集

等待验证集运行完成后，再执行本段。正式测试使用固定规则过滤后的全部合格题族，自动排除与**整个 validation 集**相同的图像，而不只是排除试跑的 10 题。具体数量以 `selection_manifest.json` 为准，不能根据结果选样。

```bash
(
  set -e
  export HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
  export EXP_DIR=/opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer
  export WORK=/opt/care-data/care_opd_tmm_visual_gpu45_v3

  "$HF_PY" - "$WORK/runs_validation10/summary/metrics.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    report = json.load(f)
assert report.get('status') == 'COMPLETE', report.get('status')
print('VALIDATION_COMPLETE')
PY

  "$HF_PY" "$EXP_DIR/prepare_slake.py" \
    --dataset-root "$WORK/data/slake" \
    --output-dir "$WORK/tasks_test" \
    --split test --max-items 0 --seed 20260915

  "$HF_PY" - "$WORK/config_validation10.json" "$WORK/config_test.json" "$WORK/tasks_test" "$WORK/runs_test" "$EXP_DIR/evaluate.py" <<'PY'
import json, subprocess, sys
source, output, tasks, run_root, evaluator = sys.argv[1:]
with open(source, encoding='utf-8') as f:
    config = json.load(f)
cmd = [sys.executable, evaluator, 'init-config', '--output', output,
       '--tasks-dir', tasks, '--run-root', run_root]
for key in ('vlm', 'clean_sft', 'gkd_full', 'tropd_full', 'care_opd'):
    cmd.extend(['--' + key.replace('_', '-'), config['models'][key]])
subprocess.run(cmd, check=True)
PY

  env HF_PY="$HF_PY" bash "$EXP_DIR/run_visual_transfer_gpu45.sh" preflight "$WORK/config_test.json"
)
```

正式配置从验证配置读取相同的模型路径；这样重新开终端后也不会因遗失模型环境变量而指错模型。它仍使用原有 `init-config` 默认生成参数，没有按验证得分调参。

## 7. GPU 4 / 5 正式后台运行

```bash
(
  HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
  RUNNER=/opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer/run_visual_transfer_gpu45.sh
  WORK=/opt/care-data/care_opd_tmm_visual_gpu45_v3
  CONFIG="$WORK/config_test.json"
  umask 077
  install -d -m 700 "$WORK/logs"
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$$"
  RUN_LOG="$WORK/logs/test_${RUN_STAMP}.nohup.log"
  RUN_PID="$WORK/logs/test_${RUN_STAMP}.pid"

  if env HF_PY="$HF_PY" bash "$RUNNER" preflight "$CONFIG"; then
    nohup env HF_PY="$HF_PY" bash "$RUNNER" all "$CONFIG" >>"$RUN_LOG" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$RUN_PID"
    disown "$pid" 2>/dev/null || true
    echo "PID=$pid"
    echo "LOG=$RUN_LOG"
    echo "PID_FILE=$RUN_PID"
    sleep 3
    ps -fp "$pid" || true
    tail -n 100 "$RUN_LOG"
  else
    echo '正式预检失败，未启动；请保留上方错误。终端保持打开。'
  fi
)
```

开始时 GPU 5 暂时空闲是预期行为，因为四个文本方法必须等待同一份共享视觉观察完成。观察不是直接替模型填入标准答案。

不要另开单卡 `run_visual_transfer.sh all` 指向同一运行根，也不要同时启动验证集和正式集。运行根有全程父锁，重复启动会被拒绝；这是防止输出并发写入的检查，不是 GPU 空卡检查。若因进程/环境异常中断，可使用本段原命令恢复；身份校验通过的既有输出保留，不能通过删除输出重新抽取更有利答案。若异常断电后提示锁仍存在，应同时核对父进程和 `runs_test/runtime/workers.jsonl` 中的工作进程，不要直接删除锁再启动。

## 8. 看日志和最终结果

列出本次启动日志，从输出中选择需要查看的文件：

```bash
find /opt/care-data/care_opd_tmm_visual_gpu45_v3/logs -maxdepth 1 -type f -name '*.nohup.log' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

实时查看最新正式启动日志：

```bash
(
  WORK=/opt/care-data/care_opd_tmm_visual_gpu45_v3
  RUN_LOG=$(find "$WORK/logs" -maxdepth 1 -type f -name 'test_*.nohup.log' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d ' ' -f 2-)
  if test -n "$RUN_LOG"; then
    tail -F "$RUN_LOG"
  else
    echo '尚未找到正式运行日志。'
  fi
)
```

`Ctrl+C` 只结束 `tail`，不会终止后台实验。

总日志中的 `GPU45_PREFLIGHT=PASS` 表示预检通过，`GPU45_WORKER` 会给出当前分支 PID 和日志路径。观察阶段和双卡回答阶段的详细进度分别查看：

```bash
tail -n 80 /opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/logs/gpu4.observe.log
tail -n 80 /opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/logs/gpu4.answers.log
tail -n 80 /opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/logs/gpu5.answers.log
```

尚未开始的阶段没有对应日志属于正常情况。验证集的分支日志同理位于 `runs_validation10/logs`。只有 `GPU45_COMPLETE` 和正式汇总完整性状态共同确认完成，不能把打印工作进程 PID 当成推理已完成。

全部完成后查看正式汇总：

```bash
(
  SUMMARY=/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/summary
  if test -s "$SUMMARY/summary.md"; then
    cat "$SUMMARY/summary.md"
    cat "$SUMMARY/metrics.csv"
  else
    echo '正式汇总尚未生成，请查看启动日志和各分支日志。'
  fi
)
```

正式产物：

```text
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/observations/observations.jsonl
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/arms/vlm_direct/responses.jsonl
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/arms/clean_sft/responses.jsonl
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/arms/gkd_full/responses.jsonl
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/arms/tropd_full/responses.jsonl
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/arms/care_opd/responses.jsonl
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/summary/metrics.csv
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/summary/metrics.json
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/summary/paired_effects.csv
/opt/care-data/care_opd_tmm_visual_gpu45_v3/runs_test/summary/summary.md
```

`metrics.csv` 分方法、分证据状态报告结果；`paired_effects.csv` 报告 CARE 相对各对照的配对差值与区间。缺失方法不会从对比中悄悄删除，未完整运行会标为 `INCOMPLETE` 或诊断状态，不能当成完整结果。

需要只重新汇总而不进行模型推理时，等两条分支均已结束，再执行：

```bash
/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python /opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer/score_results.py --config /opt/care-data/care_opd_tmm_visual_gpu45_v3/config_test.json
```

本实验回答冻结视觉前端下，文本后训练方法如何利用、拒用或处理图像证据。它不等价于端到端多模态训练，也不直接证明临床诊疗效用。
