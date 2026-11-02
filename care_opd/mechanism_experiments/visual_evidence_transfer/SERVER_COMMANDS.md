# 服务器完整命令：冻结视觉迁移 + TROPD

所有步骤只做推理，不重训。不启动 Web 服务，不要求 LightRAG 索引。
默认示例使用物理卡 **6**，运行前自行修改 `PHYSICAL_GPU`；脚本**不检查是否存在其他 GPU 计算进程，也不会清卡或杀进程**。共享卡显存不足仍会 OOM。
代码位于 `/opt/care-workspace/care_opd`，数据和新输出位于 `/opt/care-data/care_opd_tmm_visual_v1`。
如果服务器使用其他代码目录，替换 `CARE_ROOT`。所有命令均使用 Bash。

## 1. 上传并解压本次脚本包

把本次交付的 `CARE_OPD_TMM_VISUAL_TROPD_V1.tar.gz` 和对应 `.sha256` 上传到 `/opt/care-workspace`。
包只包含两个新增实验目录，不包含模型、测试数据或旧实验结果。

```bash
(
  set -e
  cd /opt/care-workspace
  sha256sum -c CARE_OPD_TMM_VISUAL_TROPD_V1.tar.gz.sha256
  test -d /opt/care-workspace/care_opd
  tar -xzf CARE_OPD_TMM_VISUAL_TROPD_V1.tar.gz -C /opt/care-workspace/care_opd
)
```

括号保证校验失败只退出子 shell，不关闭交互终端。后面的脚本同样用 `bash 文件`，不依赖可执行位。

## 2. 一次性设定路径和环境

```bash
export CARE_ROOT=/opt/care-workspace/care_opd
export EXP_DIR="$CARE_ROOT/mechanism_experiments/visual_evidence_transfer"
export WORK=/opt/care-data/care_opd_tmm_visual_v1
export HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
export PHYSICAL_GPU=6
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
install -d -m 700 "$WORK"
"$HF_PY" -c 'import torch, transformers, PIL, accelerate, huggingface_hub; from transformers import Qwen2_5_VLForConditionalGeneration; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print("qwen3_5_text",CONFIG_MAPPING["qwen3_5_text"]); print("STACK",torch.__version__,transformers.__version__)'
```

`TROP_MODEL` 是既有文档的标准目录；若实际训练保存到了别处，请改为**完整合并权重目录**，不能填只含 LoRA adapter 的目录。
`HF_PY` 沿用之前已验证能够加载 Qwen3.5 文本模型的环境；这里还需支持原生 Qwen2.5-VL。
不建议为这组实验直接升级原训练环境的 torch/vLLM。脚本完全不依赖 vLLM。
如只缺 `PIL` 或数据下载工具，可在选定环境安装缺少的 `pillow` / `huggingface_hub`；其余加载问题应先看预检具体错误。

## 3. 下载公开数据，准备实际视觉前端

```bash
"$HF_PY" "$EXP_DIR/fetch_assets.py" slake --dest "$WORK/data/slake"
```

这一步下载固定版本的官方 SLAKE JSON 和图片，记录下载来源和 SHA-256，不构建知识索引。
下载缓存放在 `/opt/care-data`，避免占满 `/root` 所在分区；网络中断后重跑下载命令，保留缓存以便续传。

如果 `VLM_MODEL` 已经是完整的 Qwen2.5-VL 检查点，直接使用，不需要重复下载。
仅在确实没有模型时执行下面的**独立大文件下载命令**（7B 视觉模型需要较多磁盘和网络）：

```bash
"$HF_PY" "$EXP_DIR/fetch_assets.py" vlm --repo Qwen/Qwen2.5-VL-7B-Instruct --dest "$VLM_MODEL"
```

正式推理启用离线模式，模型路径不存在时会报错，不偷偷从网络下载或换成别的模型。

## 4. 先在验证集试跑 10 个题族

一个题族至少有 clean/no_image 两个任务，有有效相反标签图像时还有 swapped_image。
这一步只验证实现和可用性，不能当成正式主结果；不要依据 CARE 得分反复调整提示。

```bash
"$HF_PY" "$EXP_DIR/prepare_slake.py" --dataset-root "$WORK/data/slake" --output-dir "$WORK/tasks_validation10" --split validation --max-items 10 --seed 20260915

"$HF_PY" "$EXP_DIR/evaluate.py" init-config --output "$WORK/config_validation10.json" --tasks-dir "$WORK/tasks_validation10" --run-root "$WORK/runs_validation10" --vlm "$VLM_MODEL" --clean-sft "$CLEAN_MODEL" --gkd-full "$GKD_MODEL" --tropd-full "$TROP_MODEL" --care-opd "$CARE_MODEL"

bash "$EXP_DIR/run_visual_transfer.sh" preflight "$WORK/config_validation10.json"
```

预检成功后开始试跑（五个配置依次加载，一次只驻留一个模型）：

```bash
if bash "$EXP_DIR/run_visual_transfer.sh" preflight "$WORK/config_validation10.json"; then
  RUN_LOG="$WORK/validation10.nohup.log"
  nohup env HF_PY="$HF_PY" PHYSICAL_GPU="$PHYSICAL_GPU" bash "$EXP_DIR/run_visual_transfer.sh" all "$WORK/config_validation10.json" >"$RUN_LOG" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$WORK/validation10.pid"
  disown "$pid" 2>/dev/null || true
  echo "PID=$pid LOG=$RUN_LOG"
else
  echo '预检未通过，未启动。交互终端保持打开。'
fi
```

查看日志和汇总：

```bash
tail -F /opt/care-data/care_opd_tmm_visual_v1/validation10.nohup.log
```

`Ctrl+C` 只结束 tail，不终止后台实验。
评分的 `COMPLETE` 表示冻结任务都已返回可统计输出，不表示医学正确率或合同成功率达到某个分数。
不能因为低分、空输出或格式错误而让同一任务重新生成。

## 5. 冻结正式测试任务并创建正式配置

`--max-items 0` 表示使用固定规则过滤后的全部合格题族。正式集排除与 validation 相同的图像，数量以 manifest 为准。
如需资源上限，应**在首次生成前**改为固定数量并记录，不得生成后选样。

```bash
"$HF_PY" "$EXP_DIR/prepare_slake.py" --dataset-root "$WORK/data/slake" --output-dir "$WORK/tasks_test" --split test --max-items 0 --seed 20260915

"$HF_PY" "$EXP_DIR/evaluate.py" init-config --output "$WORK/config_test.json" --tasks-dir "$WORK/tasks_test" --run-root "$WORK/runs_test" --vlm "$VLM_MODEL" --clean-sft "$CLEAN_MODEL" --gkd-full "$GKD_MODEL" --tropd-full "$TROP_MODEL" --care-opd "$CARE_MODEL"

bash "$EXP_DIR/run_visual_transfer.sh" preflight "$WORK/config_test.json"
```

config 文件会保存全部路径，因此重新开终端后不用重建巨大的 `RUN_ENV` 数组。
初次预检会计算各检查点完整哈希，日志暂时停在 `HASH_MODEL` 是读取权重做审计，不是模型推理。

## 6. 正式后台运行

下面这一段重新开终端也能执行。卡号在 `PHYSICAL_GPU=6` 处改。

```bash
HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python
RUNNER=/opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer/run_visual_transfer.sh
CONFIG=/opt/care-data/care_opd_tmm_visual_v1/config_test.json
RUN_LOG=/opt/care-data/care_opd_tmm_visual_v1/test.nohup.log
RUN_PID=/opt/care-data/care_opd_tmm_visual_v1/test.pid

if env HF_PY="$HF_PY" PHYSICAL_GPU=6 bash "$RUNNER" preflight "$CONFIG"; then
  nohup env HF_PY="$HF_PY" PHYSICAL_GPU=6 bash "$RUNNER" all "$CONFIG" >>"$RUN_LOG" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$RUN_PID"
  disown "$pid" 2>/dev/null || true
  echo "PID=$pid LOG=$RUN_LOG"
else
  echo '预检失败，未启动；请保留上方错误。'
fi
```

`all` 顺序为观察生成 → VLM 直接回答 → Clean-SFT → GKD-Full → TROPD-Full → CARE-OPD → 自动评分。
已完成的阶段/任务会按身份与输入校验后跳过，异常中断后可使用完全相同命令恢复。

## 7. 只补 TROPD 的本组图文结果

先要有同一正式运行根里的全部共享观察。
无需重跑已经完成的其他四个配置：

```bash
env HF_PY=/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python PHYSICAL_GPU=6 bash /opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer/run_visual_transfer.sh tropd_full /opt/care-data/care_opd_tmm_visual_v1/config_test.json
```

所有配置完成后单独重新汇总（不需要 GPU）：

```bash
/opt/care-data/venvs/mhg_eval_qwen25vl/bin/python /opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer/score_results.py --config /opt/care-data/care_opd_tmm_visual_v1/config_test.json
```

## 8. 结果在哪里

```text
/opt/care-data/care_opd_tmm_visual_v1/runs_test/observations/observations.jsonl
/opt/care-data/care_opd_tmm_visual_v1/runs_test/arms/vlm_direct/responses.jsonl
/opt/care-data/care_opd_tmm_visual_v1/runs_test/arms/clean_sft/responses.jsonl
/opt/care-data/care_opd_tmm_visual_v1/runs_test/arms/gkd_full/responses.jsonl
/opt/care-data/care_opd_tmm_visual_v1/runs_test/arms/tropd_full/responses.jsonl
/opt/care-data/care_opd_tmm_visual_v1/runs_test/arms/care_opd/responses.jsonl
/opt/care-data/care_opd_tmm_visual_v1/runs_test/summary/
```

`summary/metrics.csv` 是逐方法、逐条件汇总表，`metrics.json` 保存完整统计与运行完整性状态，
`paired_effects.csv` 保存 CARE 相对各对照的配对差值与区间，`summary.md` 可直接阅读。

```bash
tail -n 120 /opt/care-data/care_opd_tmm_visual_v1/test.nohup.log
find /opt/care-data/care_opd_tmm_visual_v1/runs_test -maxdepth 3 -type f -printf '%s %p\n' | sort
```

没有完成的配置不会被隐去；评分会产生 INCOMPLETE 诊断并返回非零，不能将它标为完整对比。

## 9. 之前实验的 TROPD 补测

本组图文迁移与原来 QZBC、真实外测双轨是独立评测。
后者运行方法详见 `../tropd_backfill/SERVER_COMMANDS.md`，不要把图文目录传给旧评测脚本。
同卡不要同时启动两个大模型评测；可在本组结束后顺序运行 TROPD backfill。

## 10. 本地代码测试

```bash
python -m unittest discover -s /opt/care-workspace/care_opd/mechanism_experiments/visual_evidence_transfer -p 'test_*.py' -v
```

单元测试使用合成小样例和模拟生成器，不等于已经完成服务器 GPU 推理，也不是论文实验结果。
