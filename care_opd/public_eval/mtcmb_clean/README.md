# MTCMB：污染感知的公开评测

本目录实现 MTCMB 的六模型、四 GPU、可断点续跑评测。报告不会把 MTCMB
全量误称为独立泛化：QZBC 正式训练种子直接使用了 MTCMB 中 1,062 个不同
记录（共使用 1,200 次），且全部来自 `TCM-ED-A` / `TCM-ED-B`。进一步的
result-blind 审计把直接 provenance、归一化题干精确匹配和预注册近邻规则取并集，
最终排除 1,186 条；其中 124 条是 direct provenance 之外的精确/近邻命中。

## 固定数据边界

| 视图 | 官方记录 | 本地客观评分记录 | 用途 |
|---|---:|---:|---|
| MTCMB-Full | 7,100 | 7,050 | 域内诊断，不作独立泛化结论 |
| overlap-clean | 5,914 | 5,864 | 移除 1,186 个 direct∪exact-stem∪near ID；ED-A/B 家族暴露仍存在 |
| provenance-clean Clean9 | 1,050 | 1,050 | 移除整个 ED-A/B 家族及远程 judge 任务，作补充泛化证据 |

Clean9 包含：`TCM-FT` 100、`TCMeEE` 100、`TCM-CHGD` 100、
`TCM-LitData` 100、`TCM-MSDD` 100、`TCM-Diagnosis` 200、
`TCM-PR` 100、`TCM-FRD` 200、`TCM-SE-B` 50。

`TCM-SE-A` 的官方协议要求远程 `GLM-4-Air-250414` 主观 judge，故本地协议
明确将其 50 条排除，不用自定义替代 judge 冒充官方分数。其余 11 项均按固定
官方公式离线评分：选择准确率、MSDD 部分多标签准确率、PR 处方集合复合分、
BERTScore、ROUGE 与 BLEU。各任务分数分别报告；跨任务仅给描述性等任务宏平均，
不计算样本量加权的“总分”。

选择题正确率复现上游 `make_answer` 的“首个允许的拉丁选项字母”解析器；另行报告
严格单字母格式率。两者不混用，避免用更宽松的事后 parser 抬高官方准确率。

## 鲁棒性口径

官方 clean 指标与派生鲁棒指标严格分开。派生扰动仅用于选择题：对选项作由
`protocol/task/id` 哈希确定的非恒等置换，并同步重映射 gold；预测再映回 canonical
label 后配对评分。

- overlap-clean：4,862 对（ED-A、ED-B、SE-B；仍有家族暴露）。
- 严格 Clean9：50 对，仅 TCM-SE-B。
- 不能证明语义保持扰动的生成任务，统一鲁棒字段为 `N/A`，不事后设阈值。

严格 Clean9 的 50 对为 clean TSR、perturbed TSR、conditional retention 和
regression 四个主率同时报告分子/分母、百分比及双侧 95% Wilson score interval。
该区间方法无随机抽样，因此不需要 bootstrap seed；主表保留点估计，详细 JSON 和
Markdown 保留完整区间。

方法比较另固定 `CARE-OPD − clean-SFT`（并补充 `CARE-OPD − GKD`）的成对 item
bootstrap：10,000 次、seed=20260809。Perturbed TSR 与 conditional retention 的
95% CI 下界都必须大于 `-5 pp` 才通过联合非劣门槛；50 对样本较小，门槛未通过时只
报告不确定性，不把点估计方向写成“已泛化”。

官方 ED-B 的 ID 108、109 把第四个位置的选项前缀误写成了第二个 `C`。clean
prompt 保留官方字节；由于无法构造严格的双射置换，这两条从派生鲁棒集合排除，
并在 `task_manifest.json` 中登记。

## 固定依赖与数据

MTCMB 使用官方仓库 commit：
`faffd813c67fe012cef74277a23270d5dd6db9b8`。数据文件、prompt 源码、条数和
正式训练 overlap manifest 均做 SHA-256 校验。源 overlap audit 固定 SHA-256
`5f4e14b13427f12e6658cc1769687da857f66452e071b1e1eeabd6d4ac787968`；其统计为
direct provenance 1,062、exact normalized stem 1,096、near 2，联合排除 1,186。
官方仓库当前未提供
`TCM-FT` prompt builder 引用的 YAML，因此预注册固定 fallback
`mtcmb-ft-zero-shot-fallback-v1`；该项不能声称是逐字复现其缺失 prompt。为使六模型
配对可复现，本协议统一使用 greedy decoding，而非官方不同 backend 各自的默认采样；
因此这是固定的 MTCMB 数据/评分协议评测，不宣称逐配置复现原论文 leaderboard。

服务器准备示例：

```bash
cd /opt/care-workspace
git clone https://github.com/Wayyuanyuan/MTCMB.git
git -C /opt/care-workspace/MTCMB checkout faffd813c67fe012cef74277a23270d5dd6db9b8

/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python -m pip install \
  -r /opt/care-workspace/care_opd/public_eval/mtcmb_clean/requirements-eval.txt
```

BERTScore 必须指向已经完整下载的本地 `bert-base-chinese` 目录；正式运行时不允许
临时网络漂移。协议固定 `bert-score==0.3.13` 的官方中文默认第 8 层，模型目录及
BERT scorer 都会冻结文件指纹。根静态锁同时纳入 scorer 代码、依赖版本及本地
`bert-base-chinese` 完整目录 inventory/fingerprint；六模型 suite 与每个 run manifest
再次绑定同一指纹，形成根锁、suite、run 三层 fail-closed 校验。

## CPU 审计

```bash
cd /opt/care-workspace/care_opd

MTCMB_REPO=/opt/care-workspace/MTCMB \
/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python -m unittest -v \
  public_eval/mtcmb_clean/test_protocol.py

/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python \
  public_eval/mtcmb_clean/prepare_data.py \
  --mtcmb-repo /opt/care-workspace/MTCMB \
  --train-seed-file /opt/care-workspace/care_opd/data/qzbc_formal/seeds/qzbc_native_seeds.train.jsonl \
  --output /opt/care-workspace/care_opd/public_eval/mtcmb_clean/overlap_exclusions.json \
  --verify-only
```

`prepare_data.py` 会直接从官方数据与正式训练 seed 重算联合排除。若服务器另行复制了
源审计文件，可额外传 `--overlap-report /path/to/overlap_report.json`，并对其固定 hash
再次 fail closed；正式评测不依赖该临时文件路径。

## 正式六模型运行（物理 GPU 4、5、6、7）

必须先完成所有公开 benchmark 协议文件，再生成一次根目录
`PREREGISTRATION_LOCK.json`。启动器会调用锁生成器验证其内容；任一锁定文件、
模型指纹、数据指纹、prompt 或分析规则变化都会 fail closed。若已有不同版本的旧锁，
应使用新的版本化锁文件与新的 `OUT_ROOT`，不得覆盖旧结果。

```bash
cd /opt/care-workspace/care_opd

export PYTHON_BIN=/opt/care-workspace/venvs/qzbc_baseline_gpu4/bin/python
export SOURCE_SUITE=/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/model_suite.json
export PREREGISTRATION_LOCK=/opt/care-workspace/care_opd/public_eval/PREREGISTRATION_LOCK.json
"$PYTHON_BIN" public_eval/create_preregistration_lock.py \
  --project-root /opt/care-workspace/care_opd \
  --model-suite "$SOURCE_SUITE" \
  --bert-model /opt/care-workspace/models/bert-base-chinese \
  --output "$PREREGISTRATION_LOCK"

nohup env \
  PYTHON_BIN="$PYTHON_BIN" \
  MTCMB_REPO=/opt/care-workspace/MTCMB \
  TRAIN_SEED_FILE=/opt/care-workspace/care_opd/data/qzbc_formal/seeds/qzbc_native_seeds.train.jsonl \
  BERT_MODEL=/opt/care-workspace/models/bert-base-chinese \
  OUT_ROOT=/opt/care-workspace/care_opd/runs/public_eval/mtcmb_clean_v1 \
  PREREGISTRATION_LOCK="$PREREGISTRATION_LOCK" \
  PHYSICAL_GPUS=4,5,6,7 \
  NUM_WORKERS=4 \
  DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
  bash public_eval/mtcmb_clean/run_existing_gpu4_5.sh \
  > /opt/care-workspace/mtcmb_clean_gpu4_5_6_7.log 2>&1 &

echo $!
tail -f /opt/care-workspace/mtcmb_clean_gpu4_5_6_7.log
```

六个模型按 `frozen → clean_sft → recovery_sft → recovery_dpo → gkd_full →
care_opd` 顺序运行。每个 worker 每条预测 `fsync` 到 shard；中断后原命令恢复。
每个模型的生成阶段使用四个独立 worker 和逻辑设备
`cuda:0,cuda:1,cuda:2,cuda:3`；生成 worker 全部退出后，BERTScore 继续只在
逻辑 `cuda:0` 上运行。worker 数、逻辑设备与 `CUDA_VISIBLE_DEVICES` 物理映射
均进入 run fingerprint。
只有末行因进程中断而损坏时才原子截断末行，中间损坏、指纹不一致或重复 ID 均拒绝
续跑。完整模型由带 artifact SHA-256 的 `COMPLETE.json` 判定并跳过。

每个模型生成 7,256 个 clean 调用（LitData 的 100 条展开为 306 个 annotation
调用）和 4,862 个派生扰动调用，共 12,118 次；六模型共 72,708 次。

## 输出

默认根目录：

```text
/opt/care-workspace/care_opd/runs/public_eval/mtcmb_clean_v1/
```

关键文件：

```text
inference_items.jsonl
model_suite.json
official/<model>/predictions.jsonl
official/<model>/record_scores.jsonl
official/<model>/controlled_robustness_pairs.jsonl
official/<model>/summary.json
official/<model>/run_manifest.json
official/<model>/COMPLETE.json
official/suite_summary.json
official/suite_summary.csv
official/suite_summary.md
official/paired_comparisons.csv
official/primary_noninferiority.csv
```

`suite_summary.json.main_table_records` 恰含一个完整六模型 block：严格 Clean9 的
SE-B 50 对选项置换作为 derived robustness，并复用根
`UNIFIED_METRIC_SCHEMA.json` 的十个精确字段。其余官方 Full/overlap-clean/Clean9
异构任务指标按任务保存在 `models` 与各模型 `summary.json` 中；跨任务仅提供明确
标注为 descriptive 的等任务宏平均，不把 MSDD/PR/SE-B 样本微平均包装成统一
effectiveness 指标。
