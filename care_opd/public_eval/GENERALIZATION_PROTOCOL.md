# CARE-OPD 泛化评测协议

## 1. 要验证的主张

CARE-OPD 的目标不是只提高 QZBC 自建测试集，而是将受扰状态下的错误恢复能力写入可替换的 4B 基础模型。最终证据必须分别回答四个问题，不能把不同测试集的分数相加成一个总分。

| 证据层 | 问题 | 数据与主指标 |
|---|---|---|
| L1 域内案例泛化 | 未见病例上能否恢复 | identity-bound QZBC grouped rerun；pair success、unsafe green、clean retention（旧结果仅 retrospective；当前 split 是 entity-level，不是严格来源族/OOD） |
| L2 未见故障泛化 | 未见故障族/模板族上能否恢复 | 整族留出的扰动集；OOD recovery、clean-to-OOD drop、paired McNemar |
| L3 公开任务泛化 | 是否锁死 QZBC 格式或遗忘领域能力 | 下列三个公开 benchmark，各自独立报告 |
| L4 系统转化 | 换入原 QZBC Agent 后是否改善 | 固定 Agent/知识库/规则，仅替换基础模型的端到端 A/B |

## 2. 三个公开 benchmark 的互补分工

| Benchmark | 主问题 | 主结果 | 不应声称 |
|---|---|---|---|
| TCMEval-PA v3 | 复杂 QZBC JSON 后训练后，处方审核知识与选项决策是否保留 | 328 题 option-set exact；单/多选与规范性/适宜性分层 | 不能单独证明工具错误恢复 |
| TCMBench V2 robustness track | 面对公开临床推理题的重排、逆序和难例扰动是否稳定 | clean/perturbed accuracy、paired consistency、robustness drop；任务级 macro/micro | 若采用 answer-only 协议，不与官方 CoT 数字直接横比 |
| MTCMB provenance-clean Clean9 | 去掉训练直接使用的 ED-A/ED-B 整个任务家族后，能力能否迁移到其他中医任务 | 9 个任务各自官方指标；SE-B 50 对选项置换的统一 TSR/retention | 不把异构生成指标强行平均成一个“总分” |

MTCMB-Full 不进入独立泛化主表：正式训练数据直接使用了 `SRC_MTCMB`，已有审计发现公开记录中 1,062/7,100 与训练来源直接重合，且全部位于 ED-A/ED-B。主分析删除这两个完整任务家族及依赖远程主观 judge 的 SE-A，得到预先固定的 **Clean9（1,050 条）**；Full 与逐条 overlap-clean 仅作透明诊断。Clean9 的异构官方指标分任务报告，只有可验证语义保持映射的 SE-B 50 对进入统一鲁棒性主表。

这两个新增公开集的取舍是有意的：TCMBench V2 回答“恢复/鲁棒行为能否迁移到公开中医推理扰动”，MTCMB-Clean9 回答“去除直接训练来源家族后，是否仍保持多任务中医能力”。CMB-Exam 11,200 题可作为可选的广域中文医学知识保持附录，但不参与两套新增主 benchmark 的结论门槛。

TCMBench 官方仓库当前未提供明确的数据再分发许可文件，因此评测代码只记录官方仓库版本和本地文件哈希，不把完整数据复制进比赛代码包；正式发布前应向作者确认授权。MTCMB 使用固定官方 commit，并按其数据/代码许可与引用要求使用。

## 3. 统一的有效性与鲁棒性指标

所有任务先转成逐样本二元 `Task Success`：

- QZBC：正式 verifier 全部通过记为 1；
- 选择/多选任务：预测选项集合与金标准完全一致记为 1；
- Agent 端到端任务：预注册的任务完成与安全约束同时满足记为 1。

在每个有干净—扰动配对的测试轨上统一计算：

| 指标 | 定义 | 方向 |
|---|---|---:|
| Clean TSR | 干净样本的 Task Success 均值 | ↑ |
| Perturbed TSR | 语义保持扰动后 Task Success 均值 | ↑ |
| Conditional Retention | `P(perturbed success=1 | clean success=1)` | ↑ |
| Robustness Drop | `Clean TSR - Perturbed TSR`，百分点 | ↓ |
| Prediction Consistency | 映射回规范答案空间后，干净/扰动预测完全一致 | ↑ |
| Worst-perturbation TSR | 多种扰动中最低的 TSR | ↑ |

每条配对轨只设两个主终点，避免在多个指标里挑最有利的一个：

- **主有效性终点**：`Perturbed TSR`，回答扰动后任务是否仍做对；
- **主鲁棒性终点**：`Conditional Retention`，等价地同时报告
  `regression = 1 - retention`，回答原本做对的任务有多少被扰动破坏。

`Clean TSR` 是能力锚点；`Robustness Drop`、`Prediction Consistency` 和
`Worst-perturbation TSR` 是诊断指标，不能用来挽救两个主终点失败的结论。
此外必须报告 `clean-correct -> perturbed-wrong` 与
`clean-wrong -> perturbed-correct` 的人数，避免弱模型因大量“两边都错”而显得稳定。

公开 benchmark 的官方原始结果放在 **Official** 分栏。项目额外构造的选项重排、
格式变化等统一标为 **Derived / `controlled_robustness_v1`**，保存变换映射、种子与
哈希，不得冒充官方 leaderboard 分数。TCMBench 的 `SDT`、`SDT_reverse`、
`SDT_shuffle` 先通过稳定标识与内容审计；只有映射无歧义的项目才进入配对统计，
项目生成的 pair map 仍标为 Derived，官方各子任务准确率单独保留。

不同 benchmark 的难度、扰动强度与构念不同。主结论不使用跨 benchmark 平均分；
若展示等权宏平均，只能放在附录作为描述性导航，不能作为方法排名、显著性检验或
“具有泛化性”的门槛。

在查看本轮派生扰动预测之前，CARE 相对两个主要参照（共享 warm-start 的
`clean_sft` 与最强通用蒸馏参照 `gkd_full`）冻结以下联合非劣门槛：

- TCMEval-PA controlled：按规范化题干聚类、10,000 次配对 bootstrap（seed
  `20260808`）；`Perturbed TSR` 和 `Conditional Retention` 的 CARE-minus-baseline
  95% CI 下界都必须大于 **-0.02**；
- MTCMB Clean9 SE-B：50 个题目单位、10,000 次配对 item bootstrap（seed
  `20260809`）；上述两个主终点的 95% CI 下界都必须大于 **-0.05**；
- TCMBench V2：沿用该轨冻结的 **-0.02** 联合非劣门槛与 case-clustered 配对分析。

门槛通过只表示在相应公开轨上“保持/非劣”，不表示 CARE 显著优于参照。每个
benchmark、每个 baseline 分别判定，不做跨 benchmark 合并，也不能用一个终点的优势
抵消另一个终点失败。

## 4. 主表组织

主表以“方法 × benchmark × track”为一行，用同一组字段，但 Official 与 Derived
不得混在同一数值列：

| 方法 / Benchmark | Track provenance | Clean TSR | Perturbed TSR（主有效性） | Retention（主鲁棒性） | C→W / W→C | Drop | Consistency |
|---|---|---:|---:|---:|---:|---:|---:|
| CARE-OPD / QZBC grouped test | Internal identity-bound rerun + retrospective result-blind family-clean sensitivity | | | | | | |
| CARE-OPD / TCMEval-PA | Official-clean | | N/A | N/A | N/A | N/A | N/A |
| CARE-OPD / TCMEval-PA | Derived-controlled | | | | | | |
| CARE-OPD / TCMBench V2 | Official tasks + audited derived pair map | | | | | | |
| CARE-OPD / MTCMB-Clean9 SE-B | Filtered official + derived option permutation | | | | | | |

完整六方法表使用相同结构。公开集同时给出：完整官方集合、精确题面重叠排除集合、
样本数、95% 置信区间、与 CARE 的配对差异。精确排除集合不称为“无污染”，因为它
不能排除释义、共享题库或知识级重叠。若两种口径结论不同，以预先冻结的重叠排除
集合解释直接题目迁移，并完整保留官方集合结果。

## 5. 当前 TCMEval-PA 的定位

当前 `official` 结果中，CARE-OPD 与 full GKD 均为 211/328（64.33%）。CARE-OPD 相比 frozen 多答对 5 题，但配对 McNemar `p=0.473`；相比 GKD 为 16 胜、16 负，`p=1.0`。因此当前结论仅是：

> CARE-OPD 在显著增强 QZBC 受控恢复表现的同时，在独立处方审核任务上保持了竞争力并取得并列最高观察值，尚无证据表明其显著提升通用处方审核知识。

还应补跑 `label_blind` 协议，避免只依赖由答案基数恢复的单选/多选类型提示。

## 6. Frozen 与训练血缘硬门槛

`frozen` 在 launcher 中默认指向 `/opt/care-workspace/models/Qwen3.5-4B`，其设计含义是“未做任何本项目后训练的 Qwen3.5-4B”。但汇总标签本身不能证明身份，环境变量可以覆盖路径。

发布结果前必须核验服务器上的：

```text
/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/model_suite.json
/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/official/frozen/run_manifest.json
/opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/official/frozen/COMPLETE.json
```

只有当 canonical path、权重/config/tokenizer/chat-template 指纹与独立保存的、不可变
`Qwen/Qwen3.5-4B` revision 参考一致，且没有 adapter/继续训练痕迹，才称为
**raw initial base**。若没有这份参考，即使套件内部所有哈希一致，审计也只能输出
`origin_status=NOT_VERIFIED`，不能用 `frozen` 标签反向证明身份。即使文件与本地提供的
不可变 revision 参考一致，审计也只输出 `REFERENCE_MATCHED`；除非另有独立来源证据，
不会让自写 JSON 自证为 raw initial。

另外区分：

- Raw initial base：当前 `frozen` 意图代表的原始 HF 模型；
- OPD initial policy：OPD 前的 warm-start，即 clean SFT；
- Initial deployed model：原 QZBC 使用的 Ollama/GGUF `qwen3.5:4b`，它可能有不同量化与 chat template。

要隔离“OPD 本身”的贡献，主要比较 CARE-OPD 与同一份 clean-SFT warm-start；CARE 与 frozen 的差异包含 warm-start SFT 和 OPD 的累计影响。

还必须比较以下两个目录的完整模型指纹，不能只比较 `config_sha256`：

```text
/opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged
/opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/warm_start/merged
```

若二者权重、tokenizer 或配置不同，CARE 与 GKD/SFT 的差异会混入独立 warm-start 的随机差异。正式方法比较应从同一只读 checkpoint 分叉。

正式总入口会在生成静态锁和启动 GPU 前自动执行完整血缘审计（会读取并哈希模型
权重）。若需单独运行，`origin reference` 必须按“存在才传入”的方式处理，并接受
`rc=3` 表示本地闭合但上游来源未获独立证明：

```bash
ORIGIN_REFERENCE=/opt/care-workspace/care_opd/runs/public_eval/qwen35_4b_origin_reference.json
origin_args=()
[[ -s "$ORIGIN_REFERENCE" ]] && origin_args+=(--origin-reference "$ORIGIN_REFERENCE")

set +e
python3 /opt/care-workspace/care_opd/public_eval/verify_model_lineage.py \
  --suite /opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3/model_suite.json \
  --eval-root /opt/care-workspace/care_opd/runs/public_eval/tcmeval_pa_v3 \
  --prompt-mode official \
  --expected-frozen /opt/care-workspace/models/Qwen3.5-4B \
  --care-warm-start /opt/care-workspace/care_opd/runs/qzbc_base_model_3gpu/warm_start/merged \
  --baseline-warm-start /opt/care-workspace/care_opd/runs/qzbc_baselines_3gpu/warm_start/merged \
  "${origin_args[@]}" \
  --rehash-suite \
  --output /opt/care-workspace/care_opd/runs/public_eval/model_lineage_audit.json
rc=$?
set -e
[[ "$rc" -eq 0 || "$rc" -eq 3 ]] || exit "$rc"
python3 - <<'PY'
import json
p = "/opt/care-workspace/care_opd/runs/public_eval/model_lineage_audit.json"
d = json.load(open(p, encoding="utf-8"))
assert d["local_lineage_status"] == "PASS", d["errors"]
assert d["warm_start_content_identical"] is True
print("local lineage PASS; origin status:", d["origin_status"])
PY
```

`qwen35_4b_origin_reference.json` 至少包含完整 40/64 位十六进制不可变 `revision` 与该独立快照的
`model_fingerprint`；其文件哈希应进入预注册锁。审计 v2 会从磁盘重新计算六个模型的
inventory/model/suite 指纹、六个评测 run contract、两份 warm-start 内容，以及每个
训练分支的 `export_manifest.base_model` 链。未使用 `--rehash-suite` 时只得到
`PATH_NAME_SIZE_ONLY`。本地自备 origin reference 只能闭合到 `REFERENCE_MATCHED`，不会
单独产生“raw initial 已独立验证”的 `PASS`。

## 7. 预注册锁

在查看新增公开轨预测之前，用冻结的六模型套件生成只读
`public_eval/PREREGISTRATION_LOCK.json`。它是**静态协议/模型锁**：逐文件锁定 evaluator、
统一 schema、数据清单、排除规则、模型加载代码和正式总入口，并锁定六模型的内容
fingerprint。正式总入口显式固定 prompt/track/generation 参数；每个 run manifest 再绑定
实际数据、映射、运行库版本和运行合同。锁文件 SHA-256 必须写入每个 run manifest。
这不等价于第三方时间戳服务；若需称“外部可审计预注册”，还应把锁哈希在预测前提交到
不可回写的外部记录。任何代码或协议变化必须使用新锁与新输出根，不能覆盖旧结果。

## 8. 结论门槛

只有同时满足下列条件才写“具有泛化性”：

1. 带 checkpoint fingerprint 与当前 split-integrity manifest 的 grouped QZBC 重跑上，CARE 显著提高恢复能力；现有 legacy 行不计入该门槛，并明确该证据仍是同模板/同任务分布的域内机制测试；
2. 整族未见故障/模板上仍保持预注册的正向配对增益；
3. 至少两个独立公开 benchmark 的主终点通过预先定义的配对非劣门槛；
4. TCMBench 审计后的扰动对上，Perturbed TSR 与 Conditional Retention 两个主终点
   均通过门槛；drop/consistency 只能用于解释，不能替代主终点；
5. 回接固定 QZBC Agent 后，端到端成功率提高且 unsafe-green 通过预注册的安全
   非劣门槛；
6. frozen、warm-start 与所有分支的模型指纹闭合。

在完成这些门槛前，使用“初步跨 benchmark 支持”而不是“已证明普适泛化”。
