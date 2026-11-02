# 最相关 baseline

正式比较必须在相同 `Qwen3.5-4B` 初始化、qzbc seed split、当前策略
采样数、teacher、训练 token 上限、Ollama 量化和解码参数上重跑。其他
论文的公开数字只能说明方法背景，不能直接填入本项目表格。

| 方法 | 本项目实现 | 区分的因素 |
|---|---|---|
| Frozen base | `qwen3.5:4b` | 不做后训练 |
| qzbc clean SFT | `warm_start/merged` | 领域格式/任务适配 |
| Recovery-SFT | `baselines/recovery_sft/merged` | 看到故障 target，但非 on-policy |
| Recovery-DPO | `baselines/recovery_dpo/merged` | pair-level 偏好，不使用 token teacher 分布 |
| GKD/full OPD | `baselines/gkd_full/final_model` | 同一学生 rollout 上整段蒸馏 |
| trust-region full OPD | `baselines/tropd_full/final_model` | 教师可靠性 mask，但仍更新整段 |
| qzbc CARE-OPD | `final_model` | 字段局部化 + 风险权重 + retention + 多轮重采样 |

最直接的方法基线是 GKD/full-response OPD，因为它同样使用学生的
on-policy 序列和教师逐 token 分布。Recovery-SFT 与 DPO 分别排除
“只是增加故障数据”和“只是换成偏好训练”两种解释；trust-region full
OPD 排除收益仅来自教师可靠性筛选。

## 与已有工作的对应

| 已有工作 | 本项目中的最接近对照 | 仍缺少的本方法因素 |
|---|---|---|
| [Generalized Knowledge Distillation / on-policy GKD](https://arxiv.org/abs/2306.13649) | `gkd_full` | 不按 qzbc 失效字段和风险局部加权 |
| [Agent-R](https://arxiv.org/abs/2501.11425) | `recovery_sft` + 多轮重采样对照 | 以步骤修复/SFT 为主，不是原生响应字段上的稠密 OPD |
| [PALADIN](https://arxiv.org/abs/2509.25238) | `recovery_sft` | 离线恢复轨迹与推理检索，不把能力无额外模块地蒸馏进当前 4B 响应模型 |
| [Trust Region On-Policy Distillation](https://arxiv.org/abs/2606.01249) | `tropd_full` | 过滤不可信教师 token，但不表达语义字段责任和医疗风险非对称 |

这些工作用于确定方法轴和复现实验，不把其论文数字搬入比赛主表。领域
能力保持可另用 TCMBench/TCMD 等独立中医药评测，但它们不是错误恢复
训练方法 baseline。

建议主表至少报告：

- qzbc-native held-out pass rate；
- `bad_history`、`source_outage`、`evidence_conflict` 分项恢复率；
- invented/missing source-id rate；
- role-forbidden/unsafe content rate；
- JSON/schema validity；
- clean split retention；
- qzbc 内置 12 例 LLM provider/model 集成通过率、fallback 和延迟；
- 可选的真实 qzbc Agent Graph 双后端部署完整性与延迟；
- trained wins/losses/ties 与配对统计。

运行：

```bash
bash scripts/02_train_baselines.sh
bash scripts/06_evaluate_hf_baselines.sh
```

`06` 给出所有 checkpoint 在相同未量化 HF 解码下的快速配对前指标。
比赛主表仍应把候选模型转换为相同 Ollama/GGUF outtype 后，用
`qzbc_native_eval` 重跑，以排除量化差异。

不建议把 Reflexion、额外 verifier Agent 或推理时 recovery bank 放入
主表，因为它们改变了部署 Agent 和推理预算；可作为附加系统比较，并
单独报告额外 token、延迟和模型调用。
