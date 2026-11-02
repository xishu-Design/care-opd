# 方法：qzbc-native CARE-OPD

## 目标

学习一个完整、可独立部署的 `Qwen3.5-4B` checkpoint，用它替换 qzbc
中的 `OLLAMA_MODEL`。qzbc 的 Agent Graph、MCP 工具、规则和 Safety
Guard 不参与参数训练，也不被替换。

qzbc 的真实模型接口为：

```text
qzbc chat messages(Evidence Packet) → GeneratedResponse JSON
```

模型不输出工具 action，也不负责 route/query。早期
`CareRecoveryEnv.Action` 不属于正式训练目标。

## 在线样本

每条 seed 先由 `qzbc_contract.build_qzbc_messages` 使用当前 qzbc 源码中
的 system prompt 编译。seed 的业务/病例背景为受控合成内容，领域知识
来自 qzbc 已审校且带来源标识的药材、方剂、配伍、患者科普和教学资产；
同一知识条目不会跨 train/dev/test。正式输入状态包括：

- `clean`
- `bad_history`
- `source_outage`
- `evidence_conflict`

当前学生从这些消息在线生成 `GeneratedResponse`。宿主 verifier 检查：

- JSON 与字段类型；
- qzbc 角色安全边界；
- `source_id` 必须来自当前 Evidence Packet；
- seed 的 `must_contain`、`must_not_contain` 和必需来源；
- 故障状态下是否明确不确定性、警告、下一步和免责声明。

## 因果局部化

失败被投影到输出组件：

```text
title
summary
sections
warnings
next_actions
disclaimer
```

再将组件映射到学生原始 JSON 的字符 span 和 token span。无法安全解析
时才显式退化为 full-response mask，而不是静默丢弃样本。

教师看到同一个学生序列，以及 verifier 给出的失败类型、局部字段和
已验证 oracle response；学生不看到这些特权信息。损失为：

```text
L = λ_opd · Σt m_span(t) m_trust(t) w_risk D_JS(Tt || St)
  + λ_oracle · CE(verified response on repaired fields)
  + λ_keep · L_retention(outside repaired fields)
```

- `m_span`：失败字段 token mask；
- `m_trust`：过滤教师和学生分布差异过大的 token；
- `w_risk`：clean=1、错误历史=2、证据中断/冲突=4；
- retention 约束修复区域外的原模型行为；
- 教师与学生必须使用完全相同的 tokenizer/vocabulary。

每轮将 LoRA 合并回完整模型，再用新模型重新采样。最终导出不含
adapter 文件的 standalone HF checkpoint。

## 方法 insight

创新点不是“给 Agent 加一个恢复模块”，而是把垂直模型的失败监督从
整段答案缩到真正受证据故障影响的结构化字段，并把风险非对称性直接
放入 on-policy token 蒸馏。它针对 full-response SFT/OPD 的两个不足：

1. 修复少数字段时仍更新整段正确内容，容易过修复与遗忘；
2. 把无害格式偏差和可能导致无依据医疗结论的证据错误等权处理。

部署时不增加推理步骤、反思模型或 recovery bank；能力被内化到原
4B 底座参数中。
