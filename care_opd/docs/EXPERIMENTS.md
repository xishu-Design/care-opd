# 实验协议

## 数据切分

`qzbc_seed_builder` 按 8 个 qzbc task type 分层产生 train/dev/test。
每个 seed 具有稳定 ID、数据来源声明、qzbc contract fingerprint 和
可执行 verification。只用 train 做 SFT/OPD；dev 用于选择 checkpoint
和阈值；test 仅做最终配对报告。

知识资产先于样本生成完成 exact-entry split；同一药材、方剂、配伍、
饮食或 approved-training-only 题库条目不得跨集合。manifest 的
`grounding_asset_split.overlap_check` 必须为 `PASS`。

每个 model-visible seed 被编译成：

- clean；
- bad history；
- source outage；
- evidence conflict。

`doctor_review_text` 因 qzbc 会用宿主结构化结果覆盖模型输出，只用于
格式/能力保持，不作为主要提升来源。

## 公平性

- 所有训练型方法从同一个 clean-SFT merged checkpoint 开始；
- GKD、trust-region OPD 和 CARE-OPD 使用同一 teacher/tokenizer；
- baseline 的第一轮使用同一 on-policy collection；
- CARE-OPD 每轮更新后必须重新采样；
- base 与 trained 的 Ollama outtype、上下文、温度和输出长度一致；
- `LLM_ALLOW_FALLBACK=false`。

## 主评测

`qzbc_native_eval` 对同一 test request 顺序运行 base 和 trained，使用
与 qzbc provider 相同的 messages、JSON schema 和解码参数。逐例报告
verifier 失败类型，并做配对 wins/losses/ties 与 exact McNemar。

`qzbc_bridge compare` 在不修改 qzbc 的情况下复用
`POST /api/llm/experiments/run`。它只有 12 例，因此是集成检查而不是
主要科学证据；该接口直接调用 `LLMService`，不执行 Agent Graph、工具、
检索或记忆链路。

`qzbc_agent_ab` 要求两个从相同源码启动的 qzbc 进程，并隔离两臂的
SQLite/Graphiti/Mem0 可写状态。它通过真实
`RoleAIChatGraph/PatientEducationGraph` 接口核验部署路径与模型来源，
但不替代 held-out 模型质量结果。

## 必要消融

正式比赛报告建议包含：

1. full response mask；
2. 去掉 risk weight；
3. 去掉 teacher trust mask；
4. 去掉 clean retention；
5. 单轮而非多轮 recollection；
6. 只用 hard oracle CE，不用 dense OPD。

不要把合成研究结果表述为临床安全、诊断效果或真实患者收益。
