# 竞赛补充证据工具包

本目录用于补齐项目文档所需的统计证据、专家盲评和“同一病例、三个角色”受控实验。它是内部开发与复核入口，不是正式上传包；病例、任务、模型回答和评审意见只保存在名称含 `.private` 的文件中。

## 当前状态

| 证据模块 | 当前状态 | 可公开结论 | 尚需完成 |
| --- | --- | --- | --- |
| 数据提供方真实医疗记录双轨统计 | 已完成 | 可报告聚合指标、配对差值与 95% CI | 按项目文档篇幅选择主表与补充表 |
| 三方法专家双人匿名盲评 | 任务包已冻结 | 暂无专家结果 | 两名独立评审填写评分表后运行汇总 |
| 同一病例三角色受控实验 | 任务包已冻结 | 可报告实验设计与样本构成，不能报告角色效果 | 在隔离后端生成 72 条回答，再完成两轮双人评审 |
| 四系统专家首选盲审 | 四端点执行计划已固定，尚未推理 | 可报告四系统、72 个任务和盲审协议，不能报告胜率 | 获得逐条远程处理授权后生成 288 份回答，再由独立专家评分 |
| TCM-BEST4SDT 30例开放病例盲审 | 题单、重叠审计与本地四系统脚本已冻结，尚未推理 | 可报告公开数据来源、30例预注册协议与四系统配置，不能报告胜率 | 在卡3生成120份回答并完成专家盲选 |
| 真实数据来源说明 | 模板已准备 | 可表述为数据提供方提供的真实医疗业务记录，经过去标识化与结构化 | 由数据提供方或负责人核对并签署；未签署前不得写成已认证 |
| 三训练种子 | 不纳入本轮补充 | 不报告多训练种子稳定性 | 用户已明确取消，避免占用赛前时间 |

## 数据口径

- 数据提供方共提供 72 组真实医疗记录；数据经过去标识化与结构化，并非合成数据。
- Structured 轨纳入 65 组，Raw strict 轨纳入 54 组；54 组 Raw strict 全部包含在 65 组 Structured 中，两轨不能相加为 119 组。
- 真实数据只用于独立外部评测，不进入训练。

## 1. 真实医疗记录统计

入口脚本：`real_external_stats.py`

测试：`../tests/test_real_external_stats.py`

已生成的聚合产物位于：

`../../competition_project_document/formal/competition_submission_v8_candidate/build_internal/competition_evidence/real_external_stats/`

其中：

- `SUMMARY.md`：评审友好的结论摘要；
- `real_external_report_metrics.csv`：六模型、双轨、各状态的报告级指标；
- `care_vs_all_references.csv`：CARE 与五个固定对照方法的完整比较矩阵及报告族配对 Bootstrap 区间；
- `cohort_audit.csv`：72/65/54 的队列关系审计；
- `real_external_stats.json`：机器可读的聚合结果。

默认统计模式为 `legacy_frozen`：使用原公开协议的 Python RNG、10,000 次报告族成对聚类 Bootstrap 和种子 20260819，并在同一模型比较内让所有范围与指标复用同一组报告族抽样。该模式已逐项复现 V7 中 52 个重叠总体端点。另一模式 `independent_stable` 必须显式选择，输出会标记为非权威补充重算，不能与冻结区间混写。

正文建议把 Structured 作为主外测视图，并以六模型同表呈现 clean 质量与异常恢复的权衡，避免对不同端点切换最有利的对照方法。Raw strict 是诊断性压力轨：必须同时报告低绝对值及其接口约束，不与 Structured 混排成单一排行榜，也不作为“总体最优”证据。具体写法见 `COMPETITION_REPORTING_GUIDE.md`。

## 2. 三方法专家盲评

入口：

- `prepare_blind_review.py`：冻结抽样、匿名候选和评分模板；
- `summarize_blind_review.py`：完成度校验、胜/平/负、安全正确率、精确 McNemar、报告族 Bootstrap 和评审一致性；
- `BLIND_REVIEW_PROTOCOL.md`：完整操作说明。

已冻结包：

`../runs/private_real_thyroid_external/blind_review_v1/structured_three_method_preregistered_n24/`

该包从 65 个 Structured 报告族中按固定种子简单随机抽取 24 族，覆盖 clean、source-outage 和 evidence-conflict 三种状态，对 CARE、Clean-SFT 和 GKD-Full 做匿名比较。当前评分表为空，因此不存在可公开的专家胜率或一致性结果。

## 3. 同一病例、三个角色受控实验

入口：

- `prepare_three_role_eval.py`：固定抽样并生成任务；
- `execute_three_role_eval.py`：只读计划检查或在隔离后端执行；
- `score_three_role_eval.py`：生成两轮评审包、自动审计并汇总双人评分；
- `THREE_ROLE_REAL_CASE_RUNBOOK.md`：完整运行说明。

唯一有效的冻结包：

`../runs/private_real_thyroid_external/three_role_eval/paired_fact_n12/`

设计为 12 组真实记录（9 组病理、3 组超声）× 2 个证据条件 × 3 个实际系统角色，共 72 条回答。每个病例/条件下，医生、患者和学生收到字节级相同的用户消息，只改变真实角色路由、角色系统指令与权限链。

两个证据条件为：

- `clean`：病例材料具有可核验检索证据资格；
- `evidence_unavailable_counterfactual`：保留同一病例事实文本，但将其明确降为未核验用户陈述，并把检索证据资格设为 0。

第二个条件是透明命名的配对反事实，不应写成“直接复现原始 source-outage”。这样可以把降级行为归因于证据资格变化，而不是病例内容变化。旧 `structured_n12/` 已标记为内部废弃，不得用于推理、统计或提交。

第一轮评审隐藏角色标签，检验角色能否被正确辨识、三份回答是否共享事实、是否出现跨角色矛盾及患者端诊疗式泄漏；第二轮显示目标角色，评估事实忠实度、角色适配、权限边界、证据处理和降级行为。学生侧只要求可核验的依据分解和学习反馈，不索取模型隐藏思维过程。

回答生成完成后，`prepare-review` 会在没有人工评分的情况下先输出一份自动审计摘要，包含事实锚点、异常数字、边界/降级短语及后端 Evidence Packet、回退与 trace 的聚合运行信号；这些内容只用于定位问题，不能替代医学判断。评审包使用不可从可见 bundle ID 反推的随机 A/B/C 映射，并为两位评审分别生成独立评分表；PRIVATE manifest 用哈希锁定任务、回答和角色密钥，汇总时不匹配即拒绝生成结果。

当前状态为 `PLAN_PREPARED_NO_MODEL_OUTPUTS`。只有在专用 PRIVATE 数据库、外部记忆和遥测关闭、模型与参数锁定后，才可按运行说明执行。执行时必须显式启用并认证后端的 PRIVATE 证据资格门禁：`clean` 只接收与任务哈希绑定的一条锁定材料，反事实条件强制零条检索证据；普通产品路由默认不启用该能力。当前不得把设计指标写成实验结果。

## 4. 四系统专家首选盲审

该实验比较“神农三言完整项目”与三个 API 对照智能体在同一冻结任务上的端到端回答。默认对照为 Cloudflare Workers AI、Groq 和 Cerebras；三者都通过 OpenAI-compatible Chat Completions 适配器接入，不把直接模型评测结果冒充 Agent 回答。

入口：

- `execute_four_agent_eval.py`：四系统统一执行器；默认只做计划检查，只有显式 `--execute` 才发起请求；
- `four_agent_endpoints.free.example.json`：四套固定端点、模型版本与环境变量名；不保存任何 API key；
- `prepare_agent_preference_review.py`：核验四套回答的全因子覆盖与任务哈希，生成逐专家独立 A/B/C/D 盲表；
- `summarize_agent_preference_review.py`：冻结评分后解盲，按病例族 Bootstrap 汇总安全合格专家首选份额；
- `FOUR_AGENT_EXPERT_BLIND_RUNBOOK.md`：从注册 API 到结果汇总的操作说明。

该实验支持的结论是“专家在指定日期、指定任务和指定四系统中更偏好哪一个完整项目的回答”。它不单独证明 CARE-OPD 的因果增益，也不构成临床疗效或临床验证；CARE-OPD 方法效果继续由同栈受控实验论证。

数据提供方记录虽经过去标识化与结构化，仍不能据此推定允许向第三方云服务逐条传输。执行器要求显式远程处理确认，但确认字符串不能代替数据提供方授权。Groq 还应先在账户中启用 Zero Data Retention；Cloudflare 与 Cerebras的数据条款也应在冻结实验前留档。Gemini 免费层会将内容用于改进产品，因此只列为公开或合成病例的备选，不进入真实记录默认配置。

### 公开中医开放病例本地四系统轨

`tcm_best4sdt_agent30/` 是不传输真实医疗记录的补充实验：从 TCM-BEST4SDT Figshare
v8 的300例开放病例中，以固定种子和疾病标签多样性规则预注册30例，在本地依次运行
神农三言完整项目、HuatuoGPT-3-8B、GLM-4-9B-0414 与 InternLM3-8B-Instruct。
模型仅接收病例正文与统一六段式医生任务，上游参考答案和疾病标签不进入推理包。

每位专家、每道题均使用独立 A/B/C/D 映射，主指标为安全合格专家首选份额；并列分票，
四答均不可接受保留在分母，病例级10,000次 bootstrap 给出区间。该轨支持完整项目的
端到端专家偏好比较，不是 TCM-BEST4SDT 官方全量成绩，也不能单独归因 CARE-OPD。

## 5. 来源说明与正文使用

- `REAL_DATA_SOURCE_ATTESTATION_TEMPLATE.md`：数据提供方真实医疗数据来源说明模板；
- `COMPETITION_REPORTING_GUIDE.md`：主文、补充材料和答辩中的安全表述边界。

正式材料可明确写明“数据提供方提供的真实医疗业务记录，经过去标识化与结构化”。不要将去标识化误写成合成，也不要将来源说明模板误写成已经签章的证明。

## 最短执行顺序

1. 直接采用已完成的真实医疗记录聚合统计；
2. 确认病例级远程处理授权，再生成四系统回答；没有授权时改用获批本地端点；
3. 完成四系统端到端专家首选盲审；
4. 对神农三言回答完成角色隐藏三联审查与角色明示逐答评分；
5. 两名评审全部提交后运行汇总脚本；
6. 仅将通过隐私检查的聚合结果写入下一版项目文档和辅助材料。
