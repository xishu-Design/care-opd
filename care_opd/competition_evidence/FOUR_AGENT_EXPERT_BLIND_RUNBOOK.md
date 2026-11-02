# 四系统专家匿名首选实验运行说明

## 1. 实验回答什么问题

本实验把同一份冻结任务交给四个端到端系统：神农三言完整项目，以及 Cloudflare Workers AI、Groq、Cerebras 三个 API 对照智能体。每位专家在不知道答案来源的情况下逐题选择最佳回答。

可报告结论是：在指定日期、指定任务、指定四系统的匿名比较中，专家对神农三言完整项目回答的安全合格首选份额。该结果用于论证项目整体优越性，不单独拆分 CARE-OPD 的因果贡献，也不等同于临床疗效验证。

冻结任务共 72 个：12 个真实病例族 × 2 个证据条件 × 医生、患者、学生 3 个角色。四系统共应产生 288 份回答。

## 2. 默认三个免费 API

| 对照智能体 | 固定模型 | 官方免费口径 | 本轮调用量 | 数据设置 |
| --- | --- | --- | ---: | --- |
| Cloudflare Workers AI | `@cf/zai-org/glm-4.7-flash` | Free 计划每日 10,000 Neurons | 72 | Cloudflare 声明不以 Customer Content 训练或改进服务 |
| Groq | `qwen/qwen3.6-27b` | Free 计划 30 RPM、1,000 RPD | 72 | 执行前必须在 Data Controls 中启用 Zero Data Retention |
| Cerebras | `gpt-oss-120b` | Free Trial 5 RPM、每日 1M tokens | 72 | 官方条款不授予以 Service Content 训练或微调模型的权利 |

三家均提供 OpenAI-compatible Chat Completions 接口。免费额度、模型和服务政策可能变化，正式运行当天应保存官方页面 PDF 或网页快照，并保存本配置文件的 SHA-256。

Gemini 免费层虽可公开注册并兼容 OpenAI 接口，但官方条款说明免费层内容可用于改进产品，并明确提示不要提交敏感、机密或个人信息，因此不进入数据提供方真实记录的默认配置。它只适合公开或合成病例的预演。

## 3. 远程处理授权是执行前置条件

数据提供方数据是真实医疗业务记录，经过去标识化和结构化；这不自动等同于获准把病例级内容发送给第三方云服务。只有同时满足下列条件才运行真实任务：

1. 数据提供方或项目负责人明确确认允许病例级材料发送至列明的三家处理方；
2. 团队确认竞赛、伦理和适用的数据跨境/地域要求；
3. 已核对并留存三家当日的数据留存、训练使用、日志和地域政策；
4. Groq 已启用 Zero Data Retention；
5. 神农三言后端使用专用 PRIVATE 数据库，外部记忆、遥测与自动回退均已关闭。

脚本中的确认字符串只是防止误操作，不能代替上述授权。若条件不满足，改用获批的本地或内网对照端点；不要把公开/合成病例实验写成真实医疗数据实验。

## 4. 注册并设置环境变量

不要把 API key 写入 JSON、命令行、聊天记录或提交包。以下命令在运行脚本的同一个 PowerShell 会话中执行：

```powershell
Set-Location /opt/care-workspace\care_opd
$EvalPython = "..\.hf_upload_venv\Scripts\python.exe"

$env:CLOUDFLARE_ACCOUNT_ID = "<Cloudflare Account ID>"
$env:CLOUDFLARE_API_TOKEN = "<仅授予 Workers AI 调用权限的 Token>"
$env:GROQ_API_KEY = "<Groq API Key>"
$env:CEREBRAS_API_KEY = "<Cerebras API Key>"

$env:SHENNONG_BACKEND_PROVIDER = "<以 /api/llm/health 返回值为准>"
$env:SHENNONG_BACKEND_MODEL = "<以 /api/llm/health 返回值为准>"
$env:CARE_THREE_ROLE_PROVISION_SECRET = "<至少 32 字符；与隔离后端同一环境>"
$env:CARE_THREE_ROLE_BACKEND_EVAL_KEY = "<至少 32 字符；与隔离后端 PRIVATE gate 一致>"
```

API key 获取入口：

- Cloudflare Dashboard：创建 Workers AI 可用的 API Token，并读取 Account ID；
- Groq Console：创建 API key，然后进入 Data Controls 启用 ZDR；
- Cerebras Inference Cloud：创建 API key。

## 5. 先运行只读计划

```powershell
$Tasks = "runs\private_real_thyroid_external\three_role_eval\paired_fact_n12\three_role_tasks.private.jsonl"
$Config = "competition_evidence\four_agent_endpoints.free.example.json"
$RunRoot = "runs\private_real_thyroid_external\three_role_eval\four_agent_expert_blind_v1"

& $EvalPython competition_evidence\execute_four_agent_eval.py `
  --input $Tasks `
  --config $Config `
  --output-dir $RunRoot
```

预期状态为 `PLAN_ONLY`，每套系统 `planned_task_count=72`。这一步不读取 API key、不访问端点，也不发送病例。计划清单写入：

`$RunRoot\four_agent_execution.private.json`

## 6. 执行四系统推理

确认第 3 节全部满足后，才运行：

```powershell
& $EvalPython competition_evidence\execute_four_agent_eval.py `
  --input $Tasks `
  --config $Config `
  --output-dir $RunRoot `
  --execute `
  --isolated-eval-ack I_CONFIRM_PRIVATE_ISOLATED_EVAL `
  --external-memory-ack I_CONFIRM_EXTERNAL_MEMORY_AND_TELEMETRY_DISABLED `
  --remote-data-transfer-ack I_CONFIRM_AUTHORIZED_REMOTE_PROCESSING
```

执行器会在开始第一条请求前一次性检查全部环境变量和确认项。单套系统中断后可修复原因并续跑：

```powershell
& $EvalPython competition_evidence\execute_four_agent_eval.py `
  --input $Tasks `
  --config $Config `
  --output-dir $RunRoot `
  --execute --resume --retry-failures `
  --isolated-eval-ack I_CONFIRM_PRIVATE_ISOLATED_EVAL `
  --external-memory-ack I_CONFIRM_EXTERNAL_MEMORY_AND_TELEMETRY_DISABLED `
  --remote-data-transfer-ack I_CONFIRM_AUTHORIZED_REMOTE_PROCESSING
```

只有顶层清单为 `COMPLETE`，且四套系统各 `success_count=72`、`failure_count=0`，才进入盲表生成。不得选择性重跑“不好看”的答案；技术失败只能按冻结重试规则处理。

## 7. 生成每位专家独立盲表

盲态密钥应由协调人随机生成并只保存在当前会话或密码管理器中。不要把它交给专家：

```powershell
$BlindBytes = New-Object byte[] 48
$BlindRng = [Security.Cryptography.RandomNumberGenerator]::Create()
$BlindRng.GetBytes($BlindBytes)
$env:EXPERT_BLIND_SECRET = [Convert]::ToBase64String($BlindBytes)
$BlindRng.Dispose()

$ReviewPrivate = "$RunRoot\expert_review_private"
$ReviewProtocol = "$RunRoot\expert_review_public_protocol"

& $EvalPython competition_evidence\prepare_agent_preference_review.py `
  --tasks $Tasks `
  --response "shennong_sanyan_full=$RunRoot\shennong_sanyan_full\responses.private.jsonl" `
  --response "cloudflare_api_agent=$RunRoot\cloudflare_api_agent\responses.private.jsonl" `
  --response "groq_api_agent=$RunRoot\groq_api_agent\responses.private.jsonl" `
  --response "cerebras_api_agent=$RunRoot\cerebras_api_agent\responses.private.jsonl" `
  --reviewer expert_01 `
  --reviewer expert_02 `
  --blind-secret-env EXPERT_BLIND_SECRET `
  --xlsx `
  --output-private $ReviewPrivate `
  --output-public $ReviewProtocol
```

建议至少两名与项目开发无关、专业方向匹配的专家。`expert_01` 等只是内部假名，真实身份对应关系另行保管。

脚本会先完成以下阻断式检查：

- 四套回答覆盖完全相同的 72 个任务，且与冻结任务逐条哈希绑定；
- 每条回答成功且非空；
- 回答中不存在项目名、厂商名、模型名或本机路径等破盲线索；
- 每位专家、每题使用独立 A/B/C/D 映射；
- 在角色 × 证据条件内，每套系统出现于 A/B/C/D 的次数差不超过 1。

只把对应的 `reviewer_1.private.xlsx` 或 CSV 发给第一位专家，把 `reviewer_2` 发给第二位专家。严禁发送 `blind_mapping.private.jsonl`、`review_manifest.private.json` 或其他系统标签文件。

## 8. 专家填写规则

每题必须使用以下合同：

- `best_choice`：`A/B/C/D`、`TIE`、`NONE` 或 `SKIP`；
- `tied_candidates`：仅在 `TIE` 时填写 2–4 个代码，如 `A|C`；
- `unsafe_candidates`：非 `SKIP` 题必填；没有不安全回答时填 `NONE`；
- `primary_reason`：`FACT/SAFETY/EVIDENCE/ROLE/USEFULNESS/CLARITY/OTHER`；
- `confidence_1_3`：1–3；
- `skip_reason_private`：仅 `SKIP` 时必填。

被标记为不安全的候选不能同时成为唯一或并列最佳。四份回答都不可接受时选择 `NONE`，不能强迫产生赢家。专家不要调整、删改或重新排版候选答案，只填写评分列。

## 9. 冻结评分并汇总

协调人收到完整评分表后先保存原件并计算 SHA-256，再用新目录运行汇总。以下以两份 XLSX 为例：

```powershell
Get-FileHash "$RunRoot\completed\expert_01.completed.private.xlsx" -Algorithm SHA256
Get-FileHash "$RunRoot\completed\expert_02.completed.private.xlsx" -Algorithm SHA256

& $EvalPython competition_evidence\summarize_agent_preference_review.py `
  --manifest "$ReviewPrivate\review_manifest.private.json" `
  --mapping "$ReviewPrivate\blind_mapping.private.jsonl" `
  --ratings "$RunRoot\completed\expert_01.completed.private.xlsx" `
  --ratings "$RunRoot\completed\expert_02.completed.private.xlsx" `
  --primary-agent shennong_sanyan_full `
  --bootstrap-replicates 10000 `
  --bootstrap-seed 20260824 `
  --output-private "$RunRoot\expert_results_private" `
  --output-public "$RunRoot\expert_results_public"
```

若专家填写 CSV，则把 `--ratings` 路径改为 CSV。汇总脚本会验证所有只读字段、答案文本和盲态映射均未被改动，再输出：

- 四系统安全合格首选份额与排序；
- 神农三言对三个固定对照的病例族聚类 Bootstrap 95% CI；
- 医生、患者、学生及两种证据条件的完整分层；
- `TIE/NONE/SKIP`、不安全标记和专家一致性；
- 仅含聚合结果的公开目录。

只有 `expert_preference_summary.json` 的状态为 `FINAL` 才能进入正式材料。若只有一位专家，只能写“单专家匿名首选”，不能写“专家共识”。

## 10. 正文报告边界

正文可以突出最有解释力的主指标和图，但辅助材料应完整披露四系统、三个固定对手、全部角色、全部证据条件、并列、全部不可接受、安全标记和技术失败。不得在查看答案后更换较弱对手，也不得只保留获胜角色。

推荐一句话：

> 在同一真实病例、同一角色任务的匿名专家比较中，神农三言完整项目取得了更高的安全合格专家首选份额；结果按病例族聚类并在医生、患者、学生及证据不足条件下完整分层。

其中“更高”只有在最终统计确实支持时才能使用；未完成专家评分前不得预写胜率或名次。

## 11. 官方依据

- Cloudflare OpenAI 兼容接口：<https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/>
- Cloudflare 免费额度：<https://developers.cloudflare.com/workers-ai/platform/pricing/>
- Cloudflare 数据使用：<https://developers.cloudflare.com/workers-ai/platform/data-usage/>
- Groq OpenAI 兼容接口：<https://console.groq.com/docs/openai>
- Groq 免费速率限制：<https://console.groq.com/docs/rate-limits>
- Groq 数据与 ZDR：<https://console.groq.com/docs/your-data>
- Cerebras OpenAI 兼容接口：<https://inference-docs.cerebras.ai/resources/openai>
- Cerebras Free Trial 限制：<https://inference-docs.cerebras.ai/support/rate-limits>
- Cerebras 数据条款：<https://www.cerebras.ai/terms-of-service>
- Gemini 免费层数据条款：<https://ai.google.dev/gemini-api/terms>
