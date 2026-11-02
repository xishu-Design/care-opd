# GuardTCM

完整技术栈、架构、数据流和运行边界可在本地技术文档中维护。

GuardTCM 是面向医生、患者和中医学生的三角色安全辅助系统。系统保留规则引擎、MCP 权限网关、证据包、LangGraph、人工确认和降级边界，并增加本地 Ollama、真实 LightRAG 服务与医生/学生知识图谱。

> 重要边界：系统不替代医生诊疗。患者高风险请求会在进入 LLM 前阻断；知识图谱关系不能直接生成处方、修改处方状态或升级为临床规则。

## 三角色 AI 问答

- 医生端 `/doctor/ai-chat`：面向专业知识检索和临床决策辅助，组织证候、治法、方剂、药材、风险与证据之间的关系，最终结论仍需医生人工确认。
- 患者端 `/patient/education`：只提供通俗安全科普，诊断、开方、改药、剂量和停药请求由安全检查拦截。
- 学生端 `/student/ai-chat`：按照概念解释、辨析思路、知识联系和复习建议进行教学，不作为临床诊疗依据。

三个角色的问答记录均持久化到数据库，并记录创建时间、模型信息、证据包和 Trace。真实模型开发模式同时启用：

- `Mem0 2.0.12 + 本地 Qdrant`：只镜像记忆巩固智能体审核通过的长期偏好、学习目标和专业关注点，命名空间为 `角色 + 用户编号`；完整问答不再写入 Mem0。
- `Graphiti 0.29.2 + 嵌入式 FalkorDB`：从问答或审方事件中抽取时序实体关系，使用本地 `qwen2.5:7b` 和 `bge-m3`，不连接云端服务。

SQL 问答表仍是权威持久化来源；Graphiti/Mem0 不可用时分别安全回退到 `persistent_graph` 和 `persistent_memory`，不会造成问答记录丢失。

### 分层上下文管理

问答上下文采用开源项目中常见的“短期窗口 + 长期语义记忆 + 时序关系”分层设计：

- 短期上下文：从 SQL 权威问答表按 `用户 + 角色` 检查最近 6 轮，但不会把 6 轮全部交给模型。系统先判断 `new_topic / topic_continuation / answer_reference`，新话题不注入旧消息；追问最多使用最近 1–2 轮连续主题。
- 追问消解：识别“它、这个、那、还有呢、为什么”和对上一条回答的引用。检索查询只补充上一问题，不再拼接可能有误的上一回答；只有明确引用上一回答时才提供对应回答文本。
- 长期语义记忆：`MemoryConsolidationAgent` 只从用户消息提取稳定偏好、学生学习目标/薄弱点和医生专业关注点，按 key 去重或更新，再写入 SQL 并可选镜像到 Mem0 + Qdrant。普通问题、临时事项和模型回答不进入长期层。
- 时序关系：Graphiti 保存会随时间变化的实体关系和事件来源；它不替代会话消息表。
- 线程状态：LangGraph 的 `thread_id` 对同一用户、同一角色保持稳定，请求级 `session_id` 仍单独生成用于审计。

历史对话只用于指代消解与表达连续性，不会被当成医学证据；医生、学生、患者以及不同用户之间均隔离。Evidence Packet 会同时记录“可用历史轮数”和“本轮实际使用轮数”。前端显示“已使用 N 轮相关上下文”；检测到新话题时显示“新话题 · 未引用旧回答”。可配置项：

问答页顶部的“AI 长期记忆”面板支持查看、明确记住和逐条忘记。过敏史、病史、诊断、处方、剂量、检查结果和身份信息会被记忆智能体阻断，必须写入患者资料或对应结构化医疗记录；长期记忆只能调整回答方式，不能作为医学证据。

```text
CONVERSATION_CONTEXT_ENABLED=true
CONVERSATION_RECENT_TURNS=6
CONVERSATION_CONTEXT_MAX_CHARS=6000
MEM0_SEARCH_TOP_K=8
MEM0_MIN_RELEVANCE_SCORE=0.55
```

设计参考采用并组合了以下开源项目的职责划分，而不是再引入一个重复的外部服务：

- [LangGraph](https://github.com/langchain-ai/langgraph)：线程级短期状态、持久执行与长期记忆分层。
- [Mem0](https://github.com/mem0ai/mem0)：用户级长期语义记忆和检索。
- [LangMem](https://github.com/langchain-ai/langmem)：后台提取、巩固、更新记忆与前台显式管理分层。
- [Graphiti](https://github.com/getzep/graphiti)：带时间有效期和来源的关系记忆；其开源版明确要求应用自行实现用户、线程和消息存储。
- [Letta](https://github.com/letta-ai/letta)：始终保留的核心上下文与可换入/换出的外部记忆思路。

医生、患者和学生问答均使用 SSE 连接 Ollama 的真实流式输出，前端按字符显示。生成期间“发送”按钮会切换为“停止生成”；停止后会取消当前请求、关闭模型流并保留已经显示的部分，历史记录中标记为生成已中止。

三个角色的 AI 回答均支持选区追问：在助手回答中选中一段文字，选区上方会出现“追问”按钮；点击后输入区显示引用卡片并自动聚焦。引用内容会随新问题一起发送，从而参与追问消解；最多引用 360 个字符，不能在用户自己的消息上触发，可在发送前取消引用。

## 本地问答知识库

MTCMB 与 ShenNong 问答采用分层 RAG，而不是直接微调模型：

- LightRAG：检索结构化药材、方剂和知识图谱关系。
- 本地 QA 索引：使用 SQLite FTS 从大规模问答中召回候选，先去除“时候、详细、问题”等泛词造成的伪命中，再使用本地 `bge-m3` 做语义重排；医疗流程类问题还会校验诊断、问诊、辨证等领域锚点。
- Ollama：只根据 Evidence Packet 中允许当前角色查看的证据生成回答。

当前 QA 索引包含 118360 条记录，其中 ShenNong 112565 条、MTCMB 5795 条。权限边界如下：

- 学生：可检索 MTCMB 和 ShenNong，但二者只作为教学参考；MTCMB 纯题库命中会直接复述题库答案，禁止模型补写临床结论。
- 医生：默认不检索上述未复核问答，只有明确标记为医生复核通过的问答才能进入医生 Evidence Packet。
- 患者：完全不读取原始 MTCMB/ShenNong 问答，只允许患者安全科普来源。

手动重建问答索引：

```bash
cd backend
../.conda/bin/python scripts/build_qa_knowledge_index.py
```

`./scripts/dev_backend.sh` 会在数据库比索引更新时自动重建，并默认启用真实 LightRAG：

```text
KNOWLEDGE_BACKEND=lightrag
LIGHTRAG_ENABLED=true
QA_KNOWLEDGE_ENABLED=true
```

处方中的“单味药剂量”和“处方用法”是两个独立字段：药材表格记录每味药的剂量与单位；“处方用法”记录煎服方式、每日频次和分服方法。审方提示会分别说明缺少的是哪一类信息，不再把已填写的单味药剂量显示成“剂量缺失”。

## 运行模式

稳定演示和 CI 默认使用：

```text
LLM_PROVIDER=template
KNOWLEDGE_BACKEND=local_knowledge
LIGHTRAG_ENABLED=false
```

本地真实模型模式使用：

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=bge-m3
KNOWLEDGE_BACKEND=lightrag
LIGHTRAG_ENABLED=true
LIGHTRAG_MODE=api
LIGHTRAG_API_BASE_URL=http://127.0.0.1:9621
GRAPH_BACKEND=graphiti
MEMORY_BACKEND=mem0
GRAPHITI_DB_PATH=runtime/graphiti/guardtcm.db
MEM0_QDRANT_PATH=runtime/mem0/qdrant
```

Ollama 或 LightRAG 不可用时，系统分别回退到 `template` 或 `local_knowledge`，并在 API、证据包、Trace 和前端显示 degraded/fallback 状态。

## 数据来源

自动处理的来源只有许可证明确的仓库：

- `TCMDataForML`：MIT。仅提取药材名、方剂组成、来源和教学候选，不作为临床权威。
- `MTCMB`：Apache-2.0。仅用于学生训练与系统评测，不进入患者内容或医生临床证据包。
- `TCM-MKG`：CC BY-NC 4.0。用于规范方剂的组成、功效、证候和疾病/症状关系；非商业候选检索仍需医生复核。
- `Dryad TCM heterogeneous network / CHPA`：CC0。官方补充文件已通过 Europe PMC 下载并校验哈希；草药 ID 使用 SymMap v2 中文药名表对齐。公开 CSV 的实际行数与论文摘要数字不一致，因此系统同时记录“论文声明数量”和“实际文件数量”，只按真实文件导入。草药—症状关系仅保留数值 `P<0.05` 的去重记录；英文术语由本地 Ollama 转为中文，并统一标记为“待医学复核的统计关联”，不进入患者内容或独立临床证据。
- `DB4403/T 142—2021 中药饮片处方审核规范`：深圳市市场监督管理局政府官网公开的地方标准。系统只提取附录 C 表 C.1 的十八反、十九畏事实药对，用于医生端红色审方预警和知识图谱禁配边。该公开 PDF 未发现开放数据许可证，因此登记为 `reference_only`，保留下载 URL、SHA-256 和表格定位，不把标准正文宣称为开源数据集或再分发。
- 国家药监局转载科普与北京市中医药管理局服药科普：仅整理页面明确列出的少量传统服药食忌参考，用于患者处方页的来源提示。系统将其标记为传统参考，不宣称为已经由现代临床试验证实的药食相互作用；未收录的水果、蔬菜、肉类关系不得由 Ollama 自由补全。

Dryad 的 `herbherb.csv` 表示基于共享属性/功效向量的药材关联，不是药物冲突，系统明确不把它转换为禁配关系。当前传统配伍禁忌规则共 12 组，展开为 75 条图谱药对；命中时暂停绿色结论并要求中药师或医生复核，不由 Ollama 自由判断，也不自动宣称绝对临床禁用。

`Knowlegde_Graph_TCM`、`TCM_NER_datasets`、`TCM-prescription-datasets`、`ZhongJing-OMNI` 等没有明确许可证的来源仅登记为 candidate，不自动下载或导入。

每条正式记录可追溯到 `source_id + commit_sha + source_file + source_record_id + license`。原始 clone、模型和 LightRAG 运行存储均被 Git 忽略。

CHPA 数据重新中文化和导入：

```bash
cd backend
../.conda/bin/python -m app.data_pipeline.scripts.import_chpa_relations --translate
```

导入报告位于 `backend/app/data_pipeline/reports/chpa_import_report.json`。未能从发布文件高置信度还原中文药名的旧式 `Hxxxx` 功效记录会进入报告，不会被强行绑定到某味中药。

重新校验并生成传统配伍禁忌规则：

```bash
cd backend
../.conda/bin/python -m app.data_pipeline.scripts.import_tcm_incompatibilities --download-reference
```

脚本会校验官方 PDF 的固定 SHA-256 和附录 C 关键原文标记，输出 `app/content/data/herb_incompatibilities.json` 与 `app/data_pipeline/reports/tcm_incompatibility_import_report.json`。若来源文件发生变化或校验失败，脚本会停止，不会静默更新审方规则。

## 环境准备

### 团队 Docker 开发启动

适合把项目分享给队友后快速启动前端和后端。Ollama 与 LightRAG 仍按本机服务运行，模型文件不进入 Docker 镜像，也不提交到 Git 仓库。

先在宿主机启动 Ollama 和 LightRAG：

```bash
./scripts/start_ollama.sh
./scripts/start_lightrag.sh
```

首次使用时拉取本地模型：

```bash
./runtime/ollama/bin/ollama pull qwen2.5:7b
./runtime/ollama/bin/ollama pull bge-m3
```

然后启动开发容器：

```bash
docker compose -f docker-compose.dev.yml up --build
```

如果本机是旧版 Docker Compose，使用：

```bash
docker-compose -f docker-compose.dev.yml up --build
```

默认访问地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000/health`

容器中的后端会自动安装 Python 依赖、执行数据库迁移、写入演示账号，并在需要时重建本地问答索引。若宿主机没有启动 Ollama 或 LightRAG，系统会按开发配置降级运行，并在界面上显示 fallback/degraded 状态。

### 使用统一 requirements.txt 安装

项目根目录提供 [requirements.txt](./requirements.txt)，包含后端、智能体、LightRAG、Graphiti、Mem0 和数据流水线所需的 Python 直接依赖：

```bash
cd /opt/care-user/qzbc
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm --prefix frontend install
```

`requirements.txt` 只管理 Python 包；React、TypeScript、Vite 等前端依赖由 `frontend/package.json` 和 `frontend/package-lock.json` 管理。现有 Conda 安装方式仍然保留。

项目使用仓库内的 Conda 环境：

```bash
CONDA_PKGS_DIRS="$PWD/.conda_pkgs" conda env create -p "$PWD/.conda" -f environment.yml
conda activate "$PWD/.conda"
```

前端依赖：

```bash
cd frontend
npm install
npm run build
```

LightRAG 独立服务环境：

```bash
./scripts/setup_local_ai.sh
cp services/lightrag/.env.example services/lightrag/.env
```

LightRAG 固定为 `lightrag-hku[api]==1.5.4`，只监听 `127.0.0.1:9621`，并通过 API Key 保护。浏览器不会直接访问该端口。

## 本地模型

```bash
./scripts/start_ollama.sh
./runtime/ollama/bin/ollama pull qwen2.5:7b
./runtime/ollama/bin/ollama pull bge-m3
./scripts/start_ollama_check.sh
```

`qwen2.5:7b` 用于三角色中文生成和 LightRAG 实体关系抽取，`bge-m3` 用于 1024 维中文/多语 embedding。当前机器可使用 CPU 运行，首次建图会较慢。

当前药材目录包含 1,260 味中文规范药材，其中 677 味有来源直接功效，448 味可经规范方剂建立“证候/症状 → 方剂 → 组成药材”的可追溯间接关系，两类合并后有 755 味至少具备一种可检索关系。间接关系在界面和推荐原因中均单独标注，权重低于直接功效/证候，不表示单味药可独立治疗对应证候。

## 一键知识库流水线

先运行小规模 Pilot：

```bash
./.conda/bin/python scripts/build_local_ai_stack.py \
  --sources TCMDataForML,MTCMB \
  --snapshot-version 2026.07.1 \
  --pilot
```

Pilot 通过后才允许全量构建和激活：

```bash
./.conda/bin/python scripts/build_local_ai_stack.py \
  --sources TCMDataForML,MTCMB \
  --snapshot-version 2026.07.1 \
  --activate-after-verify
```

可用参数包括 `--dry-run`、`--skip-download`、`--source`、`--max-records`、`--pilot` 和 `--resume-run-id`。流水线在 `runtime/lightrag/staging/` 构建，任一 Gate 失败都会转入 `failed/`，不会覆盖 ACTIVE 快照。

激活时会把规范化内容、LightRAG 图存储和来源文档作为同一个快照切换到 `runtime/lightrag/active/`，随后重启 LightRAG 与后端即可加载新快照。`lightrag_doc_manifest` 保存每份索引文档的稳定 ID、内容哈希、来源、许可证和可见性。

完整图谱包含数千份教学文档，建议先通过 Pilot 验收，再由医生端“重新建图”创建后台任务，并通过 `GET /api/lightrag/jobs/{run_id}` 查看真实抽取进度。任务按 20 份文档分批处理，重复文档会安全复用。

## 启动应用步骤

以下命令均从项目根目录执行：

```bash
cd /opt/care-user/qzbc
```

### 1. 首次下载本地模型

已下载模型时可跳过本步骤。项目优先使用仓库内完整的 Ollama 运行包，避免系统全局 Ollama 缺少运行器时自动回退到 template。

```bash
./runtime/ollama/bin/ollama pull qwen2.5:7b
./runtime/ollama/bin/ollama pull bge-m3
```

### 2. 启动 Ollama

在终端 1 运行，并保持该终端开启：

```bash
./scripts/start_ollama.sh
```

另开一个终端确认生成模型和向量模型均已就绪：

```bash
./scripts/start_ollama_check.sh
```

### 3. 启动 LightRAG

在终端 2 运行，并保持该终端开启：

```bash
./scripts/start_lightrag.sh
```

LightRAG 首次启动时可以为空索引；应用会显示本地可追溯图谱作为降级关系。完整知识文档建图应在医生端创建索引任务后异步执行。

### 4. 启动后端

在终端 3 运行，并保持该终端开启：

```bash
./scripts/dev_backend.sh
```

该脚本默认启用 `LLM_PROVIDER=ollama`、`GRAPH_BACKEND=graphiti`、`MEMORY_BACKEND=mem0` 和 `qwen2.5:7b`。Ollama 或外部记忆/图谱适配器不可用时，后端会安全回退，并在响应中标记 `fallback_used=true`。

### 5. 启动前端

在终端 4 运行：

```bash
./scripts/dev_frontend.sh
```

前端会连接 `http://127.0.0.1:8000`。Vite 会输出实际端口；若 `5173` 已被占用，会自动使用下一个可用端口。

### 6. 打开与验证

默认地址：

- 前端：`http://127.0.0.1:5173`（以 Vite 实际输出为准）
- 后端健康检查：`http://127.0.0.1:8000/health`
- LightRAG：`http://127.0.0.1:9621`
- Ollama：`http://127.0.0.1:11434`

Graphiti 使用进程内嵌入式 FalkorDB，Mem0 使用本地嵌入式 Qdrant，因此不额外暴露网络端口。持久化文件位于 `runtime/graphiti/` 和 `runtime/mem0/`。

可在另一个终端运行：

```bash
curl http://127.0.0.1:8000/health
./.conda/bin/python scripts/check_local_ai_stack.py
```

代码更新后，请在后端终端按 `Ctrl+C` 再次运行 `./scripts/dev_backend.sh`；旧的 `8000` 进程不会包含新接口，例如 `/api/doctor/prescription-suggestions`。

`./scripts/dev_backend.sh` 会自动执行 `alembic upgrade head`。手动启动后端时，需要先执行：

```bash
cd backend
../.conda/bin/alembic upgrade head
```

演示账号：

- 医生：`doctor_demo / password123`
- 患者：`patient_demo / password123`
- 患者2：`patient_demo2 / password1231`

- 学生：`student_demo / password123`

登录页提供医生、患者、学生三种角色的自主注册入口。医生端也可以只用已注册患者的用户名建立绑定，不需要知道患者密码；若医生为未注册患者代建账号，填写的是一次性临时密码，患者首次登录后必须在“修改密码”页设置自己的新密码。所有角色登录后均可从顶部导航再次修改密码。

医生图谱：`/doctor/knowledge-graph`。学生图谱：`/student/knowledge-graph`。患者访问 `/api/lightrag/graph*` 会返回 403。

## 健康检查

```bash
./scripts/start_guardtcm_local_ai.sh
./.conda/bin/python scripts/check_local_ai_stack.py
./scripts/demo_check.sh
```

统一检查 Ollama、两个模型、LightRAG、后端和前端。LightRAG 启动前还会校验 `/openapi.json` 是否包含 `1.5.4` 所需接口。

## 测试

```bash
cd backend
../.conda/bin/pytest

cd ../frontend
npm run build
```

医生可在 `/harness/evaluation`、`/harness/ablation`、`/harness/lightrag` 和 `/harness/traces` 查看评测、消融、索引和链路追踪结果。

## 确定性图谱边界

结构化流水线只生成以下明确关系：

- `formula CONTAINS herb`
- `entry DERIVED_FROM source`
- `case REFERENCES herb/formula`
- `herb HAS_CHPA_EFFICACY efficacy`（CHPA 统计关联，待医学复核）
- `herb ASSOCIATED_WITH_CHPA_SYMPTOM symptom`（CHPA `P<0.05` 统计关联，待医学复核）

不会自动生成 `TREATS`、`CURES`、`HAS_DOSE`、`CONTRAINDICATED_FOR` 或 `SUITABLE_FOR` 等临床推断关系。
