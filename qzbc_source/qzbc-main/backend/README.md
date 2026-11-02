# GuardTCM 后端

后端基于 FastAPI、SQLAlchemy、LangGraph 和 MCP 权限网关，实现三角色业务、安全守卫、证据包、人工确认、LLM 生成、知识库导入、LightRAG 代理和评测 Harness。

## 启动

```bash
cd backend
../.conda/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

初始化数据库和演示数据：

```bash
../.conda/bin/python -c "from app.db.init_db import init_db; init_db()"
../.conda/bin/python scripts/seed_demo.py
```

## 默认安全配置

```text
LLM_PROVIDER=template
KNOWLEDGE_BACKEND=local_knowledge
LIGHTRAG_ENABLED=false
LLM_ALLOW_FALLBACK=true
LIGHTRAG_ALLOW_FALLBACK=true
```

本地真实模式：

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_NUM_CTX=32768
KNOWLEDGE_BACKEND=lightrag
LIGHTRAG_ENABLED=true
LIGHTRAG_MODE=api
LIGHTRAG_API_BASE_URL=http://127.0.0.1:9621
LIGHTRAG_API_KEY=replace-with-a-random-local-secret
```

## 数据流水线

```bash
../.conda/bin/python -m app.data_pipeline.scripts.download_sources
../.conda/bin/python -m app.data_pipeline.scripts.verify_sources
../.conda/bin/python -m app.data_pipeline.scripts.process_approved_sources
../.conda/bin/python -m app.data_pipeline.scripts.build_guardtcm_kb
```

数据门禁包括许可证验证、commit 固定、稳定 ID、MTCMB PII/危险输出隔离、内容 schema 校验和来源追溯。MTCMB 条目固定为学生/评测用途，医生临床证据和患者内容默认拒绝。

## LightRAG API

后端是浏览器访问 LightRAG 的唯一代理：

```text
GET  /api/lightrag/health
POST /api/lightrag/reindex
GET  /api/lightrag/jobs
GET  /api/lightrag/jobs/{run_id}
POST /api/lightrag/jobs/{run_id}/cancel
GET  /api/lightrag/graph/labels?q=&limit=50
GET  /api/lightrag/graph?label=&max_depth=2&max_nodes=200
```

图谱接口只允许医生和学生访问。后端会过滤未知来源、未知许可证、未知可见性和角色不可见节点。LightRAG 失败时返回 `local_knowledge` 图谱并标记 degraded/fallback。

## 快照状态

知识快照按以下状态流转：

```text
STAGING -> VERIFYING -> ACTIVE
                   \-> FAILED
```

`knowledge_snapshots` 保存来源清单、内容哈希、记录数、LightRAG workspace 和模型版本。失败构建不得替换 ACTIVE 快照。

## LLM 安全链路

- 患者：`SafetyPreNode -> Evidence Packet -> Ollama -> SafetyPostNode`；高风险请求在 Ollama 前阻断。
- 医生：规则和证据先行，LLM 只组织说明，最终必须经过 `HumanConfirm`。
- 学生：只使用脱敏教学数据，生成后再次执行 PII 检查。

模型输出必须通过 `GeneratedResponse` schema；失败时回退 TemplateProvider。自然语言图谱回答不能直接作为 Evidence，证据来自 `/query/data` 的实体、关系、chunk 和 reference。

## 测试

```bash
../.conda/bin/pytest
../.conda/bin/pytest app/tests/test_phase9_harness.py
../.conda/bin/pytest app/tests/test_phase10_harness.py
../.conda/bin/pytest app/tests/test_phase12_evaluation.py
```

测试数据库使用 SQLite。运行数据、原始仓库、模型和 LightRAG store 不提交到 Git。
