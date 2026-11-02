# GuardTCM LightRAG 本地服务

该目录固定使用 `lightrag-hku[api]==1.5.4`，服务只监听 `127.0.0.1:9621`。浏览器不会直接访问此端口，所有查询均由 GuardTCM 后端代理并执行角色与来源过滤。

```bash
./scripts/setup_local_ai.sh
cp services/lightrag/.env.example services/lightrag/.env
./services/lightrag/start_lightrag.sh
services/lightrag/.venv/bin/python services/lightrag/health_check.py
```

`.env` 中的 `LIGHTRAG_API_KEY` 必须与后端配置一致。运行数据写入 `runtime/lightrag/`，不会提交到 Git。

流水线建图时服务使用 `runtime/lightrag/staging/current`；验收并激活后，服务使用 `runtime/lightrag/active/current`，来源文档位于 `runtime/lightrag/active/input`。切换快照后需要重启本服务。

经 `1.5.4` 实际 OpenAPI 验证，索引进度接口为 `/documents/track_status/{track_id}`；GuardTCM 客户端会在启动时校验该契约。
