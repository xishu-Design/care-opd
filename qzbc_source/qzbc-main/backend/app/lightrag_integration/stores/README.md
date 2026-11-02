# LightRAG 兼容存储

该目录用于离线测试中的轻量兼容索引。真实 GraphRAG 服务的数据写入 `runtime/lightrag/{staging,active,failed}`，并由独立 LightRAG `1.5.4` API Server 管理。

兼容存储不能冒充真实知识图谱；前端和证据包会明确显示当前 backend 与 degraded 状态。
