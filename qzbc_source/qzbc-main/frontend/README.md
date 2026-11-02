# GuardTCM 前端

前端使用 React、TypeScript、Vite、Ant Design、TanStack Query 和 Cytoscape，实现医生、患者、学生三角色界面。

## 开发

```bash
npm install
npm run dev
```

默认地址为 `http://127.0.0.1:5173`。如后端端口改变，可在 `frontend/.env` 设置：

```text
VITE_API_BASE_URL=http://127.0.0.1:8001
```

生产构建：

```bash
npm run build
```

## 知识图谱

- 医生：`/doctor/knowledge-graph`
- 学生：`/student/knowledge-graph`

共享 `KnowledgeGraphView` 支持实体搜索、节点类型筛选、拖拽、缩放、适配视图、重置布局、邻居高亮、图例和节点详情。桌面最多显示 200 个节点，移动端最多 80 个。

浏览器只访问 GuardTCM 后端 `/api/lightrag/graph*`，不会直接访问 LightRAG 的 `9621` 端口。后端不可用时页面显示明确的降级提示，不渲染空白画布。
