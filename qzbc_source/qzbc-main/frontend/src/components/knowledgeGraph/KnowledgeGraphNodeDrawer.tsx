import { Descriptions, Drawer, List, Space, Tag, Typography } from "antd";

import { NODE_TYPE_LABELS, RELATION_LABELS, type GraphSelection } from "./graphTypes";

const SOURCE_LINKS: Record<string, string> = {
  SRC_TCM_DATA_FOR_ML: "https://github.com/GuoxiangZu/TCMDataForML",
  SRC_MTCMB: "https://github.com/Wayyuanyuan/MTCMB",
  SRC_DRYAD_TCM_NETWORK: "https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwx9",
};

export function KnowledgeGraphNodeDrawer({ selection, onClose }: { selection: GraphSelection | null; onClose: () => void }) {
  const node = selection?.node;
  return (
    <Drawer rootClassName="knowledge-detail-drawer" title={node?.label || "节点详情"} open={Boolean(node)} onClose={onClose} width={420}>
      {node ? (
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Descriptions column={1} size="small">
            <Descriptions.Item label="节点类型">{NODE_TYPE_LABELS[node.type] || "其他"}</Descriptions.Item>
            <Descriptions.Item label="可见范围">{node.visibility.join("、") || "未标注"}</Descriptions.Item>
            <Descriptions.Item label="来源">
              <Space wrap>{node.source_ids.map((sourceId) => <a key={sourceId} href={SOURCE_LINKS[sourceId]} target="_blank" rel="noreferrer"><Tag color="blue">{sourceId}</Tag></a>)}</Space>
            </Descriptions.Item>
          </Descriptions>
          <Typography.Paragraph className="pre-wrap">{node.description || "暂无描述"}</Typography.Paragraph>
          <List
            header="相邻关系"
            dataSource={selection?.relations || []}
            locale={{ emptyText: "暂无可见关系" }}
            renderItem={(edge) => <List.Item><Typography.Text code>{RELATION_LABELS[edge.label] || "知识关联"}</Typography.Text>&nbsp;{edge.source} → {edge.target}{edge.evidence_count && edge.evidence_count > 1 ? <Tag color="blue" style={{ marginLeft: 8 }}>来源 {edge.evidence_count} 条</Tag> : null}</List.Item>}
          />
        </Space>
      ) : null}
    </Drawer>
  );
}
