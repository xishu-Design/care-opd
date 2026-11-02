import { Space, Tag, Typography } from "antd";

import { NODE_COLORS } from "./graphStyles";
import { NODE_TYPES, NODE_TYPE_LABELS } from "./graphTypes";

export function KnowledgeGraphLegend() {
  return (
    <Space wrap size={[6, 6]}>
      {NODE_TYPES.map((type) => (
        <Tag key={type} color={NODE_COLORS[type]}>{NODE_TYPE_LABELS[type]}</Tag>
      ))}
      <Typography.Text type="secondary">节点颜色表示类型；金色边框表示当前中心节点。</Typography.Text>
      <Tag color="#9a6a55">棕色虚线：十八反/十九畏</Tag>
      <Tag>灰色实线：普通知识关系（非配伍禁忌）</Tag>
    </Space>
  );
}
