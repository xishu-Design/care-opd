import { List, Space, Typography } from "antd";

import { StatusTag } from "../common/StatusTag";
import { asText } from "../../utils/format";

export function RuleFindingList({ items = [] }: { items?: Array<Record<string, unknown>> }) {
  return (
    <List
      size="small"
      dataSource={items}
      locale={{ emptyText: "无规则触发" }}
      renderItem={(item) => (
        <List.Item>
          <Space direction="vertical" size={4}>
            <Space wrap>
              <StatusTag value={String(item.risk_level || item.level || "")} kind="risk" />
              <Typography.Text strong>{String(item.risk_type || item.rule_type || item.rule_id || "规则")}</Typography.Text>
            </Space>
            <Typography.Text className="pre-wrap">{String(item.message || item.evidence || asText(item))}</Typography.Text>
          </Space>
        </List.Item>
      )}
    />
  );
}
