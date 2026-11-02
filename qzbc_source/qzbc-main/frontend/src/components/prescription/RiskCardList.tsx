import { Alert, List, Space, Typography } from "antd";

import { StatusTag } from "../common/StatusTag";
import { asText } from "../../utils/format";

export function RiskCardList({ items = [] }: { items?: Array<Record<string, unknown>> }) {
  if (!items.length) return <Alert type="success" showIcon message="未返回风险卡片" />;
  return (
    <List
      dataSource={items}
      renderItem={(item) => (
        <List.Item>
          <Space direction="vertical" size={4}>
            <Space wrap>
              <StatusTag value={String(item.risk_level || "yellow")} kind="risk" />
              <Typography.Text strong>{String(item.risk_type || item.rule_type || "审方风险")}</Typography.Text>
            </Space>
            <Typography.Text className="pre-wrap">{String(item.message || item.evidence || asText(item))}</Typography.Text>
          </Space>
        </List.Item>
      )}
    />
  );
}
