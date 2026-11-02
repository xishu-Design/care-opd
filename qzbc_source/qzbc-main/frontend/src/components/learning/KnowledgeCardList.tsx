import { List, Typography } from "antd";

import { asText } from "../../utils/format";

export function KnowledgeCardList({ items = [] }: { items?: Array<Record<string, unknown>> }) {
  return (
    <List
      dataSource={items}
      locale={{ emptyText: "暂无知识卡" }}
      renderItem={(item) => (
        <List.Item>
          <Typography.Text className="pre-wrap">{asText(item)}</Typography.Text>
        </List.Item>
      )}
    />
  );
}
