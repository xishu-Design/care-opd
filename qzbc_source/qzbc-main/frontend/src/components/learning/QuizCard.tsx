import { Card, List, Space, Tag, Typography } from "antd";

import { asText } from "../../utils/format";

export function QuizCard({ items = [] }: { items?: Array<Record<string, unknown>> }) {
  return (
    <Card title="测验">
      <List
        dataSource={items}
        locale={{ emptyText: "暂无测验题" }}
        renderItem={(item, index) => (
          <List.Item>
            <Space direction="vertical" size={6} style={{ width: "100%" }}>
              <Typography.Text strong className="pre-wrap">{index + 1}. {asText(item.question || item)}</Typography.Text>
              {item.case_context && typeof item.case_context === "object" && (item.case_context as Record<string, unknown>).mode !== "knowledge_evaluation" ? (
                <Space wrap>
                  <Tag>{asText((item.case_context as Record<string, unknown>).sex_label)}</Tag>
                  <Tag>{asText((item.case_context as Record<string, unknown>).age_text)}</Tag>
                  <Tag color="blue">症状：{asText((item.case_context as Record<string, unknown>).symptoms)}</Tag>
                </Space>
              ) : null}
              {Array.isArray(item.options) ? <Typography.Text>选项：{item.options.map(String).join("；")}</Typography.Text> : null}
              {item.answer ? <Typography.Text type="secondary">参考答案：{asText(item.answer)}</Typography.Text> : null}
              {item.explanation ? <Typography.Text type="secondary">解析：{asText(item.explanation)}</Typography.Text> : null}
            </Space>
          </List.Item>
        )}
      />
    </Card>
  );
}
