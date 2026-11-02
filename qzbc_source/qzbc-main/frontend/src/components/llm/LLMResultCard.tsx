import { Card, Descriptions, List, Space, Tag, Typography } from "antd";

import type { GeneratedResponsePayload, LLMResultPayload } from "../../api/types";
import { asText } from "../../utils/format";

export function LLMResultCard({ result, generated }: { result?: LLMResultPayload | null; generated?: GeneratedResponsePayload | null }) {
  if (!result && !generated) return null;
  const fallback = Boolean(result?.metadata?.fallback_used);
  const degraded = Boolean(result?.metadata?.degraded);
  return (
    <Card title="LLM 生成层">
      <Space direction="vertical" size={12} style={{ width: "100%" }}>
        <Descriptions size="small" column={2}>
          <Descriptions.Item label="provider">{result?.provider || "-"}</Descriptions.Item>
          <Descriptions.Item label="model">{result?.model || "-"}</Descriptions.Item>
          <Descriptions.Item label="fallback"><Tag color={fallback ? "gold" : "green"}>{String(fallback)}</Tag></Descriptions.Item>
          <Descriptions.Item label="degraded"><Tag color={degraded ? "orange" : "green"}>{String(degraded)}</Tag></Descriptions.Item>
          <Descriptions.Item label="source_ids">{asText(result?.grounding?.source_ids || [])}</Descriptions.Item>
          <Descriptions.Item label="safety_flags">{asText(result?.safety_flags || [])}</Descriptions.Item>
        </Descriptions>
        {generated?.title ? <Typography.Text strong>{generated.title}</Typography.Text> : null}
        {generated?.summary ? <Typography.Paragraph className="pre-wrap">{generated.summary}</Typography.Paragraph> : null}
        {generated?.sections?.length ? (
          <List
            size="small"
            dataSource={generated.sections}
            renderItem={(section) => (
              <List.Item>
                <Space direction="vertical" size={2}>
                  <Typography.Text strong>{section.heading || "生成段落"}</Typography.Text>
                  <Typography.Text className="pre-wrap">{section.content}</Typography.Text>
                </Space>
              </List.Item>
            )}
          />
        ) : null}
        {generated?.warnings?.length ? (
          <List
            header="安全提醒"
            size="small"
            dataSource={generated.warnings}
            renderItem={(item) => <List.Item>{item}</List.Item>}
          />
        ) : null}
        {generated?.next_actions?.length ? (
          <List
            header="下一步"
            size="small"
            dataSource={generated.next_actions}
            renderItem={(item) => <List.Item>{item}</List.Item>}
          />
        ) : null}
        {generated?.disclaimer ? <Typography.Text type="secondary">{generated.disclaimer}</Typography.Text> : null}
      </Space>
    </Card>
  );
}
