import { Descriptions, List, Space, Tag, Typography } from "antd";

import { asText } from "../../utils/format";

export function EvidenceSourceList({ items = [] }: { items?: Array<Record<string, unknown>> }) {
  return (
    <List
      size="small"
      dataSource={items}
      locale={{ emptyText: "无证据来源" }}
      renderItem={(item) => (
        <List.Item>
          <Space direction="vertical" size={8} style={{ width: "100%" }}>
            <Space wrap>
              <Typography.Text strong>{asText(pick(item, "source_id"))}</Typography.Text>
              <Tag color={pick(item, "retrieved_by") === "lightrag" ? "purple" : "default"}>{asText(pick(item, "retrieved_by") || "retrieval")}</Tag>
              <Tag color="blue">{asText(pick(item, "evidence_level"))}</Tag>
              <Tag>{asText(pick(item, "license"))}</Tag>
            </Space>
            <Descriptions size="small" column={1} bordered>
              <Descriptions.Item label="entry_id">{asText(pick(item, "entry_id"))}</Descriptions.Item>
              <Descriptions.Item label="entry_type">{asText(pick(item, "entry_type"))}</Descriptions.Item>
              <Descriptions.Item label="retrieval_mode">{asText(pick(item, "retrieval_mode"))}</Descriptions.Item>
              <Descriptions.Item label="source_title">{asText(pick(item, "source_title"))}</Descriptions.Item>
              <Descriptions.Item label="original_source">{asText(pick(item, "original_source"))}</Descriptions.Item>
              <Descriptions.Item label="source_record_id">{asText(sourceRecordId(item))}</Descriptions.Item>
              <Descriptions.Item label="processing_script">{asText(pick(item, "processing_script"))}</Descriptions.Item>
              <Descriptions.Item label="processing_version">{asText(pick(item, "processing_version"))}</Descriptions.Item>
              <Descriptions.Item label="graph_entities">{asText(pick(item, "graph_entities"))}</Descriptions.Item>
              <Descriptions.Item label="graph_relations">{asText(pick(item, "graph_relations"))}</Descriptions.Item>
              <Descriptions.Item label="confidence">{asText(pick(item, "confidence"))}</Descriptions.Item>
            </Descriptions>
            <Typography.Text className="pre-wrap" type="secondary">{asText(item.snippet || item.summary || item.title)}</Typography.Text>
          </Space>
        </List.Item>
      )}
    />
  );
}

function pick(item: Record<string, unknown>, key: string) {
  const metadata = isRecord(item.metadata) ? item.metadata : {};
  return item[key] ?? metadata[key];
}

function sourceRecordId(item: Record<string, unknown>) {
  const metadata = isRecord(item.metadata) ? item.metadata : {};
  const records = Array.isArray(item.source_records) ? item.source_records : Array.isArray(metadata.source_records) ? metadata.source_records : [];
  const first = records.find(isRecord);
  return first?.source_record_id || first?.source_file || "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
