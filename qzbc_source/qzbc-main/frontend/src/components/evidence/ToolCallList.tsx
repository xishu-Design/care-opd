import { Table, Tag } from "antd";

export function ToolCallList({ items = [] }: { items?: Array<Record<string, unknown>> }) {
  return (
    <Table
      size="small"
      rowKey={(row, index) => String(row.request_id || row.tool_name || index)}
      dataSource={items}
      pagination={false}
      columns={[
        { title: "Tool", dataIndex: "tool_name" },
        {
          title: "Allowed",
          dataIndex: "allowed",
          render: (value) => <Tag color={value ? "green" : "red"}>{String(value)}</Tag>,
        },
        {
          title: "Success",
          dataIndex: "success",
          render: (value) => <Tag color={value ? "green" : "red"}>{String(value)}</Tag>,
        },
        { title: "检索后端", dataIndex: "backend" },
        {
          title: "是否降级",
          dataIndex: "fallback_used",
          render: (value) => <Tag color={value ? "gold" : "default"}>{String(Boolean(value))}</Tag>,
        },
        {
          title: "Degraded",
          dataIndex: "degraded",
          render: (value) => <Tag color={value ? "gold" : "default"}>{String(Boolean(value))}</Tag>,
        },
        { title: "Error", dataIndex: "error_code" },
      ]}
    />
  );
}
