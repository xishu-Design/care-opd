import { useQuery } from "@tanstack/react-query";
import { Card, Descriptions, Space, Statistic, Table, Tag, Typography, type TableProps } from "antd";
import { ExternalLink } from "lucide-react";

import { harnessApi } from "../../api/harnessApi";
import type { Phase9SourceItem } from "../../api/types";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { StatusTag } from "../../components/common/StatusTag";
import { asText } from "../../utils/format";

const sourceColumns: TableProps<Phase9SourceItem>["columns"] = [
  {
    title: "source_id",
    dataIndex: "source_id",
    width: 220,
    fixed: "left",
    render: (value: string) => <Typography.Text strong>{value}</Typography.Text>,
  },
  {
    title: "name",
    dataIndex: "name",
    width: 260,
    render: (value: string, row) => (
      <Space direction="vertical" size={2}>
        <Typography.Text>{value}</Typography.Text>
        {row.homepage ? (
          <Typography.Link href={row.homepage} target="_blank" rel="noreferrer">
            <Space size={4}>来源主页 <ExternalLink size={12} /></Space>
          </Typography.Link>
        ) : null}
      </Space>
    ),
  },
  { title: "license", dataIndex: "license", width: 120 },
  {
    title: "status",
    dataIndex: "approval_status",
    width: 120,
    render: (value: string) => <StatusTag value={value} />,
  },
  {
    title: "downloaded",
    dataIndex: "downloaded",
    width: 120,
    render: (value: boolean) => <Tag color={value ? "green" : "default"}>{value ? "yes" : "no"}</Tag>,
  },
  {
    title: "can_process",
    dataIndex: "can_process",
    width: 130,
    render: (value: boolean) => <Tag color={value ? "green" : "gold"}>{value ? "yes" : "no"}</Tag>,
  },
  { title: "imported", dataIndex: "imported_count", width: 100 },
  { title: "allowed_use", dataIndex: "allowed_use", width: 150 },
  { title: "license_status", dataIndex: "license_status", width: 150 },
  {
    title: "notes",
    dataIndex: "notes",
    width: 360,
    render: (value: string) => <Typography.Text className="pre-wrap">{value || "-"}</Typography.Text>,
  },
];

export function SourceRegistryPage() {
  const query = useQuery({ queryKey: ["harness", "phase9", "sources"], queryFn: harnessApi.getSources });

  return (
    <>
      <PageHeader title="数据源登记" subtitle="第 9 阶段数据源、许可证、下载和导入状态。" />
      {query.isLoading ? <LoadingState /> : query.isError ? <ErrorState error={query.error} /> : query.data ? (
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <div className="metric-grid">
            <Card><Statistic title="已批准" value={Number(query.data.summary.approved_count || 0)} /></Card>
            <Card><Statistic title="待审核" value={Number(query.data.summary.pending_count || 0)} /></Card>
            <Card><Statistic title="已拒绝" value={Number(query.data.summary.rejected_count || 0)} /></Card>
            <Card><Statistic title="下载队列" value={Array.isArray(query.data.summary.download_queue) ? query.data.summary.download_queue.length : 0} /></Card>
          </div>
          <Card title="数据源清单">
            <Table
              size="small"
              rowKey="source_id"
              columns={sourceColumns}
              dataSource={query.data.sources}
              pagination={{ pageSize: 8 }}
              scroll={{ x: 1700 }}
            />
          </Card>
          <Card title="运行报告">
            <Descriptions column={1} size="small" bordered>
              <Descriptions.Item label="download_report">{asText(query.data.reports.download_report)}</Descriptions.Item>
              <Descriptions.Item label="license_report">{asText(query.data.reports.license_report)}</Descriptions.Item>
            </Descriptions>
          </Card>
        </Space>
      ) : null}
    </>
  );
}
