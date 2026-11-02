import { useQuery } from "@tanstack/react-query";
import { Card, Descriptions, Space, Statistic, Tabs, Table, Typography } from "antd";

import { harnessApi } from "../../api/harnessApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { asText } from "../../utils/format";

function countRows(counts: Record<string, number>) {
  return Object.entries(counts || {}).map(([file, count]) => ({ file, count }));
}

export function DataPipelinePage() {
  const query = useQuery({ queryKey: ["harness", "phase9", "data-pipeline"], queryFn: harnessApi.getDataPipeline });
  const data = query.data;

  return (
    <>
      <PageHeader title="数据流水线" subtitle="第 9 阶段下载、许可证、规范化处理和内容构建报告。" />
      {query.isLoading ? <LoadingState /> : query.isError ? <ErrorState error={query.error} /> : data ? (
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <div className="metric-grid">
            <Card><Statistic title="批准来源" value={Number(data.summary.approved_source_count || 0)} /></Card>
            <Card><Statistic title="已处理条目" value={Number(data.summary.processed_item_count || 0)} /></Card>
            <Card><Statistic title="待处理条目" value={Number(data.summary.pending_item_count || 0)} /></Card>
            <Card><Statistic title="内容条目" value={Number(data.summary.content_item_count || 0)} /></Card>
          </div>
          <div className="two-column">
            <Card title="规范化文件">
              <Table
                size="small"
                rowKey="file"
                pagination={false}
                dataSource={countRows(data.processed_counts)}
                columns={[
                  { title: "file", dataIndex: "file" },
                  { title: "count", dataIndex: "count", width: 100 },
                ]}
              />
            </Card>
            <Card title="内容文件">
              <Table
                size="small"
                rowKey="file"
                pagination={false}
                dataSource={countRows(data.content_counts)}
                columns={[
                  { title: "file", dataIndex: "file" },
                  { title: "count", dataIndex: "count", width: 100 },
                ]}
              />
            </Card>
          </div>
          <Card title="流水线报告">
            <Tabs
              items={[
                {
                  key: "download",
                  label: "download_report",
                  children: <pre className="json-box">{JSON.stringify(data.reports.download_report, null, 2)}</pre>,
                },
                {
                  key: "license",
                  label: "license_report",
                  children: <pre className="json-box">{JSON.stringify(data.reports.license_report, null, 2)}</pre>,
                },
                {
                  key: "processing",
                  label: "processing_report",
                  children: <pre className="json-box">{JSON.stringify(data.reports.processing_report, null, 2)}</pre>,
                },
                {
                  key: "quality",
                  label: "quality_report",
                  children: <pre className="json-box">{JSON.stringify(data.reports.quality_report, null, 2)}</pre>,
                },
              ]}
            />
          </Card>
          <Card title="当前数据库快照">
            <Descriptions column={2} size="small" bordered>
              {Object.entries(data.content_stats || {}).map(([key, value]) => (
                <Descriptions.Item key={key} label={key}>
                  <Typography.Text className="pre-wrap">{asText(value)}</Typography.Text>
                </Descriptions.Item>
              ))}
            </Descriptions>
          </Card>
        </Space>
      ) : null}
    </>
  );
}
