import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Card, Descriptions, Space, Statistic, Tabs, Tag } from "antd";
import { DatabaseZap, RefreshCw } from "lucide-react";

import { lightragApi } from "../../api/lightragApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { asText } from "../../utils/format";

export function LightRAGStatusPage() {
  const queryClient = useQueryClient();
  const health = useQuery({ queryKey: ["lightrag", "health"], queryFn: lightragApi.getHealth });
  const report = useQuery({ queryKey: ["lightrag", "report"], queryFn: lightragApi.getReport });
  const buildDocs = useMutation({
    mutationFn: lightragApi.buildDocs,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["lightrag"] }),
  });
  const reindex = useMutation({
    mutationFn: lightragApi.reindex,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["lightrag"] }),
  });
  const data = health.data;

  return (
    <>
      <PageHeader
        title="LightRAG"
        subtitle="第 10 阶段知识图谱索引、GraphRAG 检索和降级状态。"
        extra={
          <Space>
            <Button icon={<DatabaseZap size={16} />} loading={buildDocs.isPending} onClick={() => buildDocs.mutate()}>生成索引文档</Button>
            <Button type="primary" icon={<RefreshCw size={16} />} loading={reindex.isPending} onClick={() => reindex.mutate()}>重新建图</Button>
          </Space>
        }
      />
      {health.isLoading ? <LoadingState /> : health.isError ? <ErrorState error={health.error} /> : data ? (
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <div className="metric-grid">
            <Card><Statistic title="检索后端" value={asText(data.backend || "lightrag")} /></Card>
            <Card><Statistic title="运行模式" value={asText(data.mode)} /></Card>
            <Card><Statistic title="文档数" value={Number(data.docs_count || data.docs_indexed || 0)} /></Card>
            <Card>
              <Space direction="vertical" size={4}>
                <span>状态</span>
                <Tag color={data.status === "ok" ? "green" : data.status === "degraded" ? "gold" : "red"}>{asText(data.status)}</Tag>
              </Space>
            </Card>
          </div>
          <Card title="健康状态">
            <Descriptions column={1} bordered size="small">
              {Object.entries(data).map(([key, value]) => (
                <Descriptions.Item key={key} label={key}>{asText(value)}</Descriptions.Item>
              ))}
            </Descriptions>
          </Card>
          {report.isLoading ? <LoadingState /> : report.isError ? <ErrorState error={report.error} /> : (
            <Card title="运行报告">
              <Tabs
                items={[
                  { key: "docs", label: "文档构建", children: <pre className="json-box">{JSON.stringify(report.data?.doc_build_report, null, 2)}</pre> },
                  { key: "index", label: "索引", children: <pre className="json-box">{JSON.stringify(report.data?.index_report, null, 2)}</pre> },
                  { key: "query", label: "查询", children: <pre className="json-box">{JSON.stringify(report.data?.query_report, null, 2)}</pre> },
                ]}
              />
            </Card>
          )}
        </Space>
      ) : null}
    </>
  );
}
