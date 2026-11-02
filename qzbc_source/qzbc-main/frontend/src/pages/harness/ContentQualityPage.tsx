import { useQuery } from "@tanstack/react-query";
import { Card, Descriptions, List, Space, Statistic, Typography } from "antd";

import { harnessApi } from "../../api/harnessApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { asText, percent } from "../../utils/format";

function IssueList({ title, items = [] }: { title: string; items?: Array<Record<string, unknown>> }) {
  return (
    <Card title={title}>
      <List
        size="small"
        locale={{ emptyText: "none" }}
        dataSource={items}
        renderItem={(item) => (
          <List.Item>
            <Typography.Text className="pre-wrap">{asText(item)}</Typography.Text>
          </List.Item>
        )}
      />
    </Card>
  );
}

export function ContentQualityPage() {
  const query = useQuery({ queryKey: ["harness", "phase9", "content-quality"], queryFn: harnessApi.getDataPipeline });
  const quality = query.data?.content_quality;
  const metrics = quality?.metrics || {};

  return (
    <>
      <PageHeader title="内容质量" subtitle="第 9 阶段内容数据覆盖率、安全校验和脱敏指标。" />
      {query.isLoading ? <LoadingState /> : query.isError ? <ErrorState error={query.error} /> : quality ? (
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <div className="metric-grid">
            <Card><Statistic title="来源覆盖率" value={percent(Number(metrics.content_source_coverage_rate ?? 0))} /></Card>
            <Card><Statistic title="测验绑定率" value={percent(Number(metrics.quiz_case_binding_rate ?? 0))} /></Card>
            <Card><Statistic title="脱敏通过率" value={percent(Number(metrics.learning_case_deid_pass_rate ?? 0))} /></Card>
            <Card><Statistic title="严重问题" value={Number(metrics.critical_issue_count || 0)} /></Card>
          </div>
          <Card title="质量指标">
            <Descriptions column={2} size="small" bordered>
              {Object.entries(metrics).map(([key, value]) => (
                <Descriptions.Item key={key} label={key}>{asText(value)}</Descriptions.Item>
              ))}
            </Descriptions>
          </Card>
          <div className="two-column">
            <IssueList title="严重问题" items={quality.critical_issues} />
            <IssueList title="高风险问题" items={quality.high_issues} />
          </div>
          <IssueList title="警告" items={quality.warnings} />
        </Space>
      ) : null}
    </>
  );
}
