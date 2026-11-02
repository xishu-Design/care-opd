import { useMutation, useQuery } from "@tanstack/react-query";
import { Alert, Button, Card, Descriptions, List, Space, Statistic, Typography } from "antd";
import { Play } from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { harnessApi } from "../../api/harnessApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { percent, shortId, asText } from "../../utils/format";
import { errorMessage } from "../../utils/errors";

export function HarnessReportPage() {
  const navigate = useNavigate();
  const routeRunId = Number(useParams().runId || 0);
  const [runId, setRunId] = useState<number | null>(routeRunId || null);
  const run = useMutation({
    mutationFn: () => harnessApi.runHarness({ phase: "phase_11_demo_readiness", suite: "quick", reset_demo_data: true }),
    onSuccess: (data) => {
      setRunId(data.run_id);
      navigate(`/harness/reports/${data.run_id}`);
    },
  });
  const report = useQuery({
    queryKey: ["harness", "report", runId],
    queryFn: () => harnessApi.getHarnessReport(runId!),
    enabled: Boolean(runId),
  });

  return (
    <>
      <PageHeader
        title="验收报告"
        subtitle="第 11 阶段演示验收、演示就绪状态与冒烟检查报告。"
        extra={<Button type="primary" icon={<Play size={16} />} loading={run.isPending} onClick={() => run.mutate()}>运行快速套件</Button>}
      />
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        {run.isError ? <Alert type="error" showIcon message={errorMessage(run.error)} /> : null}
        {report.isLoading ? <LoadingState /> : report.isError ? <ErrorState error={report.error} /> : report.data ? (
          <>
            <div className="metric-grid">
              <Card><Statistic title="通过率" value={percent(Number(report.data.summary.pass_rate))} /></Card>
              <Card><Statistic title="总用例" value={Number(report.data.summary.total_cases || 0)} /></Card>
              <Card><Statistic title="失败" value={Number(report.data.summary.failed || 0)} /></Card>
            </div>
            <Card title="指标">
              <Descriptions column={2} size="small" bordered>
                {Object.entries(report.data.competition_metrics || {}).map(([key, value]) => (
                  <Descriptions.Item key={key} label={key}>{asText(value)}</Descriptions.Item>
                ))}
              </Descriptions>
            </Card>
            <Card title="关键链路追踪">
              <List
                dataSource={report.data.key_traces}
                renderItem={(traceId) => (
                  <List.Item actions={[<Button key="trace" onClick={() => navigate(`/harness/traces/${traceId}`)}>查看</Button>]}>
                    <Typography.Text>{shortId(traceId)}</Typography.Text>
                  </List.Item>
                )}
              />
            </Card>
            <Card title="报告摘要">
              <Typography.Paragraph className="pre-wrap">{report.data.markdown_summary}</Typography.Paragraph>
            </Card>
          </>
        ) : (
          <Card>
            <Typography.Text type="secondary">运行验收套件后显示报告。</Typography.Text>
          </Card>
        )}
      </Space>
    </>
  );
}
