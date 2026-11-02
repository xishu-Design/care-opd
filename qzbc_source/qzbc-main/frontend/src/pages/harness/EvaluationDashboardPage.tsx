import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Button, Card, Select, Space, Statistic, Table, Tag, Typography } from "antd";
import { Download, Play, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { evaluationApi } from "../../api/evaluationApi";
import type { EvaluationResultItem, EvaluationRun } from "../../api/types";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { MetricBars } from "../../components/evaluation/MetricBars";
import { errorMessage } from "../../utils/errors";
import { percent, shortId } from "../../utils/format";


export function EvaluationDashboardPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [runId, setRunId] = useState<string>();
  const runs = useQuery({ queryKey: ["evaluation", "runs"], queryFn: evaluationApi.listRuns });
  useEffect(() => {
    if (!runId && runs.data?.[0]) setRunId(runs.data[0].run_id);
  }, [runId, runs.data]);
  const report = useQuery({
    queryKey: ["evaluation", "run", runId],
    queryFn: () => evaluationApi.getRun(runId!),
    enabled: Boolean(runId),
  });
  const run = useMutation({
    mutationFn: (suite: "quick" | "full") => evaluationApi.runEvaluation({ suite, reset_demo_data: true }),
    onSuccess: async (data) => {
      setRunId(data.run_id);
      await queryClient.invalidateQueries({ queryKey: ["evaluation", "runs"] });
      await queryClient.invalidateQueries({ queryKey: ["evaluation", "run", data.run_id] });
    },
  });
  const data = report.data;
  const scoreBars = useMemo(() => data ? [
    { label: "安全", value: data.safety_score, color: "#2f7d62" },
    { label: "证据", value: data.evidence_score, color: "#3778a8" },
    { label: "智能体链路", value: data.agent_trace_score, color: "#8a6a2f" },
    { label: "LLM 安全", value: data.llm_safety_score, color: "#8b4f71" },
  ] : [], [data]);
  const failures = data?.results?.filter((item) => !item.passed) || data?.key_failures || [];

  return (
    <>
      <PageHeader
        title="系统评测"
        subtitle="第 12 阶段固定数据集、核心指标与演示文稿就绪报告。"
        extra={(
          <>
            <Select
              value={runId}
              placeholder="选择评测运行"
              style={{ width: 220 }}
              options={(runs.data || []).map((item) => ({ value: item.run_id, label: `${item.suite} · ${shortId(item.run_id)}` }))}
              onChange={setRunId}
            />
            <Button icon={<RefreshCw size={16} />} onClick={() => runs.refetch()} aria-label="刷新评测列表" />
            <Button icon={<Play size={16} />} loading={run.isPending} onClick={() => run.mutate("quick")}>快速评测</Button>
            <Button type="primary" icon={<Play size={16} />} loading={run.isPending} onClick={() => run.mutate("full")}>完整评测</Button>
          </>
        )}
      />
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        {run.isError ? <Alert type="error" showIcon message={errorMessage(run.error)} /> : null}
        {report.isLoading ? <LoadingState /> : report.isError ? <ErrorState error={report.error} /> : data ? (
          <>
            <div className="metric-grid">
              <Card><Statistic title="总通过率" value={percent(data.pass_rate)} /></Card>
              <Card><Statistic title="综合评分" value={percent(data.overall_score)} /></Card>
              <Card><Statistic title="评测用例" value={data.total_cases} /></Card>
              <Card><Statistic title="失败案例" value={data.failed} valueStyle={{ color: data.failed ? "#a33d34" : "#2f7d62" }} /></Card>
            </div>
            <div className="two-column">
              <Card title="能力评分"><MetricBars items={scoreBars} /></Card>
              <Card title="安全指标"><MetricBars items={metricItems(data.metrics.safety)} /></Card>
            </div>
            <div className="two-column">
              <Card title="证据与检索"><MetricBars items={metricItems(data.metrics.evidence)} /></Card>
              <Card title="智能体与 LLM"><MetricBars items={[...metricItems(data.metrics.agent), ...metricItems(data.metrics.llm).filter((item) => item.label !== "fallback_used_rate")]} /></Card>
            </div>
            <Card
              title="关键失败案例"
              extra={<Button icon={<Download size={16} />} disabled={!data.markdown_report} onClick={() => downloadReport(data)}>下载 Markdown</Button>}
            >
              <Table<EvaluationResultItem>
                rowKey="case_id"
                size="small"
                pagination={{ pageSize: 8 }}
                dataSource={failures}
                columns={[
                  { title: "Case", dataIndex: "case_id", width: 130 },
                  { title: "数据集", dataIndex: "dataset", ellipsis: true },
                  { title: "指标", dataIndex: "metric_name", ellipsis: true },
                  { title: "级别", dataIndex: "severity", width: 90, render: (value: string) => <Tag color={value === "critical" ? "red" : "orange"}>{value}</Tag> },
                  { title: "原因", dataIndex: "failure_reason", ellipsis: true },
                  { title: "链路", dataIndex: "trace_id", width: 120, render: (value?: string) => value ? <Button type="link" onClick={() => navigate(`/harness/traces/${value}`)}>{shortId(value)}</Button> : "-" },
                ]}
                locale={{ emptyText: "无失败案例" }}
              />
            </Card>
            <Card title="比赛摘要"><Typography.Paragraph className="pre-wrap">{data.competition_summary}</Typography.Paragraph></Card>
          </>
        ) : <Card><Typography.Text type="secondary">暂无评测运行。</Typography.Text></Card>}
      </Space>
    </>
  );
}

function metricItems(metrics: Record<string, number> | undefined) {
  return Object.entries(metrics || {}).map(([label, value]) => ({ label, value: label.endsWith("leak_rate") || label === "fallback_used_rate" ? 1 - Number(value) : Number(value) }));
}

function downloadReport(data: EvaluationRun) {
  const blob = new Blob([data.markdown_report || ""], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `guardtcm-evaluation-${data.run_id}.md`;
  anchor.click();
  URL.revokeObjectURL(url);
}
