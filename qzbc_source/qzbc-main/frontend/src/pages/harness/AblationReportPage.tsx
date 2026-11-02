import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Button, Card, Space, Statistic, Table, Tag, Typography } from "antd";
import { FlaskConical, RefreshCw } from "lucide-react";

import { evaluationApi } from "../../api/evaluationApi";
import type { AblationExperiment } from "../../api/types";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { errorMessage } from "../../utils/errors";
import { percent } from "../../utils/format";


export function AblationReportPage() {
  const queryClient = useQueryClient();
  const latest = useQuery({ queryKey: ["evaluation", "ablation", "latest"], queryFn: evaluationApi.getLatestAblation, retry: false });
  const run = useMutation({
    mutationFn: () => evaluationApi.runAblation({ suite: "quick" }),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: ["evaluation", "ablation", "latest"] }),
  });
  const data = latest.data;
  const chartRows = (data?.experiments || []).filter((item) => item.name !== "template_vs_ollama");

  return (
    <>
      <PageHeader
        title="消融实验"
        subtitle="同一固定数据集下的架构模块贡献对比。"
        extra={(
          <>
            <Button icon={<RefreshCw size={16} />} onClick={() => latest.refetch()} aria-label="刷新消融报告" />
            <Button type="primary" icon={<FlaskConical size={16} />} loading={run.isPending} onClick={() => run.mutate()}>运行快速消融</Button>
          </>
        )}
      />
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        {run.isError ? <Alert type="error" showIcon message={errorMessage(run.error)} /> : null}
        {latest.isLoading ? <LoadingState /> : latest.isError && !data ? <ErrorState error={latest.error} /> : data ? (
          <>
            <div className="metric-grid">
              <Card><Statistic title="基线综合评分" value={percent(data.baseline.overall_score)} /></Card>
              <Card><Statistic title="消融组数" value={data.experiment_count} /></Card>
              <Card><Statistic title="基线用例" value={data.baseline.total_cases} /></Card>
              <Card><Statistic title="执行耗时" value={data.duration_ms} suffix="ms" /></Card>
            </div>
            <Card title="核心模块贡献">
              <Space direction="vertical" size={14} style={{ width: "100%" }}>
                {chartRows.map((item) => (
                  <div key={item.name} className="ablation-row">
                    <Typography.Text className="ablation-label">{item.label}</Typography.Text>
                    <div className="ablation-bars">
                      <div className="ablation-bar-track"><div className="ablation-bar baseline" style={{ width: `${item.baseline_value * 100}%` }} /></div>
                      <div className="ablation-bar-track"><div className="ablation-bar ablated" style={{ width: `${item.ablated_value * 100}%` }} /></div>
                    </div>
                    <Typography.Text strong>{percent(item.baseline_value)} → {percent(item.ablated_value)}</Typography.Text>
                  </div>
                ))}
                <Space><Tag color="green">基线</Tag><Tag color="red">消融后</Tag></Space>
              </Space>
            </Card>
            <Card title="消融结果">
              <Table<AblationExperiment>
                rowKey="name"
                size="small"
                pagination={false}
                scroll={{ x: 980 }}
                dataSource={data.experiments}
                columns={[
                  { title: "实验组", dataIndex: "label", width: 180, fixed: "left" },
                  { title: "目标指标", dataIndex: "target_metric", width: 240 },
                  { title: "Baseline", dataIndex: "baseline_value", width: 100, render: percent },
                  { title: "Ablated", dataIndex: "ablated_value", width: 100, render: percent },
                  { title: "变化", dataIndex: "delta", width: 100, render: (value: number) => <Tag color={value < 0 ? "red" : "blue"}>{value > 0 ? "+" : ""}{percent(value)}</Tag> },
                  { title: "结论", dataIndex: "proves", ellipsis: true },
                ]}
              />
            </Card>
            <Alert type="success" showIcon message={data.competition_summary} />
          </>
        ) : null}
      </Space>
    </>
  );
}
