import { useQuery } from "@tanstack/react-query";
import { Card, Descriptions, Space, Statistic } from "antd";

import { contentApi } from "../../api/contentApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { asText, percent } from "../../utils/format";

export function ContentStatsPage() {
  const query = useQuery({ queryKey: ["content", "stats"], queryFn: contentApi.getStats });
  const stats = query.data;
  const coverage = stats?.coverage || {};

  return (
    <>
      <PageHeader title="内容统计" subtitle="第 9 阶段内容资产规模、来源覆盖和规则证据绑定统计。" />
      {query.isLoading ? <LoadingState /> : query.isError ? <ErrorState error={query.error} /> : stats ? (
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <div className="metric-grid">
            <Card><Statistic title="知识条目" value={stats.knowledge_entries} /></Card>
            <Card><Statistic title="规则" value={stats.rules || 0} /></Card>
            <Card><Statistic title="学习案例" value={stats.learning_cases} /></Card>
            <Card><Statistic title="测验" value={stats.quizzes} /></Card>
          </div>
          <div className="metric-grid">
            <Card><Statistic title="来源覆盖率" value={percent(Number(coverage.source_coverage_rate ?? 0))} /></Card>
            <Card><Statistic title="规则证据绑定率" value={percent(Number(coverage.rule_source_binding_rate ?? 0))} /></Card>
            <Card><Statistic title="测验案例绑定率" value={percent(Number(coverage.quiz_case_binding_rate ?? 0))} /></Card>
            <Card><Statistic title="学生案例" value={stats.student_learning_cases} /></Card>
          </div>
          <Card title="内容分类统计">
            <Descriptions column={2} bordered size="small">
              {Object.entries(stats).filter(([key]) => key !== "coverage").map(([key, value]) => (
                <Descriptions.Item key={key} label={key}>{asText(value)}</Descriptions.Item>
              ))}
            </Descriptions>
          </Card>
        </Space>
      ) : null}
    </>
  );
}
