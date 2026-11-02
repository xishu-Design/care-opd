import { useQuery } from "@tanstack/react-query";
import { Button, Card, Statistic } from "antd";
import { useNavigate } from "react-router-dom";

import { studentApi } from "../../api/studentApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";

export function StudentDashboardPage() {
  const navigate = useNavigate();
  const cases = useQuery({ queryKey: ["student", "cases", "clinical"], queryFn: () => studentApi.listLearningCases("clinical") });
  const progress = useQuery({ queryKey: ["student", "learning-progress"], queryFn: studentApi.getLearningProgress });
  if (cases.isLoading) return <LoadingState />;
  if (cases.isError) return <ErrorState error={cases.error} />;
  return (
    <>
      <PageHeader title="学习总览" extra={<Button type="primary" onClick={() => navigate("/student/cases")}>进入训练</Button>} />
      <div className="metric-grid">
        <Card><Statistic title="脱敏案例" value={cases.data?.length || 0} /></Card>
        <Card><Statistic title="已完成训练" value={progress.data?.completed_training_count || 0} /></Card>
        <Card><Statistic title="待复习错题" value={progress.data?.wrong_quiz_count || 0} /></Card>
      </div>
    </>
  );
}
