import { useQuery } from "@tanstack/react-query";
import { Button, Card, Empty, Space, Table, Tag, Typography } from "antd";
import { useNavigate, useSearchParams } from "react-router-dom";

import { studentApi } from "../../api/studentApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";

function questionOf(row: Record<string, unknown>) {
  const source = row.anonymized_case && typeof row.anonymized_case === "object" ? row.anonymized_case as Record<string, unknown> : {};
  const training = source.training_payload && typeof source.training_payload === "object" ? source.training_payload as Record<string, unknown> : {};
  return String(training.question || "原始题库未提供题干");
}

function sourceOf(row: Record<string, unknown>) {
  const source = row.anonymized_case && typeof row.anonymized_case === "object" ? row.anonymized_case as Record<string, unknown> : {};
  const training = source.training_payload && typeof source.training_payload === "object" ? source.training_payload as Record<string, unknown> : {};
  return String(training.original_source || "MTCMB");
}

export function StudentQuizPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const cases = useQuery({ queryKey: ["student", "cases", "quiz"], queryFn: () => studentApi.listLearningCases("quiz") });
  const progress = useQuery({ queryKey: ["student", "learning-progress"], queryFn: studentApi.getLearningProgress });
  if (cases.isLoading) return <LoadingState />;
  if (cases.isError) return <ErrorState error={cases.error} />;
  const wrongOnly = searchParams.get("mode") === "wrong";
  const wrongIds = new Set(progress.data?.wrong_quiz_case_ids || []);
  const quizzes = (cases.data || []).filter((item) =>
    item.anonymized_case?.case_type === "mtcmb_student_evaluation"
    && item.anonymized_case?.source_id === "SRC_MTCMB"
    && (!wrongOnly || wrongIds.has(item.id))
  );
  return (
    <>
      <PageHeader title={wrongOnly ? "错题复习" : "测验"} subtitle={wrongOnly ? "仅显示你最近一次作答仍不正确的题目。" : "MTCMB 题库知识测验；提交答案后会计入学习进度。"} />
      <Card>
        {quizzes.length ? (
          <Table
            rowKey="id"
            dataSource={quizzes}
            pagination={{ pageSize: 12, showSizeChanger: false }}
            columns={[
              { title: "题目", render: (_, row) => <Typography.Text>{questionOf(row as unknown as Record<string, unknown>)}</Typography.Text> },
              { title: "来源", width: 220, render: (_, row) => sourceOf(row as unknown as Record<string, unknown>) },
              { title: "操作", width: 140, render: (_, row) => <Button type="link" onClick={() => navigate(`/student/cases/${(row as typeof quizzes[number]).id}/training`)}>开始答题</Button> },
            ]}
          />
        ) : <Empty description={wrongOnly ? "当前没有待复习错题。" : "当前没有可用的 MTCMB 测验题。"} />}
      </Card>
    </>
  );
}
