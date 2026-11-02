import { useQuery } from "@tanstack/react-query";
import { Alert, Button, Card, List, Space, Statistic, Tag, Typography } from "antd";
import { useNavigate } from "react-router-dom";

import { studentApi } from "../../api/studentApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";

export function StudentProgressPage() {
  const navigate = useNavigate();
  const progress = useQuery({ queryKey: ["student", "learning-progress"], queryFn: studentApi.getLearningProgress });
  if (progress.isLoading) return <LoadingState />;
  if (progress.isError) return <ErrorState error={progress.error} />;
  const data = progress.data!;
  return (
    <>
      <PageHeader title="学习进度" subtitle="根据你已经提交并保存的训练答案统计。" />
      <div className="metric-grid">
        <Card><Statistic title="完成训练" value={data.completed_training_count} suffix="个案例" /></Card>
        <Card><Statistic title="知识测验错题" value={data.wrong_quiz_count} suffix="题" /></Card>
        <Card><Statistic title="知识测验正确率" value={data.quiz_accuracy ?? "-"} suffix={data.quiz_accuracy === null ? "" : "%"} /></Card>
      </div>
      <Card
        title="复习建议"
        style={{ marginTop: 16 }}
        extra={data.wrong_quiz_count ? <Button type="primary" onClick={() => navigate("/student/quiz?mode=wrong")}>只做错题</Button> : null}
      >
        <Alert type="info" showIcon message={data.review_suggestion} />
      </Card>
      <Card title="最近提交" style={{ marginTop: 16 }}>
        <List
          locale={{ emptyText: "暂无提交记录。完成一次训练并点击“提交答案”后，这里会自动出现记录。" }}
          dataSource={data.recent_records}
          renderItem={(item) => (
            <List.Item>
              <Space direction="vertical" size={2}>
                <Typography.Text strong>{item.case_title}</Typography.Text>
                <Space wrap>
                  <Tag>{item.kind}</Tag>
                  {item.correct === true ? <Tag color="green">答对</Tag> : null}
                  {item.correct === false ? <Tag color="red">待复习</Tag> : null}
                  <Typography.Text type="secondary">{new Date(item.created_at).toLocaleString("zh-CN")}</Typography.Text>
                </Space>
              </Space>
            </List.Item>
          )}
        />
      </Card>
    </>
  );
}
