import { useQuery } from "@tanstack/react-query";
import { Button, Space, Table, Tag, Typography } from "antd";
import { useNavigate } from "react-router-dom";

import { studentApi } from "../../api/studentApi";
import { EmptyState } from "../../components/common/EmptyState";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";

export function StudentCaseListPage({ mode = "clinical" }: { mode?: "clinical" | "syndrome" }) {
  const navigate = useNavigate();
  const isSyndrome = mode === "syndrome";
  const cases = useQuery({ queryKey: ["student", "cases", mode], queryFn: () => studentApi.listLearningCases(mode) });
  if (cases.isLoading) return <LoadingState />;
  if (cases.isError) return <ErrorState error={cases.error} />;
  const learningType = (row: { anonymized_case?: Record<string, unknown> }) => {
    const type = row.anonymized_case?.case_type;
    if (type === "mtcmb_student_evaluation") return <Tag color="blue">知识测验</Tag>;
    if (type === "syndrome_differentiation_training") return <Tag color="cyan">辨证训练</Tag>;
    if (type === "prescription_composition_learning_candidate") return <Tag color="gold">方剂组成学习</Tag>;
    return <Tag color="green">审方训练</Tag>;
  };
  return (
    <>
      <PageHeader
        title={isSyndrome ? "脱敏辨证案例" : "脱敏临床拟方案例"}
        subtitle={isSyndrome ? "来自 CC BY 4.0 的 TCMEval-SDT；包含年龄、性别、症状与来源辨证标签，但不含处方，专用于辨证训练。" : "仅展示含年龄、性别、症状和参考处方的脱敏临床拟方训练案例；知识测验请进入“测验”。"}
      />
      {cases.data?.length ? (
        <section className="data-workbench">
          <Table
            className="data-workbench-table"
            rowKey="id"
            dataSource={cases.data}
            pagination={{ pageSize: 12, showSizeChanger: false }}
            columns={[
              { title: "案例编号", dataIndex: "id", width: 110 },
              { title: "案例名称", dataIndex: "title", render: (value: string) => <Typography.Text strong>{value}</Typography.Text> },
              { title: "学习类型", width: 150, render: (_, row) => learningType(row) },
              { title: "来源", width: 220, render: (_, row) => String(row.anonymized_case?.source_id || "已脱敏学习案例") },
              { title: "操作", width: 190, render: (_, row) => <Space><Button type="link" onClick={() => navigate(`/student/cases/${row.id}/training`)}>{isSyndrome ? "辨证训练" : row.anonymized_case?.case_type === "mtcmb_student_evaluation" ? "答题" : row.anonymized_case?.case_type === "prescription_composition_learning_candidate" ? "复盘" : "拟方训练"}</Button><Button type="link" onClick={() => navigate(`/student/cases/${row.id}/analysis`)}>标准解析</Button></Space> },
            ]}
          />
        </section>
      ) : (
        <EmptyState label={isSyndrome ? "暂无可用的脱敏辨证案例" : "暂无可用的脱敏临床拟方案例"} description={isSyndrome ? "请先导入经许可的脱敏辨证数据集。" : "需要导入并人工复核含年龄、性别、症状和处方组成的案例后才会出现在这里。"} />
      )}
    </>
  );
}
