import { useMutation, useQuery } from "@tanstack/react-query";
import { Alert, Button, Card, Descriptions, Space, Typography } from "antd";
import { useParams } from "react-router-dom";

import { studentApi } from "../../api/studentApi";
import { EvidencePacketCard } from "../../components/evidence/EvidencePacketCard";
import { GraphNodePath } from "../../components/evidence/GraphNodePath";
import { LLMResultCard } from "../../components/llm/LLMResultCard";
import { KnowledgeGraphView } from "../../components/knowledgeGraph";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { KnowledgeCardList } from "../../components/learning/KnowledgeCardList";
import { QuizCard } from "../../components/learning/QuizCard";
import { RiskCardList } from "../../components/prescription/RiskCardList";

export function StudentStandardAnalysisPage() {
  const caseId = Number(useParams().caseId);
  const analysis = useQuery({ queryKey: ["student", "analysis", caseId], queryFn: () => studentApi.getStandardAnalysis(caseId), enabled: Boolean(caseId) });
  const explanation = useMutation({ mutationFn: () => studentApi.generateTeachingExplanation(caseId) });
  if (analysis.isLoading) return <LoadingState />;
  if (analysis.isError) return <ErrorState error={analysis.error} />;
  const data = analysis.data;
  if (!data) return <ErrorState error={new Error("标准解析接口未返回有效内容，请刷新后重试。")} />;
  const sourceCase = data.anonymized_case || {};
  const isKnowledgeEvaluation = sourceCase.case_type === "mtcmb_student_evaluation" && sourceCase.source_id === "SRC_MTCMB";
  const training = sourceCase.training_payload && typeof sourceCase.training_payload === "object" ? sourceCase.training_payload as Record<string, unknown> : {};
  const options = Array.isArray(training.options) ? training.options.map(String).filter(Boolean) : [];
  const generated = explanation.data?.generated_response || data.generated_response;
  const llmResult = explanation.data?.llm_result || (isKnowledgeEvaluation ? undefined : data.llm_result);
  return (
    <>
      <PageHeader title="标准解析" subtitle={data.title} />
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        <Card title="解析">
          <Typography.Paragraph className="pre-wrap">{data.standard_analysis || "暂无标准解析"}</Typography.Paragraph>
        </Card>
        {isKnowledgeEvaluation ? (
          <Card
            title="题目与标准答案"
            extra={<Button type="primary" loading={explanation.isPending} onClick={() => explanation.mutate()}>使用 Ollama 生成教学讲解</Button>}
          >
            <Descriptions column={1} size="small" bordered>
              <Descriptions.Item label="题库来源">{String(training.original_source || "MTCMB")}</Descriptions.Item>
              <Descriptions.Item label="题目">{String(training.question || "原始题库未提供题干")}</Descriptions.Item>
              <Descriptions.Item label="选项">{options.length ? options.join("；") : "原始题库未提供选项"}</Descriptions.Item>
              <Descriptions.Item label="题库标准答案">{String(training.answer || "原始题库未提供标准答案")}</Descriptions.Item>
            </Descriptions>
            <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
              标准答案来自题库；Ollama 仅负责生成教学讲解，不会改写答案或把题库题当作患者病例。
            </Typography.Paragraph>
            {explanation.isError ? <Alert style={{ marginTop: 12 }} type="error" showIcon message="教学讲解生成失败，请确认 Ollama 已启动后重试。" /> : null}
          </Card>
        ) : null}
        <RiskCardList items={(data.risk_points as Array<Record<string, unknown>>) || []} />
        <EvidencePacketCard packet={data.evidence_packet} traceId={data.trace_id} />
        <LLMResultCard result={llmResult} generated={generated} />
        {data.graph_node_path?.length ? <Card title="图节点路径"><GraphNodePath nodes={data.graph_node_path} /></Card> : null}
        <Card title="案例关联知识图谱"><KnowledgeGraphView initialLabel={data.title || ""} compact /></Card>
        <Card title="知识卡"><KnowledgeCardList items={data.knowledge_cards || []} /></Card>
        <QuizCard items={data.quiz || []} />
      </Space>
    </>
  );
}
