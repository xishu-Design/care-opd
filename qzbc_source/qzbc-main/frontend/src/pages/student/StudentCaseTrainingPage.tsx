import { useMutation, useQuery } from "@tanstack/react-query";
import { Alert, Button, Card, Descriptions, Divider, Form, Input, Radio, Space, Tag, Typography } from "antd";
import { useNavigate, useParams } from "react-router-dom";

import { studentApi } from "../../api/studentApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { ReviewAnswerForm, type ReviewAnswerValues } from "../../components/learning/ReviewAnswerForm";
import { errorMessage } from "../../utils/errors";

function textList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(String).filter(Boolean);
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

function prescriptionItems(value: unknown): Array<{ herbName: string; dosage?: string }> {
  const prescription = value && typeof value === "object" ? value as Record<string, unknown> : {};
  const items = Array.isArray(prescription.items) ? prescription.items : Array.isArray(value) ? value : [];
  return items.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const row = item as Record<string, unknown>;
    const herbName = String(row.herb_name || row.name || "").trim();
    const dosage = String(row.dosage || row.dose || "").trim();
    return herbName ? [{ herbName, dosage: dosage || undefined }] : [];
  });
}

function CaseOverview({ value, revealReference }: { value: Record<string, unknown>; revealReference: boolean }) {
  const profile = value.patient_profile && typeof value.patient_profile === "object" ? value.patient_profile as Record<string, unknown> : {};
  const symptoms = textList(value.symptoms || value.symptom_summary || value.chief_complaint || value.presenting_symptoms);
  const herbs = prescriptionItems(value.prescription);
  const knowledgePoints = textList(value.knowledge_points);
  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Descriptions column={1} size="small" bordered>
        <Descriptions.Item label="病例类型">{String(value.case_type || "脱敏教学案例")}</Descriptions.Item>
        <Descriptions.Item label="年龄">{profile.age !== null && profile.age !== undefined ? `${String(profile.age)}岁` : String(profile.age_range || "原始记录未提供")}</Descriptions.Item>
        <Descriptions.Item label="性别">{{ female: "女", male: "男" }[String(profile.gender || profile.sex)] || String(profile.gender || profile.sex || "原始记录未提供")}</Descriptions.Item>
        <Descriptions.Item label="症状/主诉">
          {symptoms.length ? symptoms.join("；") : "原始教学记录未提供症状，不能据处方反推或补造病例表现。"}
        </Descriptions.Item>
      </Descriptions>
      <div className="learning-section">
        <Typography.Text strong>{revealReference ? "参考处方组成" : "参考处方组成（提交后显示）"}</Typography.Text>
        {revealReference ? (herbs.length ? <div className="learning-herb-list">{herbs.map((item) => <Tag key={`${item.herbName}-${item.dosage || ""}`} color="blue">{item.herbName}{item.dosage ? `（${item.dosage}）` : ""}</Tag>)}</div> : <Typography.Text type="secondary">原始记录未提供处方组成。</Typography.Text>) : <Typography.Text type="secondary">请先依据病例资料独立拟方并提交；系统将在提交后展示参考组成与差异。</Typography.Text>}
      </div>
      {knowledgePoints.length ? <div className="learning-section"><Typography.Text strong>学习要点</Typography.Text><div className="learning-herb-list">{knowledgePoints.map((item) => <Tag key={item} color="purple">{item}</Tag>)}</div></div> : null}
    </Space>
  );
}

function optionRows(value: unknown): Array<{ value: string; label: string }> {
  if (Array.isArray(value)) return value.map((item, index) => ({ value: String(item).trim().charAt(0) || String(index + 1), label: String(item) }));
  if (value && typeof value === "object") return Object.entries(value as Record<string, unknown>).map(([key, item]) => ({ value: key, label: `${key}. ${String(item)}` }));
  return [];
}

/** Accept both `B` and `B. 某选项`, which are both used by the MTCMB source. */
function optionKey(value: unknown): string {
  const normalized = String(value || "").trim().toUpperCase();
  return normalized.match(/^([A-Z])(?:[.、．:：\s]|$)/)?.[1] || normalized;
}

function KnowledgeEvaluationOverview({ value }: { value: Record<string, unknown> }) {
  const training = value.training_payload && typeof value.training_payload === "object" ? value.training_payload as Record<string, unknown> : {};
  const options = optionRows(training.options);
  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="这是 MTCMB 题库知识测验，不是患者病例"
        description="原始题库只提供题干、选项和标准答案，不包含患者年龄、性别、症状或处方。系统不会补造这些病例信息。"
      />
      <Descriptions column={1} size="small" bordered>
        <Descriptions.Item label="题库来源">{String(training.original_source || "MTCMB")}</Descriptions.Item>
        <Descriptions.Item label="题目">{String(training.question || "原始题库未提供题干")}</Descriptions.Item>
      </Descriptions>
      {options.length ? <Radio.Group className="learning-question-options" options={options} disabled /> : null}
      <div className="learning-section"><Typography.Text strong>学习要点</Typography.Text><div className="learning-herb-list">{textList(value.knowledge_points).map((item) => <Tag key={item} color="purple">{item}</Tag>)}</div></div>
    </Space>
  );
}

type QuizEvaluation = {
  submitted_answer?: string;
  expected_answer?: string;
  is_correct?: boolean;
};

function KnowledgeEvaluationForm({ value, loading, evaluation, onSubmit }: { value: Record<string, unknown>; loading?: boolean; evaluation?: QuizEvaluation; onSubmit: (values: ReviewAnswerValues) => void }) {
  const training = value.training_payload && typeof value.training_payload === "object" ? value.training_payload as Record<string, unknown> : {};
  const options = optionRows(training.options);
  const submittedKey = optionKey(evaluation?.submitted_answer);
  const expectedKey = optionKey(evaluation?.expected_answer);
  return (
    <Form layout="vertical" onFinish={(values) => onSubmit({ answer_text: String(values.answer_text || ""), proposed_herbs: [], has_risk: false })}>
      <Form.Item name="answer_text" label="你的答案" rules={[{ required: true, message: "请选择或填写答案" }]}>
        {options.length ? <Radio.Group className="learning-question-options" disabled={Boolean(evaluation)}>
          {options.map((option) => {
            const key = optionKey(option.value);
            const isSubmitted = Boolean(evaluation) && key === submittedKey;
            const isCorrectOption = Boolean(evaluation) && key === expectedKey;
            const marker = isSubmitted
              ? evaluation?.is_correct
                ? <Tag color="success">✓ 你的答案正确</Tag>
                : <Tag color="error">✕ 你的答案错误</Tag>
              : isCorrectOption
                ? <Tag color="success">✓ 正确答案</Tag>
                : null;
            return <Radio value={option.value} key={option.value}>
              <Space size={6} wrap>
                <span>{option.label}</span>
                {marker}
              </Space>
            </Radio>;
          })}
        </Radio.Group> : <Input.TextArea rows={4} placeholder="填写题库答案" disabled={Boolean(evaluation)} />}
      </Form.Item>
      <Typography.Paragraph type="secondary">本题只按 MTCMB 标准答案进行教学对照，不会生成处方、风险判断或患者用药建议。</Typography.Paragraph>
      {!evaluation ? <Button type="primary" htmlType="submit" loading={loading}>提交答案</Button> : null}
    </Form>
  );
}

function CompositionLearningOverview({ value }: { value: Record<string, unknown> }) {
  const herbs = prescriptionItems(value.prescription);
  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="这是开源方剂组成复盘，不是患者病例或审方题"
        description="来源记录只提供方剂药材组成，没有患者年龄、性别、症状和风险信息；因此系统不会补造病例资料，也不要求填写风险等级。"
      />
      <Descriptions column={1} size="small" bordered>
        <Descriptions.Item label="学习任务">识别并复盘下方来源处方中的药材组成，再提交你的组成对照。</Descriptions.Item>
        <Descriptions.Item label="数据来源">{String(value.source_id || "开源方剂组成数据")}</Descriptions.Item>
      </Descriptions>
      <div className="learning-section">
        <Typography.Text strong>来源处方组成（用于复盘）</Typography.Text>
        {herbs.length ? <div className="learning-herb-list">{herbs.map((item) => <Tag key={`${item.herbName}-${item.dosage || ""}`} color="blue">{item.herbName}{item.dosage ? `（${item.dosage}）` : ""}</Tag>)}</div> : <Typography.Text type="secondary">原始记录未提供药材组成。</Typography.Text>}
      </div>
    </Space>
  );
}

function SyndromeDifferentiationOverview({ value }: { value: Record<string, unknown> }) {
  const training = value.training_payload && typeof value.training_payload === "object" ? value.training_payload as Record<string, unknown> : {};
  const clinicalData = String(training.clinical_data || "来源未提供完整临床资料");
  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="这是脱敏辨证训练，不是处方题"
        description="该公开基准提供年龄、性别、临床资料和中医证候标签，但没有处方组成。请填写你的中医证候辨析；系统只按来源标签对照，不会要求或生成药材、剂量。"
      />
      <Descriptions column={1} size="small" bordered>
        <Descriptions.Item label="临床资料">{clinicalData}</Descriptions.Item>
      </Descriptions>
      <div className="learning-section"><Typography.Text strong>学习要点</Typography.Text><div className="learning-herb-list">{textList(value.knowledge_points).map((item) => <Tag key={item} color="purple">{item}</Tag>)}</div></div>
    </Space>
  );
}

function SyndromeDifferentiationForm({ loading, onSubmit }: { loading?: boolean; onSubmit: (values: ReviewAnswerValues) => void }) {
  return (
    <Form layout="vertical" onFinish={(values) => onSubmit({ answer_text: String(values.answer_text || ""), proposed_herbs: [], has_risk: false })}>
      <Form.Item name="answer_text" label="你的中医证候辨析" rules={[{ required: true, message: "请填写中医证候辨析" }]}>
        <Input.TextArea rows={6} placeholder="例如：填写你判断的中医证候，并可简要说明依据。" />
      </Form.Item>
      <Typography.Paragraph type="secondary">正误只按公开来源提供的中医证候标签进行教学对照，不用于真实患者诊断或开方。</Typography.Paragraph>
      <Button type="primary" htmlType="submit" loading={loading}>提交辨证答案</Button>
    </Form>
  );
}

export function StudentCaseTrainingPage() {
  const caseId = Number(useParams().caseId);
  const navigate = useNavigate();
  const learningCase = useQuery({ queryKey: ["student", "case", caseId], queryFn: () => studentApi.getLearningCase(caseId), enabled: Boolean(caseId) });
  const quizCases = useQuery({ queryKey: ["student", "cases", "quiz", "next"], queryFn: () => studentApi.listLearningCases("quiz"), enabled: Boolean(caseId) });
  const submit = useMutation({ mutationFn: (values: ReviewAnswerValues) => studentApi.submitReviewAnswer(caseId, values) });
  if (learningCase.isLoading) return <LoadingState />;
  if (learningCase.isError) return <ErrorState error={learningCase.error} />;
  const data = learningCase.data!;
  const isKnowledgeEvaluation = data.anonymized_case.case_type === "mtcmb_student_evaluation" && data.anonymized_case.source_id === "SRC_MTCMB";
  const isCompositionLearning = data.anonymized_case.case_type === "prescription_composition_learning_candidate";
  const isSyndromeTraining = data.anonymized_case.case_type === "syndrome_differentiation_training";
  const quizCaseIds = quizCases.data?.map((item) => item.id) || [];
  const currentQuizIndex = quizCaseIds.indexOf(caseId);
  const previousQuizId = currentQuizIndex >= 0 && quizCaseIds.length > 1 ? quizCaseIds[(currentQuizIndex - 1 + quizCaseIds.length) % quizCaseIds.length] : undefined;
  const nextQuizId = currentQuizIndex >= 0 && quizCaseIds.length > 1 ? quizCaseIds[(currentQuizIndex + 1) % quizCaseIds.length] : undefined;
  const judgment = submit.data?.evaluation.judgment;
  const resultAlertType = judgment === "correct" ? "success" : judgment === "partial" ? "warning" : judgment === "incorrect" ? "error" : "info";
  return (
    <>
      <PageHeader title={isKnowledgeEvaluation ? "中医知识测验" : isCompositionLearning ? "方剂组成复盘" : isSyndromeTraining ? "中医辨证训练" : "拟方训练"} subtitle={data.title} extra={<Button onClick={() => navigate(`/student/cases/${caseId}/analysis`)}>标准解析</Button>} />
      <div className="two-column">
        <Space direction="vertical" size={16}>
          <Card title={isKnowledgeEvaluation ? "教学题目" : isCompositionLearning ? "组成复盘材料" : isSyndromeTraining ? "脱敏病例资料" : "脱敏病例"}>
            {isKnowledgeEvaluation ? <KnowledgeEvaluationOverview value={data.anonymized_case} /> : isCompositionLearning ? <CompositionLearningOverview value={data.anonymized_case} /> : isSyndromeTraining ? <SyndromeDifferentiationOverview value={data.anonymized_case} /> : <CaseOverview value={data.anonymized_case} revealReference={submit.isSuccess} />}
          </Card>
          <Card title={isKnowledgeEvaluation ? "题库使用边界" : isCompositionLearning ? "组成学习边界" : isSyndromeTraining ? "辨证训练边界" : "证据链摘要"}>
            <Typography.Text>{isKnowledgeEvaluation ? "该题来自 MTCMB，仅用于中医知识学习和评测；题库答案不是患者诊疗或处方依据。" : isCompositionLearning ? "该材料仅用于药材组成识别与复盘；没有患者临床资料，不能据此诊断、审方或形成处方建议。" : isSyndromeTraining ? "该案例仅用于中医病机和证候辨析；公开来源未提供处方组成，不能据此反推、生成或评价临床方剂。" : `该案例来自 ${String(data.anonymized_case.source_id || "已脱敏教学来源")}；仅用于教学中的处方组成识别与复盘，不构成临床处方建议。`}</Typography.Text>
          </Card>
        </Space>
        <Card title={isKnowledgeEvaluation ? "提交测验答案" : isCompositionLearning ? "提交组成对照" : isSyndromeTraining ? "提交辨证答案" : "提交拟方答案"}>
          {submit.isError ? <Alert type="error" showIcon message={errorMessage(submit.error)} /> : null}
          {submit.isSuccess ? <>
            <Alert type={resultAlertType} showIcon message={submit.data.feedback} style={{ marginBottom: 16 }} />
            {!isKnowledgeEvaluation ? <div className="learning-feedback">
              <Typography.Text strong>教学对照</Typography.Text>
              <div className="learning-herb-list">{submit.data.evaluation.matched_herbs.map((item) => <Tag color="green" key={item}>{isSyndromeTraining ? `命中：${item}` : `一致：${item}`}</Tag>)}{submit.data.evaluation.missing_herbs.map((item) => <Tag color="gold" key={item}>待复盘：{item}</Tag>)}{submit.data.evaluation.extra_herbs.map((item) => <Tag color="red" key={item}>额外：{item}</Tag>)}</div>
              {submit.data.evaluation.semantic_assessments?.length ? <Typography.Paragraph type="secondary" style={{ marginTop: 12, whiteSpace: "pre-line" }}>
                {submit.data.evaluation.semantic_assessments.map((item) => `${item.herb_name}：${item.verdict === "related_but_not_equivalent" ? `与本病例“${item.direct_symptom_matches?.join("、") || item.function_matches?.join("、") || "已收录关系"}”存在可追溯关联；本地收录功效：${item.known_functions?.slice(0, 5).join("、") || "未收录"}。这仅说明可作为学习比较，不能替代参考药材。` : `未检索到与当前症状或参考功效的直接可追溯关联；本地收录功效：${item.known_functions?.slice(0, 5).join("、") || "未收录"}。`}`).join("\n")}
              </Typography.Paragraph> : null}
              <Divider />
              <Typography.Text strong>对应知识点</Typography.Text>
              <div className="learning-herb-list">{submit.data.evaluation.knowledge_points.map((item) => <Tag color="purple" key={item}>{item}</Tag>)}</div>
              <Typography.Paragraph type="secondary" style={{ marginTop: 12 }}>{submit.data.evaluation.model_feedback}{submit.data.evaluation.model_provider ? `（${submit.data.evaluation.model_provider}）` : ""}</Typography.Paragraph>
            </div> : <Space direction="vertical" size={12}>
              <Typography.Paragraph type="secondary" style={{ margin: 0 }}>{submit.data.evaluation.model_feedback}</Typography.Paragraph>
              <Space>
                <Button disabled={!previousQuizId} onClick={() => previousQuizId && navigate(`/student/cases/${previousQuizId}/training`)}>
                  {previousQuizId ? "上一道题" : "暂无上一道题"}
                </Button>
                <Button type="primary" disabled={!nextQuizId} onClick={() => nextQuizId && navigate(`/student/cases/${nextQuizId}/training`)}>
                  {nextQuizId ? "下一道题" : "暂无下一道题"}
                </Button>
              </Space>
            </Space>}
          </> : null}
          {isKnowledgeEvaluation ? <KnowledgeEvaluationForm value={data.anonymized_case} loading={submit.isPending} evaluation={submit.data?.evaluation} onSubmit={(values) => submit.mutate(values)} /> : isSyndromeTraining ? <SyndromeDifferentiationForm loading={submit.isPending} onSubmit={(values) => submit.mutate(values)} /> : <ReviewAnswerForm mode={isCompositionLearning ? "composition" : "review"} loading={submit.isPending} onSubmit={(values) => submit.mutate(values)} />}
        </Card>
      </div>
    </>
  );
}
