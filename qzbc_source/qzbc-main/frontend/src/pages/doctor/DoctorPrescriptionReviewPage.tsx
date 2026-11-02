import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Button, Card, Descriptions, Form, Input, Modal, Popconfirm, Space, Typography, message } from "antd";
import { Check, FilePenLine, FileUp, Play, RotateCcw, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import { doctorApi } from "../../api/doctorApi";
import type { ReviewResult } from "../../api/types";
import { EvidencePacketCard } from "../../components/evidence/EvidencePacketCard";
import { GraphNodePath } from "../../components/evidence/GraphNodePath";
import { LLMResultCard } from "../../components/llm/LLMResultCard";
import { KnowledgeGraphView } from "../../components/knowledgeGraph";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { StatusTag } from "../../components/common/StatusTag";
import { PrescriptionItemsTable } from "../../components/prescription/PrescriptionItemsTable";
import { RiskCardList } from "../../components/prescription/RiskCardList";
import { useUiStore } from "../../stores/uiStore";
import { errorMessage } from "../../utils/errors";

export function DoctorPrescriptionReviewPage() {
  const prescriptionId = Number(useParams().prescriptionId);
  const [params] = useSearchParams();
  const autoStarted = useRef(false);
  const [review, setReview] = useState<ReviewResult | null>(null);
  const [comment, setComment] = useState("同意本次审方结果");
  const [usageModalOpen, setUsageModalOpen] = useState(false);
  const [usageForm] = Form.useForm<{ usage_method: string }>();
  const queryClient = useQueryClient();
  const openTrace = useUiStore((state) => state.openTrace);
  const [api, holder] = message.useMessage();

  const prescription = useQuery({
    queryKey: ["doctor", "prescription", prescriptionId],
    queryFn: () => doctorApi.getPrescriptionDetail(prescriptionId),
    enabled: Boolean(prescriptionId),
  });
  const runReview = useMutation({
    mutationFn: () => doctorApi.reviewPrescription(prescriptionId),
    onSuccess: (data) => setReview(data),
  });
  const updateUsage = useMutation({
    mutationFn: (payload: { usage_method: string }) => doctorApi.updatePrescription(prescriptionId, payload),
    onSuccess: () => {
      setUsageModalOpen(false);
      setReview(null);
      queryClient.invalidateQueries({ queryKey: ["doctor", "prescription", prescriptionId] });
      api.success("处方用法已保存，请重新运行审方");
    },
  });
  const resume = useMutation({
    mutationFn: (payload: { approved: boolean }) => doctorApi.resumeDoctorReview(review!.trace_id!, { ...payload, comment }),
    onSuccess: (data) => {
      setReview((prev) => ({ ...(prev || data), ...data, human_confirmed: data.human_confirmed, resume_action: data.resume_action }));
      queryClient.invalidateQueries({ queryKey: ["doctor", "prescription", prescriptionId] });
      api.success(data.resume_action === "approved" ? "已确认审方" : "已驳回审方");
    },
  });
  const publish = useMutation({
    mutationFn: () => doctorApi.publishPrescription(prescriptionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["doctor", "prescription", prescriptionId] });
      api.success("已发布给患者");
    },
  });
  const deidentify = useMutation({
    mutationFn: () => doctorApi.deidentifyPrescription(prescriptionId),
    onSuccess: () => api.success("已生成学习案例"),
  });

  useEffect(() => {
    if (params.get("autoReview") === "1" && !autoStarted.current && prescription.data) {
      autoStarted.current = true;
      runReview.mutate();
    }
  }, [params, prescription.data, runReview]);

  if (prescription.isLoading) return <LoadingState />;
  if (prescription.isError) return <ErrorState error={prescription.error} />;
  const data = prescription.data!;
  const riskItems = review?.risk_cards?.length ? review.risk_cards : review?.rule_findings || [];
  const traceId = review?.trace_id;

  return (
    <>
      {holder}
      <PageHeader
        title="处方审方"
        subtitle={data.title}
        extra={
          <Space wrap>
            <Button
              icon={<FilePenLine size={16} />}
              disabled={!["draft", "reviewing"].includes(prescription.data?.status || "")}
              onClick={() => {
                usageForm.setFieldsValue({ usage_method: prescription.data?.usage_method || "" });
                setUsageModalOpen(true);
              }}
            >
              补充处方用法
            </Button>
            <Button type="primary" icon={<Play size={16} />} loading={runReview.isPending} onClick={() => runReview.mutate()}>
              运行审方
            </Button>
            {traceId ? <Button onClick={() => openTrace(traceId)}>查看链路追踪</Button> : null}
          </Space>
        }
      />
      <div className="two-column">
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Card title="处方信息">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="状态"><StatusTag value={data.status} /></Descriptions.Item>
              <Descriptions.Item label="patient_id">{data.patient_id}</Descriptions.Item>
              <Descriptions.Item label="患者年龄（就诊时）">{data.patient_age_at_visit ?? "未记录"}</Descriptions.Item>
              <Descriptions.Item label="患者性别（就诊时）">{{ female: "女", male: "男" }[String(data.patient_sex_at_visit)] || data.patient_sex_at_visit || "未记录"}</Descriptions.Item>
              <Descriptions.Item label="症状与诊疗背景" span={2}>{data.symptom_context || "未填写"}</Descriptions.Item>
              <Descriptions.Item label="处方用法">{data.usage_method || "未填写（单味药剂量见下表）"}</Descriptions.Item>
              <Descriptions.Item label="备注">{data.doctor_notes || "-"}</Descriptions.Item>
            </Descriptions>
            <PrescriptionItemsTable items={data.items} />
          </Card>
          {review?.graph_node_path?.length ? (
            <Card title="图节点路径">
              <GraphNodePath nodes={review.graph_node_path} />
            </Card>
          ) : null}
        </Space>
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          {runReview.isError ? <Alert type="error" showIcon message={errorMessage(runReview.error)} /> : null}
          {resume.isError ? <Alert type="error" showIcon message={errorMessage(resume.error)} /> : null}
          {publish.isError ? <Alert type="error" showIcon message={errorMessage(publish.error)} /> : null}
          {review ? (
            <>
              <Card title="审方结果">
                <Space direction="vertical" size={12} style={{ width: "100%" }}>
                  <Space wrap>
                    <StatusTag value={review.risk_level || "yellow"} kind="risk" />
                    <StatusTag value={review.pending_action_status || data.status} />
                    {review.interrupted ? <StatusTag value="interrupted" /> : null}
                  </Space>
                  <Typography.Paragraph className="pre-wrap">{review.review_summary || "等待医生复核。"}</Typography.Paragraph>
                  <RiskCardList items={riskItems} />
                </Space>
              </Card>
              <EvidencePacketCard packet={review.evidence_packet} traceId={traceId} />
              <Card title="本次审方关联知识图谱"><KnowledgeGraphView initialLabel={data.items?.[0]?.herb_name || ""} compact /></Card>
              <LLMResultCard result={review.llm_result} generated={review.generated_response} />
              {review.interrupted && !review.human_confirmed && review.resume_action !== "rejected" ? (
                <Card title="人工确认">
                  <Space direction="vertical" size={12} style={{ width: "100%" }}>
                    <Alert type="warning" showIcon message="审方流程已暂停，等待医生人工确认" />
                    <Input.TextArea value={comment} onChange={(event) => setComment(event.target.value)} rows={3} />
                    <Space wrap>
                      <Button type="primary" icon={<Check size={16} />} loading={resume.isPending} onClick={() => resume.mutate({ approved: true })}>
                        批准本次审方
                      </Button>
                      <Button danger icon={<X size={16} />} loading={resume.isPending} onClick={() => resume.mutate({ approved: false })}>
                        驳回本次审方
                      </Button>
                    </Space>
                  </Space>
                </Card>
              ) : null}
              <Card title="发布与教学">
                <Space wrap>
                  <Popconfirm title="发布后患者端将能查看处方说明和饮食注意事项。" onConfirm={() => publish.mutate()}>
                    <Button icon={<FileUp size={16} />} loading={publish.isPending} disabled={data.status !== "doctor_confirmed"}>
                      发布给患者
                    </Button>
                  </Popconfirm>
                  <Button icon={<RotateCcw size={16} />} loading={deidentify.isPending} disabled={!["doctor_confirmed", "published_to_patient"].includes(data.status)} onClick={() => deidentify.mutate()}>
                    生成学习案例
                  </Button>
                </Space>
              </Card>
            </>
          ) : (
            <Card>
              <Typography.Text type="secondary">处方已创建，运行审方后将展示风险、证据包和人工确认步骤。</Typography.Text>
            </Card>
          )}
        </Space>
      </div>
      <Modal
        title="补充处方级用法"
        open={usageModalOpen}
        confirmLoading={updateUsage.isPending}
        okText="保存并重新审方"
        cancelText="取消"
        onCancel={() => setUsageModalOpen(false)}
        onOk={() => usageForm.validateFields().then((values) => updateUsage.mutate(values))}
      >
        <Alert
          type="info"
          showIcon
          message="单味药剂量已经保留"
          description="这里仅补充整张处方的煎服方式、每日频次和分服方法，不会修改下方药材剂量。"
          style={{ marginBottom: 16 }}
        />
        <Form form={usageForm} layout="vertical">
          <Form.Item
            name="usage_method"
            label="处方用法"
            rules={[{ required: true, whitespace: true, message: "请填写处方级用法说明" }]}
          >
            <Input.TextArea rows={4} placeholder="例如：每日一剂，水煎取汁，早晚分服。具体内容由医生确认。" />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}
