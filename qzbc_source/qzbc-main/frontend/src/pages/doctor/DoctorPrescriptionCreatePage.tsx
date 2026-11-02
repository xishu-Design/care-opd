import { useMutation, useQuery } from "@tanstack/react-query";
import { Alert, Button, Card, Form, Input, Select, Space, Tag, Typography } from "antd";
import { Lightbulb, MinusCircle, Plus } from "lucide-react";
import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { doctorApi, type PrescriptionCreatePayload } from "../../api/doctorApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { errorMessage } from "../../utils/errors";

export function DoctorPrescriptionCreatePage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [form] = Form.useForm();
  const [submitReview, setSubmitReview] = useState(false);
  const [suggestionError, setSuggestionError] = useState("");
  const patientId = Number(params.get("patientId") || 0);
  const patients = useQuery({ queryKey: ["doctor", "patients"], queryFn: doctorApi.listPatients });
  const create = useMutation({
    mutationFn: (payload: PrescriptionCreatePayload & { submitReview?: boolean }) => {
      const { submitReview: _submitReview, ...request } = payload;
      return doctorApi.createPrescription(request);
    },
    onSuccess: (data, variables) => {
      navigate(`/doctor/prescriptions/${data.id}/review${variables.submitReview ? "?autoReview=1" : ""}`);
    },
  });
  const suggestions = useMutation({
    mutationFn: () => {
      const selectedPatient = Number(form.getFieldValue("patient_id"));
      if (!selectedPatient) throw new Error("请先选择患者，再生成候选拟方");
      const symptoms = String(form.getFieldValue("symptom_context") || "").trim();
      if (!symptoms) throw new Error("请先填写患者症状与诊疗背景，再生成候选拟方");
      return doctorApi.getPrescriptionSuggestions({ patient_id: selectedPatient, symptoms, query: String(form.getFieldValue("doctor_notes") || "") });
    },
    onError: (error) => setSuggestionError(errorMessage(error)),
    onSuccess: () => setSuggestionError(""),
  });

  const applySuggestion = (items: Array<{ herb_name: string }>) => {
    const existing = form.getFieldValue("items") || [];
    const merged = [...existing, ...items.map((item) => ({ herb_name: item.herb_name, unit: "g" }))];
    const unique = merged.filter((item, index, rows) => item.herb_name && rows.findIndex((row) => row.herb_name === item.herb_name) === index);
    form.setFieldsValue({ title: "知识库辅助拟方草稿", items: unique });
  };

  if (patients.isLoading) return <LoadingState />;
  if (patients.isError) return <ErrorState error={patients.error} />;
  const initialPatient = patients.data?.find((item) => item.id === patientId);

  return (
    <>
      <PageHeader title="新建拟方" />
      <Card>
        {create.isError ? <Alert type="error" showIcon message={errorMessage(create.error)} /> : null}
        <Form
          form={form}
          layout="vertical"
          initialValues={{ patient_id: patientId || undefined, symptom_context: initialPatient?.current_symptoms || "", title: "辅助拟方草稿", items: [] }}
          onFinish={(values) => create.mutate({ ...values, submitReview })}
        >
          <Form.Item
            name="patient_id"
            label="患者"
            rules={[
              { required: true },
              { validator: (_, value) => {
                const selected = patients.data?.find((item) => item.id === value);
                return selected && selected.age !== null && selected.age !== undefined && selected.sex
                  ? Promise.resolve()
                  : Promise.reject(new Error("患者档案缺少年龄或性别，请先到患者详情补全"));
              } },
            ]}
          >
            <Select
              onChange={(value) => {
                const selected = patients.data?.find((item) => item.id === value);
                form.setFieldValue("symptom_context", selected?.current_symptoms || "");
              }}
              options={patients.data?.map((item) => ({
                value: item.id,
                label: `${item.display_name} · ${{ female: "女", male: "男" }[String(item.sex)] || "性别未填"} · ${item.age ?? "年龄未填"}岁`,
              }))}
            />
          </Form.Item>
          <Form.Item name="title" label="处方标题" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="symptom_context" label="患者症状与诊疗背景" rules={[{ required: true, whitespace: true, message: "请填写患者当前症状或医生确认的诊疗背景" }]}>
            <Input.TextArea rows={3} placeholder="例如：患者已记录的主诉、症状变化、体质背景。仅供医生生成候选并人工复核。" />
          </Form.Item>
          <Alert type="info" showIcon message="本次填写内容会随处方保存；医生确认处方后，将自动写入患者诊疗历史并用于脱敏教学案例。" style={{ marginBottom: 16 }} />
          <Form.Item
            name="usage_method"
            label="处方用法"
            extra="这里填写整张处方的煎服方式、每日频次和分服方法；下方表格填写每味药的剂量和单位。"
          >
            <Input.TextArea rows={2} placeholder="例如：每日一剂，水煎取汁，早晚分服。具体内容由医生确认。" />
          </Form.Item>
          <Form.Item name="doctor_notes" label="医生备注"><Input.TextArea rows={2} /></Form.Item>
          <div className="suggestion-panel">
            <Space direction="vertical" size={10} style={{ width: "100%" }}>
              <Space wrap>
                <Typography.Text strong>知识库候选拟方</Typography.Text>
                <Tag color="gold">候选仅含药名，剂量和处方用法需医生填写</Tag>
                <Button icon={<Lightbulb size={16} />} loading={suggestions.isPending} onClick={() => suggestions.mutate()}>生成候选</Button>
              </Space>
              {suggestionError ? <Alert type="error" showIcon message={suggestionError} /> : null}
              {suggestions.data ? <>
                <Alert type="warning" showIcon message={suggestions.data.disclaimer} />
                {suggestions.data.excluded_allergy_herbs.length || suggestions.data.excluded_formula_suggestions.length ? (
                  <Alert
                    type="error"
                    showIcon
                    message={`已依据过敏史排除：${[
                      ...suggestions.data.excluded_allergy_herbs,
                      ...suggestions.data.excluded_formula_suggestions.map((item) => `${item.title}（含${item.matched_allergens.join("、")}）`),
                    ].join("、")}`}
                    description={`当前记录的过敏史：${suggestions.data.allergy_history || "未记录"}`}
                  />
                ) : null}
                <Typography.Text type="secondary">本次检索症状/背景：{suggestions.data.symptoms || "未提供"}</Typography.Text>
                <Typography.Text type="secondary">
                  {suggestions.data.llm_rerank.used
                    ? `已使用 Ollama（${suggestions.data.llm_rerank.model}）对规范方剂候选排序。`
                    : suggestions.data.herb_llm_rerank.used
                      ? `已使用 Ollama（${suggestions.data.herb_llm_rerank.model}）对规范药材候选排序。`
                      : suggestions.data.llm_rerank.fallback_used || suggestions.data.herb_llm_rerank.fallback_used
                      ? "Ollama 排序暂不可用，当前按可追溯的规则匹配排序。"
                      : "当前按可追溯的规则匹配排序。"}
                </Typography.Text>
                {suggestions.data.no_match_message ? <Alert type="info" showIcon message={suggestions.data.no_match_message} /> : null}
                {suggestions.data.suggestions.map((item) => (
                  <div key={item.herb_name} className="suggestion-row">
                    <div><Typography.Text strong>{item.herb_name}</Typography.Text><Typography.Text type="secondary"> 功效：{item.functions.join("、") || "未提取"}；症状/证候：{item.symptoms.join("、") || "未提取"}；命中依据：{item.match_reasons.join("；") || "目录字段匹配"}</Typography.Text></div>
                    <Button size="small" onClick={() => applySuggestion([item])}>加入草稿</Button>
                  </div>
                ))}
                {suggestions.data.formula_suggestions.map((item) => <div key={item.title} className="suggestion-row"><div><Typography.Text strong>{item.title}</Typography.Text><Typography.Text type="secondary"> 功效：{item.functions.join("、")}；证候：{item.syndromes.join("、") || "未关联"}；症状/疾病：{item.symptoms.join("、") || "未关联"}；组成：{item.herbs.join("、")}</Typography.Text>{item.review_status === "pending_doctor_clinical_review" ? <Tag color="gold">待医生临床复核</Tag> : null}</div><Button size="small" onClick={() => applySuggestion(item.herbs.map((herb_name) => ({ herb_name })))}>使用组成</Button></div>)}
              </> : null}
            </Space>
          </div>
          <Form.List name="items">
            {(fields, { add, remove }) => (
              <Space direction="vertical" style={{ width: "100%" }}>
                <Typography.Text strong>单味药剂量</Typography.Text>
                {fields.map((field) => (
                  <Space key={field.key} align="baseline" wrap>
                    <Form.Item {...field} name={[field.name, "herb_name"]} rules={[{ required: true }]}><Input placeholder="药材" /></Form.Item>
                    <Form.Item {...field} name={[field.name, "dose"]}><Input placeholder="剂量" /></Form.Item>
                    <Form.Item {...field} name={[field.name, "unit"]}><Input placeholder="单位" /></Form.Item>
                    <Form.Item {...field} name={[field.name, "notes"]}><Input placeholder="备注" /></Form.Item>
                    <Button icon={<MinusCircle size={16} />} onClick={() => remove(field.name)} />
                  </Space>
                ))}
                <Button icon={<Plus size={16} />} onClick={() => add({ unit: "g" })}>添加药材</Button>
              </Space>
            )}
          </Form.List>
          <Space style={{ marginTop: 20 }}>
            <Button htmlType="submit" loading={create.isPending} onClick={() => setSubmitReview(false)}>保存草稿</Button>
            <Button
              type="primary"
              htmlType="submit"
              loading={create.isPending}
              onClick={() => setSubmitReview(true)}
            >
              保存后进入审方
            </Button>
          </Space>
        </Form>
      </Card>
    </>
  );
}
