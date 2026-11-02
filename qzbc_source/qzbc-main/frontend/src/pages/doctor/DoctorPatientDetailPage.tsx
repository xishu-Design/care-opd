import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Button, Card, Descriptions, Form, Input, InputNumber, Modal, Select, Space } from "antd";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { doctorApi } from "../../api/doctorApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { errorMessage } from "../../utils/errors";

export function DoctorPatientDetailPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [form] = Form.useForm();
  const patientId = Number(useParams().patientId);
  const patient = useQuery({ queryKey: ["doctor", "patient", patientId], queryFn: () => doctorApi.getPatientDetail(patientId), enabled: Boolean(patientId) });
  const update = useMutation({
    mutationFn: (values: Record<string, unknown>) => doctorApi.updatePatient(patientId, values),
    onSuccess: (data) => {
      queryClient.setQueryData(["doctor", "patient", patientId], data);
      queryClient.invalidateQueries({ queryKey: ["doctor", "patients"] });
      setEditing(false);
    },
  });
  if (patient.isLoading) return <LoadingState />;
  if (patient.isError) return <ErrorState error={patient.error} />;
  const data = patient.data!;
  return (
    <>
      <PageHeader title={data.display_name} extra={<Space><Button onClick={() => { form.setFieldsValue(data); setEditing(true); }}>编辑资料</Button><Button type="primary" onClick={() => navigate(`/doctor/prescriptions/new?patientId=${data.id}`)}>新建拟方</Button></Space>} />
      <Card>
        <Descriptions bordered column={2}>
          <Descriptions.Item label="用户名">{data.username}</Descriptions.Item>
          <Descriptions.Item label="年龄">{data.age || "-"}</Descriptions.Item>
          <Descriptions.Item label="性别">{{ female: "女", male: "男" }[String(data.sex)] || data.sex || "-"}</Descriptions.Item>
          <Descriptions.Item label="身高">{data.height_cm || "-"}</Descriptions.Item>
          <Descriptions.Item label="体重">{data.weight_kg || "-"}</Descriptions.Item>
          <Descriptions.Item label="身体状态">{data.body_state || "-"}</Descriptions.Item>
          <Descriptions.Item label="过敏史" span={2}>{data.allergy_history || "-"}</Descriptions.Item>
          <Descriptions.Item label="既往史" span={2}>{data.medical_history || "-"}</Descriptions.Item>
          <Descriptions.Item label="当前症状" span={2}>{data.current_symptoms || "-"}</Descriptions.Item>
        </Descriptions>
      </Card>
      <Modal title="编辑患者资料" open={editing} onCancel={() => setEditing(false)} footer={null} destroyOnClose>
        {update.isError ? <Alert type="error" showIcon message={errorMessage(update.error)} style={{ marginBottom: 12 }} /> : null}
        <Form form={form} layout="vertical" onFinish={(values) => update.mutate(values)}>
          <Form.Item name="display_name" label="姓名" rules={[{ required: true }]}><Input /></Form.Item>
          <Space.Compact style={{ width: "100%" }}>
            <Form.Item name="age" label="年龄" rules={[{ required: true, message: "请填写患者年龄" }]} style={{ width: "50%" }}><InputNumber min={0} max={130} style={{ width: "100%" }} /></Form.Item>
            <Form.Item name="sex" label="性别" rules={[{ required: true, message: "请选择患者性别" }]} style={{ width: "50%" }}><Select options={[{ value: "female", label: "女" }, { value: "male", label: "男" }]} /></Form.Item>
          </Space.Compact>
          <Space.Compact style={{ width: "100%" }}>
            <Form.Item name="height_cm" label="身高（cm）" style={{ width: "50%" }}><InputNumber min={0} style={{ width: "100%" }} /></Form.Item>
            <Form.Item name="weight_kg" label="体重（kg）" style={{ width: "50%" }}><InputNumber min={0} style={{ width: "100%" }} /></Form.Item>
          </Space.Compact>
          <Form.Item name="allergy_history" label="过敏史" extra="例如：对丹参过敏。该字段会用于候选过滤和结构化审方。"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="medical_history" label="既往史"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="body_state" label="身体状态"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="current_symptoms" label="当前症状"><Input.TextArea rows={2} /></Form.Item>
          <Button type="primary" htmlType="submit" loading={update.isPending}>保存修改</Button>
        </Form>
      </Modal>
    </>
  );
}
