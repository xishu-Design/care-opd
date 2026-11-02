import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Button, Form, Input, InputNumber, Modal, Select, Space, Table } from "antd";
import { Plus } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { doctorApi, type PatientCreatePayload } from "../../api/doctorApi";
import { EmptyState } from "../../components/common/EmptyState";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { errorMessage } from "../../utils/errors";

export function DoctorPatientListPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const patients = useQuery({ queryKey: ["doctor", "patients"], queryFn: doctorApi.listPatients });
  const create = useMutation({
    mutationFn: (payload: PatientCreatePayload) => doctorApi.createPatient(payload),
    onSuccess: () => {
      setOpen(false);
      queryClient.invalidateQueries({ queryKey: ["doctor", "patients"] });
    },
  });

  if (patients.isLoading) return <LoadingState />;
  if (patients.isError) return <ErrorState error={patients.error} />;

  return (
    <>
      <PageHeader title="患者列表" extra={<Button type="primary" icon={<Plus size={16} />} onClick={() => setOpen(true)}>新建患者</Button>} />
      <Table
        rowKey="id"
        dataSource={patients.data}
        locale={{ emptyText: <EmptyState label="暂无患者" /> }}
        columns={[
          { title: "姓名", dataIndex: "display_name" },
          { title: "年龄", dataIndex: "age" },
          { title: "性别", dataIndex: "sex", render: (value) => ({ female: "女", male: "男" }[String(value)] || "-") },
          { title: "过敏史", dataIndex: "allergy_history" },
          { title: "既往史", dataIndex: "medical_history" },
          {
            title: "操作",
            render: (_, record) => (
              <Space>
                <Button onClick={() => navigate(`/doctor/patients/${record.id}`)}>详情</Button>
                <Button type="primary" onClick={() => navigate(`/doctor/prescriptions/new?patientId=${record.id}`)}>拟方</Button>
              </Space>
            ),
          },
        ]}
      />
      <Modal title="新建并绑定患者" open={open} onCancel={() => setOpen(false)} footer={null}>
        {create.isError ? <Alert type="error" showIcon message={errorMessage(create.error)} /> : null}
        <Alert
          style={{ marginBottom: 16 }}
          type="info"
          showIcon
          message="已自主注册的患者只需填写用户名，临时密码可留空；未注册患者需设置一次性临时密码，首次登录会强制患者修改。"
        />
        <Form layout="vertical" onFinish={(values) => create.mutate(values)}>
          <Form.Item name="username" label="用户名" rules={[{ required: true, min: 3 }]}><Input /></Form.Item>
          <Form.Item name="password" label="一次性临时密码（已有账号可留空）" rules={[{ min: 6 }]}><Input.Password autoComplete="new-password" /></Form.Item>
          <Form.Item name="display_name" label="姓名" rules={[{ required: true }]}><Input /></Form.Item>
          <Space.Compact style={{ width: "100%" }}>
            <Form.Item name="age" label="年龄" rules={[{ required: true, message: "请填写患者年龄" }]} style={{ width: "50%" }}><InputNumber min={0} max={130} style={{ width: "100%" }} /></Form.Item>
            <Form.Item name="sex" label="性别" rules={[{ required: true, message: "请选择患者性别" }]} style={{ width: "50%" }}><Select options={[{ value: "female", label: "女" }, { value: "male", label: "男" }]} /></Form.Item>
          </Space.Compact>
          <Form.Item name="allergy_history" label="过敏史"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="medical_history" label="既往史"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="body_state" label="身体状态"><Input.TextArea rows={2} /></Form.Item>
          <Form.Item name="current_symptoms" label="当前症状"><Input.TextArea rows={2} /></Form.Item>
          <Button type="primary" htmlType="submit" loading={create.isPending}>保存</Button>
        </Form>
      </Modal>
    </>
  );
}
