import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Alert, Button, Card, Descriptions, Popconfirm, Space, message } from "antd";
import { FileUp } from "lucide-react";
import { useParams } from "react-router-dom";

import { doctorApi } from "../../api/doctorApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { StatusTag } from "../../components/common/StatusTag";
import { PrescriptionItemsTable } from "../../components/prescription/PrescriptionItemsTable";
import { errorMessage } from "../../utils/errors";

export function DoctorPublishPage() {
  const prescriptionId = Number(useParams().prescriptionId);
  const queryClient = useQueryClient();
  const [api, holder] = message.useMessage();
  const prescription = useQuery({ queryKey: ["doctor", "prescription", prescriptionId], queryFn: () => doctorApi.getPrescriptionDetail(prescriptionId), enabled: Boolean(prescriptionId) });
  const publish = useMutation({
    mutationFn: () => doctorApi.publishPrescription(prescriptionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["doctor", "prescription", prescriptionId] });
      api.success("已发布给患者");
    },
  });
  if (prescription.isLoading) return <LoadingState />;
  if (prescription.isError) return <ErrorState error={prescription.error} />;
  const data = prescription.data!;
  return (
    <>
      {holder}
      <PageHeader title="发布处方" subtitle={data.title} />
      <Card>
        {publish.isError ? <Alert type="error" showIcon message={errorMessage(publish.error)} /> : null}
        <Descriptions column={1} size="small">
          <Descriptions.Item label="状态"><StatusTag value={data.status} /></Descriptions.Item>
          <Descriptions.Item label="患者年龄（就诊时）">{data.patient_age_at_visit ?? "未记录"}</Descriptions.Item>
          <Descriptions.Item label="患者性别（就诊时）">{{ female: "女", male: "男" }[String(data.patient_sex_at_visit)] || data.patient_sex_at_visit || "未记录"}</Descriptions.Item>
          <Descriptions.Item label="症状与诊疗背景">{data.symptom_context || "未填写"}</Descriptions.Item>
          <Descriptions.Item label="处方用法">{data.usage_method || "未填写（单味药剂量见下表）"}</Descriptions.Item>
        </Descriptions>
        <PrescriptionItemsTable items={data.items} />
        <Space style={{ marginTop: 16 }}>
          <Popconfirm title="发布后患者端将能查看该处方。" onConfirm={() => publish.mutate()}>
            <Button type="primary" icon={<FileUp size={16} />} loading={publish.isPending} disabled={data.status !== "doctor_confirmed"}>
              发布给患者
            </Button>
          </Popconfirm>
        </Space>
      </Card>
    </>
  );
}
