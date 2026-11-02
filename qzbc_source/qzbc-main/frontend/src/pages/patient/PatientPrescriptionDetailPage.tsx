import { useQuery } from "@tanstack/react-query";
import { Card, Descriptions, Space } from "antd";
import { useParams } from "react-router-dom";

import { patientApi } from "../../api/patientApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { StatusTag } from "../../components/common/StatusTag";
import { DietPrecautionCard } from "../../components/prescription/DietPrecautionCard";
import { PatientInstructionCard } from "../../components/prescription/PatientInstructionCard";
import { PrescriptionItemsTable } from "../../components/prescription/PrescriptionItemsTable";

export function PatientPrescriptionDetailPage() {
  const prescriptionId = Number(useParams().prescriptionId);
  const prescription = useQuery({
    queryKey: ["patient", "prescription", prescriptionId],
    queryFn: () => patientApi.getPrescriptionDetail(prescriptionId),
    enabled: Boolean(prescriptionId),
  });
  if (prescription.isLoading) return <LoadingState />;
  if (prescription.isError) return <ErrorState error={prescription.error} />;
  const data = prescription.data!;
  return (
    <>
      <PageHeader title={data.title} />
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        <Card title="处方">
          <Descriptions size="small" column={1}>
            <Descriptions.Item label="状态"><StatusTag value={data.status} /></Descriptions.Item>
            <Descriptions.Item label="处方用法">{data.usage_method || "医生尚未填写处方级用法"}</Descriptions.Item>
          </Descriptions>
          <PrescriptionItemsTable items={data.items} />
        </Card>
        <PatientInstructionCard
          content={data.patient_instruction?.content}
          statement={data.patient_instruction?.safety_statement}
          doctorNote={data.patient_instruction?.doctor_note}
          doctorUsage={data.patient_instruction?.doctor_usage_method}
          itemNotes={data.patient_instruction?.item_notes}
        />
        <DietPrecautionCard data={data.diet_precaution} />
      </Space>
    </>
  );
}
