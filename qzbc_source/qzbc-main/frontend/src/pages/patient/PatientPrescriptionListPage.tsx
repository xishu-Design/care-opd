import { useQuery } from "@tanstack/react-query";
import { Button, Table } from "antd";
import { useNavigate } from "react-router-dom";

import { patientApi } from "../../api/patientApi";
import { EmptyState } from "../../components/common/EmptyState";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { StatusTag } from "../../components/common/StatusTag";

export function PatientPrescriptionListPage() {
  const navigate = useNavigate();
  const prescriptions = useQuery({ queryKey: ["patient", "prescriptions"], queryFn: patientApi.listPrescriptions });
  if (prescriptions.isLoading) return <LoadingState />;
  if (prescriptions.isError) return <ErrorState error={prescriptions.error} />;
  return (
    <>
      <PageHeader title="我的处方" subtitle="仅显示医生已发布给患者的处方。" />
      <Table
        rowKey="id"
        dataSource={prescriptions.data}
        locale={{ emptyText: <EmptyState label="暂无已发布处方" /> }}
        columns={[
          { title: "标题", dataIndex: "title" },
          { title: "状态", dataIndex: "status", render: (value) => <StatusTag value={value} /> },
          { title: "药味数", dataIndex: "items", render: (items) => items?.length || 0 },
          { title: "操作", render: (_, record) => <Button onClick={() => navigate(`/patient/prescriptions/${record.id}`)}>查看</Button> },
        ]}
      />
    </>
  );
}
