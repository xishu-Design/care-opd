import { useQuery } from "@tanstack/react-query";
import { Button, Card, Statistic } from "antd";
import { useNavigate } from "react-router-dom";

import { doctorApi } from "../../api/doctorApi";
import { PageHeader } from "../../components/common/PageHeader";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";

export function DoctorDashboardPage() {
  const navigate = useNavigate();
  const patients = useQuery({ queryKey: ["doctor", "patients"], queryFn: doctorApi.listPatients });
  if (patients.isLoading) return <LoadingState />;
  if (patients.isError) return <ErrorState error={patients.error} />;

  return (
    <>
      <PageHeader
        title="医生总览"
        subtitle="审方、确认和发布都以服务端状态为准。"
        extra={<Button type="primary" onClick={() => navigate("/doctor/patients")}>患者列表</Button>}
      />
      <div className="metric-grid">
        <Card><Statistic title="我的患者" value={patients.data?.length || 0} /></Card>
        <Card><Statistic title="待审方" value="-" /></Card>
        <Card><Statistic title="待确认" value="-" /></Card>
        <Card><Statistic title="学习案例" value="-" /></Card>
      </div>
    </>
  );
}
