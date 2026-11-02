import { useQuery } from "@tanstack/react-query";
import { Button, Card, Statistic } from "antd";
import { useNavigate } from "react-router-dom";

import { patientApi } from "../../api/patientApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";

export function PatientDashboardPage() {
  const navigate = useNavigate();
  const profile = useQuery({ queryKey: ["patient", "profile"], queryFn: patientApi.getProfile });
  const prescriptions = useQuery({ queryKey: ["patient", "prescriptions"], queryFn: patientApi.listPrescriptions });
  if (profile.isLoading || prescriptions.isLoading) return <LoadingState />;
  if (profile.isError) return <ErrorState error={profile.error} />;
  if (prescriptions.isError) return <ErrorState error={prescriptions.error} />;
  return (
    <>
      <PageHeader title={`你好，${profile.data?.display_name || profile.data?.username}`} extra={<Button type="primary" onClick={() => navigate("/patient/education")}>科普问答</Button>} />
      <div className="metric-grid">
        <Card><Statistic title="已发布处方" value={prescriptions.data?.length || 0} /></Card>
        <Card><Statistic title="过敏史" value={profile.data?.allergy_history ? "已记录" : "未记录"} /></Card>
        <Card><Statistic title="既往史" value={profile.data?.medical_history ? "已记录" : "未记录"} /></Card>
      </div>
    </>
  );
}
