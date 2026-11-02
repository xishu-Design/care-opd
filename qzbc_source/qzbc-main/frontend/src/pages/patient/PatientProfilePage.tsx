import { useQuery } from "@tanstack/react-query";
import { Card, Descriptions } from "antd";

import { patientApi } from "../../api/patientApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";

export function PatientProfilePage() {
  const profile = useQuery({ queryKey: ["patient", "profile"], queryFn: patientApi.getProfile });
  if (profile.isLoading) return <LoadingState />;
  if (profile.isError) return <ErrorState error={profile.error} />;
  const data = profile.data!;
  return (
    <>
      <PageHeader title="我的档案" />
      <Card>
        <Descriptions bordered column={2}>
          <Descriptions.Item label="姓名">{data.display_name}</Descriptions.Item>
          <Descriptions.Item label="用户名">{data.username}</Descriptions.Item>
          <Descriptions.Item label="年龄">{data.age || "-"}</Descriptions.Item>
          <Descriptions.Item label="性别">{{ female: "女", male: "男" }[String(data.sex)] || data.sex || "-"}</Descriptions.Item>
          <Descriptions.Item label="身高">{data.height_cm || "-"}</Descriptions.Item>
          <Descriptions.Item label="体重">{data.weight_kg || "-"}</Descriptions.Item>
          <Descriptions.Item label="基础身体状态" span={2}>{data.body_state || "-"}</Descriptions.Item>
          <Descriptions.Item label="过敏史" span={2}>{data.allergy_history || "-"}</Descriptions.Item>
          <Descriptions.Item label="既往史" span={2}>{data.medical_history || "-"}</Descriptions.Item>
          <Descriptions.Item label="当前症状" span={2}>{data.current_symptoms || "-"}</Descriptions.Item>
        </Descriptions>
      </Card>
    </>
  );
}
