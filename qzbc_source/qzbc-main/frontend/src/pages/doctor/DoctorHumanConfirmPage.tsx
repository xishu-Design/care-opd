import { useQuery } from "@tanstack/react-query";
import { Button, Card, List, Space, Typography } from "antd";
import { useNavigate } from "react-router-dom";

import { doctorApi } from "../../api/doctorApi";
import { EmptyState } from "../../components/common/EmptyState";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { StatusTag } from "../../components/common/StatusTag";
import { shortId } from "../../utils/format";

export function DoctorHumanConfirmPage() {
  const navigate = useNavigate();
  const pending = useQuery({ queryKey: ["doctor", "pending-actions"], queryFn: doctorApi.listPendingActions });
  if (pending.isLoading) return <LoadingState />;
  if (pending.isError) return <ErrorState error={pending.error} />;
  const items = pending.data?.items || [];
  return (
    <>
      <PageHeader title="待确认审方" />
      <Card>
        <List
          dataSource={items}
          locale={{ emptyText: <EmptyState label="暂无待确认审方" /> }}
          renderItem={(item) => (
            <List.Item
              actions={[
                <Button key="review" type="primary" onClick={() => navigate(`/doctor/prescriptions/${item.prescription_id}/review`)}>
                  打开处方
                </Button>,
              ]}
            >
              <Space direction="vertical">
                <Space wrap>
                  <Typography.Text strong>处方 {item.prescription_id}</Typography.Text>
                  <StatusTag value={item.status} />
                  <StatusTag value={item.risk_level || ""} kind="risk" />
                </Space>
                <Typography.Text type="secondary">trace_id: {shortId(item.trace_id)}</Typography.Text>
              </Space>
            </List.Item>
          )}
        />
      </Card>
    </>
  );
}
