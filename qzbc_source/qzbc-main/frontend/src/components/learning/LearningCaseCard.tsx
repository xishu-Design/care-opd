import { Button, Card, Space, Typography } from "antd";
import { ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";

import type { LearningCase } from "../../api/types";
import { asText } from "../../utils/format";

export function LearningCaseCard({ item }: { item: LearningCase }) {
  const navigate = useNavigate();
  return (
    <Card>
      <Space direction="vertical" size={12} style={{ width: "100%" }}>
        <Typography.Title level={5} style={{ margin: 0 }}>
          {item.title}
        </Typography.Title>
        <Typography.Text type="secondary" className="pre-wrap">
          {asText(item.anonymized_case)}
        </Typography.Text>
        <Button type="primary" icon={<ArrowRight size={16} />} onClick={() => navigate(`/student/cases/${item.id}/training`)}>
          开始训练
        </Button>
      </Space>
    </Card>
  );
}
