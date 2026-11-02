import { Button, Card, Space, Typography } from "antd";
import { BookOpen, GraduationCap, Stethoscope } from "lucide-react";
import { useNavigate } from "react-router-dom";

import type { Role } from "../../api/types";

const roleCards: Array<{ role: Role; title: string; summary: string; icon: JSX.Element }> = [
  { role: "doctor", title: "医生端", summary: "患者、拟方、审方、确认、发布", icon: <Stethoscope size={24} /> },
  { role: "patient", title: "患者端", summary: "处方、说明、饮食注意、科普", icon: <BookOpen size={24} /> },
  { role: "student", title: "学习端", summary: "脱敏案例、审方训练、标准解析", icon: <GraduationCap size={24} /> },
];

export function RoleSelectPage() {
  const navigate = useNavigate();
  return (
    <Space direction="vertical" size={24} style={{ width: "100%" }}>
      <div>
        <Typography.Title level={2} style={{ marginBottom: 4 }}>
          GuardTCM
        </Typography.Title>
        <Typography.Text type="secondary">岐智本草三端演示入口</Typography.Text>
      </div>
      <div className="responsive-grid">
        {roleCards.map((item) => (
          <Card key={item.role}>
            <Space direction="vertical" size={16} style={{ width: "100%" }}>
              {item.icon}
              <div>
                <Typography.Title level={4} style={{ marginTop: 0 }}>
                  {item.title}
                </Typography.Title>
                <Typography.Text type="secondary">{item.summary}</Typography.Text>
              </div>
              <Space>
                <Button type="primary" onClick={() => navigate(`/login?role=${item.role}`)}>
                  登录
                </Button>
                <Button onClick={() => navigate(`/register?role=${item.role}`)}>注册</Button>
              </Space>
            </Space>
          </Card>
        ))}
      </div>
    </Space>
  );
}
