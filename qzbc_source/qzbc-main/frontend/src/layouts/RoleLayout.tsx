import { Breadcrumb, Button, Drawer, Layout, Menu, Space, Tag, Typography, type MenuProps } from "antd";
import { Activity, BarChart3, BookOpen, Database, FileSearch, FileText, FlaskConical, FolderKanban, GraduationCap, KeyRound, LayoutDashboard, LogOut, Menu as MenuIcon, MessageCircle, MonitorCheck, Network, Pill, ShieldCheck, Sparkles, TestTube2, Users } from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";

import type { Role } from "../api/types";
import { useAuthStore } from "../stores/authStore";
import { roleLabel } from "../utils/role";

interface MenuItem {
  key: string;
  label: string;
  icon: ReactNode;
}

interface MenuGroup {
  label: string;
  items: MenuItem[];
}

const menuByRole: Record<Role, MenuGroup[]> = {
  doctor: [
    { label: "临床工作", items: [
      { key: "/doctor/dashboard", label: "工作总览", icon: <LayoutDashboard size={16} /> },
      { key: "/doctor/ai-chat", label: "专业 AI 问答", icon: <MessageCircle size={16} /> },
      { key: "/doctor/patients", label: "患者管理", icon: <Users size={16} /> },
      { key: "/doctor/prescriptions/new", label: "辅助拟方", icon: <Pill size={16} /> },
    ] },
    { label: "知识资源", items: [
      { key: "/doctor/knowledge-graph", label: "知识图谱", icon: <Network size={16} /> },
      { key: "/doctor/knowledge/herbs", label: "药材目录", icon: <BookOpen size={16} /> },
      { key: "/doctor/knowledge/formulas", label: "方剂关系", icon: <FolderKanban size={16} /> },
      { key: "/doctor/knowledge/sources", label: "来源与许可", icon: <Database size={16} /> },
    ] },
    { label: "系统验收", items: [
      { key: "/harness/demo", label: "演示控制台", icon: <MonitorCheck size={16} /> },
      { key: "/harness/evaluation", label: "系统评测", icon: <TestTube2 size={16} /> },
      { key: "/harness/ablation", label: "消融实验", icon: <FlaskConical size={16} /> },
      { key: "/harness/sources", label: "来源登记", icon: <Database size={16} /> },
      { key: "/harness/data-pipeline", label: "数据流水线", icon: <FileSearch size={16} /> },
      { key: "/harness/content-stats", label: "内容统计", icon: <BarChart3 size={16} /> },
      { key: "/harness/content-quality", label: "内容质量", icon: <ShieldCheck size={16} /> },
      { key: "/harness/lightrag", label: "LightRAG", icon: <Sparkles size={16} /> },
      { key: "/harness/traces", label: "链路追踪", icon: <Activity size={16} /> },
    ] },
  ],
  patient: [
    { label: "个人服务", items: [
      { key: "/patient/dashboard", label: "服务总览", icon: <LayoutDashboard size={16} /> },
      { key: "/patient/profile", label: "健康档案", icon: <Users size={16} /> },
      { key: "/patient/prescriptions", label: "处方查看", icon: <FileText size={16} /> },
      { key: "/patient/education", label: "安全科普", icon: <MessageCircle size={16} /> },
    ] },
  ],
  student: [
    { label: "学习中心", items: [
      { key: "/student/dashboard", label: "学习总览", icon: <LayoutDashboard size={16} /> },
      { key: "/student/ai-chat", label: "教学 AI 问答", icon: <MessageCircle size={16} /> },
      { key: "/student/cases", label: "脱敏案例", icon: <BookOpen size={16} /> },
      { key: "/student/syndrome-cases", label: "辨证案例", icon: <FlaskConical size={16} /> },
      { key: "/student/quiz", label: "训练测验", icon: <GraduationCap size={16} /> },
      { key: "/student/progress", label: "学习进度", icon: <BarChart3 size={16} /> },
    ] },
    { label: "知识资源", items: [
      { key: "/student/knowledge-graph", label: "学习知识图谱", icon: <Network size={16} /> },
    ] },
  ],
};

export function RoleLayout({ role }: { role: Role }) {
  const navigate = useNavigate();
  const location = useLocation();
  const user = useAuthStore((state) => state.user);
  const clearAuth = useAuthStore((state) => state.clearAuth);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const groups = menuByRole[role];
  const items = groups.flatMap((group) => group.items);
  const selectedItem = [...items].sort((left, right) => right.key.length - left.key.length).find((item) => location.pathname.startsWith(item.key));
  const selected = selectedItem?.key || items[0].key;
  const selectedGroup = groups.find((group) => group.items.some((item) => item.key === selected))?.label || "工作台";
  const menuItems = useMemo<MenuProps["items"]>(() => groups.map((group) => ({
    type: "group",
    label: group.label,
    children: group.items.map((item) => ({ ...item, onClick: () => { setMobileMenuOpen(false); navigate(item.key); } })),
  })), [groups, navigate]);

  const navigation = () => <Menu className="platform-menu" mode="inline" selectedKeys={[selected]} items={menuItems} />;

  return (
    <Layout className={`app-layout platform-layout ${role === "patient" ? "patient-platform" : ""}`}>
      <Layout.Sider className="platform-sider" width={236} breakpoint="lg" collapsedWidth={0}>
        <div className="platform-brand">
          <div className="brand">
            <span className="brand-mark">岐</span>
            <span>GuardTCM</span>
          </div>
          <Typography.Text>中医诊疗平台</Typography.Text>
        </div>
        {navigation()}
      </Layout.Sider>
      <Layout>
        <Layout.Header className="app-header">
          <Space className="platform-header-title" size={12}>
            <Button className="mobile-nav-trigger" type="text" aria-label="打开导航" icon={<MenuIcon size={19} />} onClick={() => setMobileMenuOpen(true)} />
            <Typography.Text className="platform-title">GuardTCM 中医诊疗平台</Typography.Text>
          </Space>
          <Space>
            <Tag color="gold">{roleLabel(role)}</Tag>
            <Typography.Text className="platform-user" strong>{user?.username}</Typography.Text>
            <Button type="text" icon={<KeyRound size={15} />} onClick={() => navigate("/account/password")}>修改密码</Button>
          </Space>
          <Button
            className="platform-logout"
            type="text"
            icon={<LogOut size={16} />}
            onClick={() => {
              clearAuth();
              navigate("/role", { replace: true });
            }}
          >
            退出
          </Button>
        </Layout.Header>
        <Layout.Content className="page">
          <div className="page-content">
            <Breadcrumb className="platform-breadcrumb" items={[{ title: "诊疗工作台" }, { title: selectedGroup }, { title: selectedItem?.label || "详情" }]} />
            <Outlet />
          </div>
        </Layout.Content>
      </Layout>
      <Drawer className="mobile-navigation-drawer" placement="left" open={mobileMenuOpen} onClose={() => setMobileMenuOpen(false)} width={264} title="GuardTCM 中医诊疗平台">
        {navigation()}
      </Drawer>
    </Layout>
  );
}
