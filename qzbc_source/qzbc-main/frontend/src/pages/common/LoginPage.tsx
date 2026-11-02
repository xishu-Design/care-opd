import { useMutation } from "@tanstack/react-query";
import { Alert, Button, Card, Form, Input, Select, Space, Typography } from "antd";
import { LogIn } from "lucide-react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { authApi } from "../../api/authApi";
import type { Role } from "../../api/types";
import { useAuthStore } from "../../stores/authStore";
import { errorMessage } from "../../utils/errors";
import { dashboardPath, isRole, roleLabel } from "../../utils/role";

export function LoginPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const setAuth = useAuthStore((state) => state.setAuth);
  const roleParam = params.get("role");
  const initialRole = isRole(roleParam) ? roleParam : undefined;
  const mutation = useMutation({
    mutationFn: (payload: { role: Role; username: string; password: string }) => authApi.login({
      username: payload.username,
      password: payload.password,
      entry_role: payload.role,
    }),
    onSuccess: (data) => {
      if (data.user.role !== mutation.variables?.role) return;
      setAuth(data.access_token, data.user);
      navigate(data.user.must_change_password ? "/account/password" : dashboardPath(data.user.role), { replace: true });
    },
  });

  return (
    <Card className="login-card" bordered={false}>
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        <div className="login-heading">
          <span className="login-seal">本草</span>
          <div>
            <Typography.Title level={3}>GuardTCM</Typography.Title>
            <Typography.Text type="secondary">岐智本草 · 安全辅助平台</Typography.Text>
          </div>
        </div>
        {mutation.isError ? <Alert type="error" showIcon message={errorMessage(mutation.error)} /> : null}
        {mutation.isSuccess && mutation.data.user.role !== mutation.variables?.role ? (
          <Alert type="error" showIcon message="账号角色与所选角色不一致，请重新选择。" />
        ) : null}
        <Form layout="vertical" initialValues={{ role: initialRole }} onFinish={(values) => mutation.mutate(values)}>
          <Form.Item name="role" label="选择角色" rules={[{ required: true, message: "请选择角色" }]}>
            <Select
              placeholder="请选择登录角色"
              options={(["doctor", "patient", "student"] as Role[]).map((value) => ({ value, label: roleLabel(value) }))}
            />
          </Form.Item>
          <Form.Item name="username" label="用户名" rules={[{ required: true }]}>
            <Input autoComplete="username" />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true }]}>
            <Input.Password autoComplete="current-password" />
          </Form.Item>
          <Space>
            <Button type="primary" htmlType="submit" loading={mutation.isPending} icon={<LogIn size={16} />}>
              登录
            </Button>
            <Link to="/register">没有账号？自主注册</Link>
          </Space>
        </Form>
        <Typography.Text type="secondary">医生代建的患者账号首次登录后必须修改临时密码。</Typography.Text>
      </Space>
    </Card>
  );
}
