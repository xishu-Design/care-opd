import { useMutation } from "@tanstack/react-query";
import { Alert, Button, Card, Form, Input, Select, Space, Typography } from "antd";
import { UserPlus } from "lucide-react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { authApi } from "../../api/authApi";
import type { Role } from "../../api/types";
import { errorMessage } from "../../utils/errors";
import { isRole, roleLabel } from "../../utils/role";

export function RegisterPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const roleParam = params.get("role");
  const initialRole: Role = isRole(roleParam) ? roleParam : "patient";
  const mutation = useMutation({
    mutationFn: (payload: { role: Role; username: string; password: string; display_name: string }) => authApi.register(payload),
    onSuccess: (_data, variables) => navigate(`/login?role=${variables.role}`),
  });

  return (
    <Card title="账号注册">
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        {mutation.isError ? <Alert type="error" showIcon message={errorMessage(mutation.error)} /> : null}
        <Form layout="vertical" onFinish={(values) => mutation.mutate(values)} initialValues={{ role: initialRole }}>
          <Form.Item name="role" label="选择角色" rules={[{ required: true, message: "请选择角色" }]}>
            <Select options={(['doctor', 'patient', 'student'] as Role[]).map((value) => ({ value, label: roleLabel(value) }))} />
          </Form.Item>
          <Form.Item name="username" label="用户名" rules={[{ required: true, min: 3 }]}>
            <Input />
          </Form.Item>
          <Form.Item name="display_name" label="显示名" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true, min: 6 }]}>
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <Form.Item
            name="confirm_password"
            label="确认密码"
            dependencies={["password"]}
            rules={[
              { required: true, message: "请再次输入密码" },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  return !value || getFieldValue("password") === value
                    ? Promise.resolve()
                    : Promise.reject(new Error("两次输入的密码不一致"));
                },
              }),
            ]}
          >
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <div style={{ border: "1px solid #e1e7e4", borderRadius: 6, padding: 12, background: "#f8faf9" }}>
            <Typography.Text type="secondary">机构、过敏史、学习方向等扩展字段由后续资料页补充。</Typography.Text>
          </div>
          <Space style={{ marginTop: 16 }}>
            <Button type="primary" htmlType="submit" loading={mutation.isPending} icon={<UserPlus size={16} />}>
              注册
            </Button>
            <Link to="/login">已有账号，返回登录</Link>
          </Space>
        </Form>
      </Space>
    </Card>
  );
}
