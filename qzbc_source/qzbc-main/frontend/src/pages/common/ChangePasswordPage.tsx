import { useMutation } from "@tanstack/react-query";
import { Alert, Button, Card, Form, Input, Space, Typography } from "antd";
import { KeyRound, LogOut } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { authApi } from "../../api/authApi";
import { useAuthStore } from "../../stores/authStore";
import { errorMessage } from "../../utils/errors";
import { dashboardPath } from "../../utils/role";

export function ChangePasswordPage() {
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const updateUser = useAuthStore((state) => state.updateUser);
  const clearAuth = useAuthStore((state) => state.clearAuth);
  const mutation = useMutation({
    mutationFn: authApi.changePassword,
    onSuccess: (updated) => {
      updateUser(updated);
      navigate(dashboardPath(updated.role), { replace: true });
    },
  });

  return (
    <div style={{ maxWidth: 520, margin: "64px auto", padding: 16 }}>
      <Card title={<Space><KeyRound size={18} />修改密码</Space>}>
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          {user?.must_change_password ? (
            <Alert type="warning" showIcon message="这是医生代建账号的临时密码，请先设置只有你本人知道的新密码。" />
          ) : null}
          {mutation.isError ? <Alert type="error" showIcon message={errorMessage(mutation.error)} /> : null}
          <Typography.Text type="secondary">账号：{user?.username}</Typography.Text>
          <Form layout="vertical" onFinish={(values) => mutation.mutate(values)}>
            <Form.Item name="current_password" label="当前密码" rules={[{ required: true, min: 6 }]}>
              <Input.Password autoComplete="current-password" />
            </Form.Item>
            <Form.Item name="new_password" label="新密码" rules={[{ required: true, min: 8, message: "新密码至少 8 位" }]}>
              <Input.Password autoComplete="new-password" />
            </Form.Item>
            <Form.Item
              name="confirm_password"
              label="再次输入新密码"
              dependencies={["new_password"]}
              rules={[
                { required: true, message: "请再次输入新密码" },
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    return !value || getFieldValue("new_password") === value
                      ? Promise.resolve()
                      : Promise.reject(new Error("两次输入的新密码不一致"));
                  },
                }),
              ]}
            >
              <Input.Password autoComplete="new-password" />
            </Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={mutation.isPending}>保存新密码</Button>
              {!user?.must_change_password ? <Button onClick={() => navigate(-1)}>取消</Button> : null}
              <Button
                type="text"
                icon={<LogOut size={15} />}
                onClick={() => { clearAuth(); navigate("/login", { replace: true }); }}
              >
                退出登录
              </Button>
            </Space>
          </Form>
        </Space>
      </Card>
    </div>
  );
}
