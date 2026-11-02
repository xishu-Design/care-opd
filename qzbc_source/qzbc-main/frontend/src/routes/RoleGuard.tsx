import { Alert } from "antd";
import { Navigate, Outlet } from "react-router-dom";

import type { Role } from "../api/types";
import { useAuthStore } from "../stores/authStore";

export function RoleGuard({ role }: { role: Role }) {
  const currentRole = useAuthStore((state) => state.role);
  if (currentRole !== role) {
    return (
      <>
        <Alert type="error" message="当前账号无权访问该角色端。" />
        <Navigate to="/role" replace />
      </>
    );
  }
  return <Outlet />;
}
