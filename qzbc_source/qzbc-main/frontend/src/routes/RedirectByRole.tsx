import { Navigate } from "react-router-dom";

import { useAuthStore } from "../stores/authStore";
import { dashboardPath } from "../utils/role";

export function RedirectByRole() {
  const role = useAuthStore((state) => state.role);
  if (!role) return <Navigate to="/role" replace />;
  return <Navigate to={dashboardPath(role)} replace />;
}
