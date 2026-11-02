import { Navigate, Outlet, useLocation } from "react-router-dom";

import { useAuthStore } from "../stores/authStore";

export function ProtectedRoute() {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const user = useAuthStore((state) => state.user);
  const location = useLocation();
  if (!isAuthenticated) {
    return <Navigate to="/role" replace state={{ from: location.pathname }} />;
  }
  if (user?.must_change_password && location.pathname !== "/account/password") {
    return <Navigate to="/account/password" replace />;
  }
  return <Outlet />;
}
