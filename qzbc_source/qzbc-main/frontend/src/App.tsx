import { Outlet } from "react-router-dom";
import { useEffect } from "react";

import { authApi } from "./api/authApi";
import { GlobalDrawers } from "./components/GlobalDrawers";
import { useAuthStore } from "./stores/authStore";

export function App() {
  const token = useAuthStore((state) => state.token);
  const updateUser = useAuthStore((state) => state.updateUser);
  useEffect(() => {
    if (!token) return;
    authApi.me().then(updateUser).catch(() => undefined);
  }, [token, updateUser]);
  return (
    <>
      <Outlet />
      <GlobalDrawers />
    </>
  );
}
