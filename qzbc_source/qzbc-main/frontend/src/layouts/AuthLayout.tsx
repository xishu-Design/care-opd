import { Outlet } from "react-router-dom";

export function AuthLayout() {
  return (
    <div className="auth-shell">
      <div className="auth-herb-motif" aria-hidden="true">
        <span>本草</span>
        <small>HERBAL NOTES</small>
      </div>
      <main className="auth-panel">
        <Outlet />
      </main>
    </div>
  );
}
