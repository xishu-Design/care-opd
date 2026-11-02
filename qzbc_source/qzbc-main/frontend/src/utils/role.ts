import { DASHBOARD_BY_ROLE, ROLE_LABELS } from "../app/constants";
import type { Role } from "../api/types";

export const roles: Role[] = ["doctor", "patient", "student"];

export function isRole(value: string | null): value is Role {
  return value === "doctor" || value === "patient" || value === "student";
}

export function roleLabel(role?: Role | null) {
  return role ? ROLE_LABELS[role] : "未选择";
}

export function dashboardPath(role: Role) {
  return DASHBOARD_BY_ROLE[role];
}
