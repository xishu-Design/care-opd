export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export const ROLE_LABELS = {
  doctor: "医生端",
  patient: "患者端",
  student: "学习端",
} as const;

export const DASHBOARD_BY_ROLE = {
  doctor: "/doctor/dashboard",
  patient: "/patient/dashboard",
  student: "/student/dashboard",
} as const;
