import axios, { AxiosError, type AxiosRequestConfig } from "axios";

import { API_BASE_URL } from "../app/constants";
import { useAuthStore } from "../stores/authStore";
import type { ApiEnvelope } from "./types";

export class ApiError extends Error {
  status?: number;
  errorCode?: string;
  details?: Record<string, unknown>;

  constructor(message: string, status?: number, errorCode?: string, details?: Record<string, unknown>) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errorCode = errorCode;
    this.details = details;
  }
}

const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20_000,
});

http.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

http.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiEnvelope<unknown>>) => {
    const status = error.response?.status;
    const payload = error.response?.data;
    if (status === 401) {
      useAuthStore.getState().clearAuth();
      if (window.location.pathname !== "/role") {
        window.location.assign("/role");
      }
    }
    return Promise.reject(
      new ApiError(
        payload?.message || error.message || "请求失败",
        status,
        payload?.error_code,
        payload?.details,
      ),
    );
  },
);

export async function apiRequest<T>(config: AxiosRequestConfig): Promise<T> {
  const response = await http.request<ApiEnvelope<T>>(config);
  const payload = response.data;
  if (!payload.success) {
    throw new ApiError(payload.message || "请求失败", response.status, payload.error_code, payload.details);
  }
  return payload.data;
}
