import { apiRequest } from "./http";
import type { ApiUser, AuthPayload, Role } from "./types";

export const authApi = {
  register(payload: { username: string; password: string; role: Role; display_name: string }) {
    return apiRequest<ApiUser>({ method: "POST", url: "/api/auth/register", data: payload });
  },
  login(payload: { username: string; password: string; entry_role: Role }) {
    return apiRequest<AuthPayload>({ method: "POST", url: "/api/auth/login", data: payload });
  },
  me() {
    return apiRequest<ApiUser>({ method: "GET", url: "/api/auth/me" });
  },
  changePassword(payload: { current_password: string; new_password: string; confirm_password: string }) {
    return apiRequest<ApiUser>({ method: "POST", url: "/api/auth/change-password", data: payload });
  },
};
