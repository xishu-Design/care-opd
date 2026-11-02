import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { ApiUser, Role } from "../api/types";

interface AuthState {
  token: string | null;
  user: ApiUser | null;
  role: Role | null;
  isAuthenticated: boolean;
  setAuth: (token: string, user: ApiUser) => void;
  updateUser: (user: ApiUser) => void;
  clearAuth: () => void;
  hasRole: (role: Role) => boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      role: null,
      isAuthenticated: false,
      setAuth: (token, user) => set({ token, user, role: user.role, isAuthenticated: true }),
      updateUser: (user) => set({ user, role: user.role }),
      clearAuth: () => set({ token: null, user: null, role: null, isAuthenticated: false }),
      hasRole: (role) => get().role === role,
    }),
    {
      name: "guardtcm-auth",
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        role: state.role,
        isAuthenticated: state.isAuthenticated,
      }),
    },
  ),
);
