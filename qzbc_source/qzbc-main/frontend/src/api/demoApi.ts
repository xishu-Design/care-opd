import { apiRequest } from "./http";
import type { DemoResetResult, DemoSmokeResult, DemoStatus } from "./types";

export const demoApi = {
  getStatus() {
    return apiRequest<DemoStatus>({ method: "GET", url: "/api/demo/status" });
  },
  resetDemoData(payload: { reset_scope: "demo_only"; reseed: boolean } = { reset_scope: "demo_only", reseed: true }) {
    return apiRequest<DemoResetResult>({ method: "POST", url: "/api/demo/reset", data: payload });
  },
  runSmoke() {
    return apiRequest<DemoSmokeResult>({ method: "POST", url: "/api/demo/smoke" });
  },
};
