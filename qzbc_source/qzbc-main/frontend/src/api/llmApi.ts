import { apiRequest } from "./http";
import type { LLMExperimentReport } from "./types";

export const llmApi = {
  runExperiment(payload: { provider?: string; model?: string | null; suite?: "quick" | "full" } = { provider: "ollama", suite: "quick" }) {
    return apiRequest<LLMExperimentReport>({ method: "POST", url: "/api/llm/experiments/run", data: payload });
  },
  getLatestExperiment() {
    return apiRequest<LLMExperimentReport>({ method: "GET", url: "/api/llm/experiments/latest" });
  },
  getExperiment(experimentId: string) {
    return apiRequest<LLMExperimentReport>({ method: "GET", url: `/api/llm/experiments/${experimentId}` });
  },
};
