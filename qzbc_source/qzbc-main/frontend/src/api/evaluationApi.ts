import { apiRequest } from "./http";
import type { AblationReport, EvaluationRun } from "./types";


export const evaluationApi = {
  runEvaluation(payload: { suite: "quick" | "full"; reset_demo_data?: boolean }) {
    return apiRequest<EvaluationRun>({ method: "POST", url: "/api/evaluation/run", data: payload });
  },
  listRuns() {
    return apiRequest<EvaluationRun[]>({ method: "GET", url: "/api/evaluation/runs" });
  },
  getRun(runId: string) {
    return apiRequest<EvaluationRun>({ method: "GET", url: `/api/evaluation/runs/${runId}` });
  },
  getReport(runId: string) {
    return apiRequest<EvaluationRun>({ method: "GET", url: `/api/evaluation/runs/${runId}/report` });
  },
  runAblation(payload: { suite: "quick" | "full"; baseline_run_id?: string }) {
    return apiRequest<AblationReport>({ method: "POST", url: "/api/evaluation/ablation/run", data: payload });
  },
  getLatestAblation() {
    return apiRequest<AblationReport>({ method: "GET", url: "/api/evaluation/ablation/latest" });
  },
};
