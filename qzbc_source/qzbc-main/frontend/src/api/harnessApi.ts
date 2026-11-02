import { apiRequest } from "./http";
import type { HarnessReport, HarnessRun, Phase9DataPipelineReport, Phase9SourceRegistryReport, TraceDetail } from "./types";

export const harnessApi = {
  listTraces() {
    return apiRequest<{ items: Array<{ trace_id: string; role: string; task_type: string; status: string; created_at: string }> }>({
      method: "GET",
      url: "/api/harness/traces",
    });
  },
  getTrace(traceId: string) {
    return apiRequest<TraceDetail>({ method: "GET", url: `/api/harness/traces/${traceId}` });
  },
  runHarness(payload: { phase?: string; suite?: string; reset_demo_data?: boolean }) {
    return apiRequest<HarnessRun>({ method: "POST", url: "/api/harness/run", data: payload });
  },
  getHarnessReport(runId: number) {
    return apiRequest<HarnessReport>({ method: "GET", url: `/api/harness/runs/${runId}/report` });
  },
  getSources() {
    return apiRequest<Phase9SourceRegistryReport>({ method: "GET", url: "/api/harness/sources" });
  },
  getDataPipeline() {
    return apiRequest<Phase9DataPipelineReport>({ method: "GET", url: "/api/harness/data-pipeline" });
  },
};
