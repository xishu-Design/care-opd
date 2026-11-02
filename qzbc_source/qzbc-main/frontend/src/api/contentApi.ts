import { apiRequest } from "./http";

export interface ContentStats {
  sources: number;
  knowledge_entries: number;
  herbs?: number;
  formulas?: number;
  contraindications?: number;
  diet_precautions?: number;
  public_education?: number;
  rules?: number;
  learning_cases: number;
  quizzes: number;
  patient_safe_entries: number;
  student_learning_cases: number;
  coverage: Record<string, number>;
}

export const contentApi = {
  getStats() {
    return apiRequest<ContentStats>({ method: "GET", url: "/api/content/stats" });
  },
};
