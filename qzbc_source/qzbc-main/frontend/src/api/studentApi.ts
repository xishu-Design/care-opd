import { apiRequest } from "./http";
import { postChatStream, type ChatStreamEvent } from "./chatStream";
import type { AIChatHistoryItem, AIChatResult, LearningCase, LongTermMemory, StudentAnalysis, StudentLearningProgress } from "./types";

export const studentApi = {
  aiChat(message: string) {
    return apiRequest<AIChatResult>({ method: "POST", url: "/api/student/ai-chat", data: { message }, timeout: 90_000 });
  },
  aiChatStream(message: string, onEvent: (event: ChatStreamEvent<AIChatResult>) => void, signal?: AbortSignal) {
    return postChatStream<AIChatResult>("/api/student/ai-chat/stream", message, onEvent, signal);
  },
  aiChatHistory() {
    return apiRequest<AIChatHistoryItem[]>({ method: "GET", url: "/api/student/ai-chat/history" });
  },
  deleteAIChat(chatId: number) {
    return apiRequest<{ deleted: boolean; chat_id: number }>({ method: "DELETE", url: `/api/student/ai-chat/${chatId}` });
  },
  listAIMemories() {
    return apiRequest<LongTermMemory[]>({ method: "GET", url: "/api/student/ai-memory" });
  },
  rememberAI(content: string) {
    return apiRequest<{ memory_id: number }>({ method: "POST", url: "/api/student/ai-memory", data: { content } });
  },
  forgetAI(memoryId: number) {
    return apiRequest<{ deleted: boolean; memory_id: number }>({ method: "DELETE", url: `/api/student/ai-memory/${memoryId}` });
  },
  listLearningCases(kind?: "clinical" | "syndrome" | "quiz" | "composition") {
    return apiRequest<LearningCase[]>({ method: "GET", url: kind ? `/api/student/learning-cases?kind=${kind}` : "/api/student/learning-cases" });
  },
  getLearningCase(caseId: number) {
    return apiRequest<LearningCase>({ method: "GET", url: `/api/student/learning-cases/${caseId}` });
  },
  submitReviewAnswer(caseId: number, payload: { answer_text: string; proposed_herbs: string[]; has_risk: boolean; risk_level?: string }) {
    return apiRequest<{ id: number; learning_case_id: number; feedback: string; evaluation: {
      matched_herbs: string[];
      missing_herbs: string[];
      extra_herbs: string[];
      knowledge_points: string[];
      model_feedback: string;
      model_provider: string;
      judgment?: "correct" | "partial" | "incorrect" | "not_scored";
      score?: number | null;
      /** Present for deterministic MTCMB multiple-choice grading. */
      submitted_answer?: string;
      expected_answer?: string;
      is_correct?: boolean;
      semantic_assessments?: Array<{ herb_name: string; verdict: string; direct_symptom_matches?: string[]; function_matches?: string[]; known_functions?: string[] }>;
    } }>({
      method: "POST",
      url: `/api/student/learning-cases/${caseId}/submit-answer`,
      data: payload,
    });
  },
  getStandardAnalysis(caseId: number) {
    return apiRequest<StudentAnalysis>({ method: "GET", url: `/api/student/learning-cases/${caseId}/standard-analysis`, timeout: 90_000 });
  },
  generateTeachingExplanation(caseId: number) {
    return apiRequest<Pick<StudentAnalysis, "generated_response" | "llm_result">>({ method: "POST", url: `/api/student/learning-cases/${caseId}/teaching-explanation`, timeout: 90_000 });
  },
  getLearningProgress() {
    return apiRequest<StudentLearningProgress>({ method: "GET", url: "/api/student/learning-progress" });
  },
};
