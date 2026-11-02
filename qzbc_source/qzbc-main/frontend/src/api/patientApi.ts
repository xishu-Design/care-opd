import { apiRequest } from "./http";
import { postChatStream, type ChatStreamEvent } from "./chatStream";
import type { GeneratedResponsePayload, LLMResultPayload, LongTermMemory, PatientProfile, Prescription, ReviewResult } from "./types";

export type EducationChatResult = ReviewResult & { allowed?: boolean; safe_reply?: string; reason?: string; chat_id?: number; created_at?: string };
export type EducationChatHistoryItem = {
  id: number;
  question: string;
  answer: string;
  allowed: boolean;
  trace_id?: string;
  evidence_packet?: Record<string, unknown>;
  generated_response?: GeneratedResponsePayload;
  llm_result?: LLMResultPayload;
  created_at: string;
};

export type EducationStreamEvent = ChatStreamEvent<EducationChatResult>;

export const patientApi = {
  getProfile() {
    return apiRequest<PatientProfile>({ method: "GET", url: "/api/patient/profile" });
  },
  listPrescriptions() {
    return apiRequest<Prescription[]>({ method: "GET", url: "/api/patient/prescriptions" });
  },
  getPrescriptionDetail(prescriptionId: number) {
    return apiRequest<Prescription>({ method: "GET", url: `/api/patient/prescriptions/${prescriptionId}` });
  },
  educationChat(message: string) {
    return apiRequest<EducationChatResult>({
      method: "POST",
      url: "/api/patient/education-chat",
      data: { message },
    });
  },
  educationHistory() {
    return apiRequest<EducationChatHistoryItem[]>({ method: "GET", url: "/api/patient/education-chat/history" });
  },
  deleteEducationChat(chatId: number) {
    return apiRequest<{ deleted: boolean; chat_id: number }>({ method: "DELETE", url: `/api/patient/education-chat/${chatId}` });
  },
  listAIMemories() {
    return apiRequest<LongTermMemory[]>({ method: "GET", url: "/api/patient/ai-memory" });
  },
  rememberAI(content: string) {
    return apiRequest<{ memory_id: number }>({ method: "POST", url: "/api/patient/ai-memory", data: { content } });
  },
  forgetAI(memoryId: number) {
    return apiRequest<{ deleted: boolean; memory_id: number }>({ method: "DELETE", url: `/api/patient/ai-memory/${memoryId}` });
  },
  educationChatStream(message: string, onEvent: (event: EducationStreamEvent) => void, signal?: AbortSignal) {
    return postChatStream<EducationChatResult>("/api/patient/education-chat/stream", message, onEvent, signal);
  },
};
