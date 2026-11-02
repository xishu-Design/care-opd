import { apiRequest } from "./http";
import { postChatStream, type ChatStreamEvent } from "./chatStream";
import type { AIChatHistoryItem, AIChatResult, LongTermMemory, PatientProfile, PendingHumanAction, Prescription, ReviewResult } from "./types";

export interface PatientCreatePayload {
  username: string;
  password?: string;
  display_name: string;
  age?: number;
  sex?: string;
  height_cm?: number;
  weight_kg?: number;
  allergy_history?: string;
  medical_history?: string;
  body_state?: string;
  current_symptoms?: string;
}

export interface PrescriptionCreatePayload {
  patient_id: number;
  title: string;
  symptom_context: string;
  usage_method?: string;
  doctor_notes?: string;
  items: Array<{ herb_name: string; dose?: string; unit?: string; notes?: string }>;
}

export interface PrescriptionSuggestion {
  herb_name: string;
  functions: string[];
  symptoms: string[];
  source_ids: string[];
  risk_labels: string[];
  match_reasons: string[];
  no_dose_or_usage: boolean;
  status: string;
}

export interface PrescriptionSuggestionsResponse {
  patient_id: number;
  symptoms: string;
  query: string;
  allergy_history: string;
  excluded_allergy_herbs: string[];
  excluded_formula_suggestions: Array<{ title: string; matched_allergens: string[] }>;
  suggestions: PrescriptionSuggestion[];
  formula_suggestions: Array<{ title: string; herbs: string[]; functions: string[]; symptoms: string[]; syndromes: string[]; source_ids: string[]; status: string; review_status?: string }>;
  llm_rerank: { used: boolean; provider: string; model?: string; fallback_used?: boolean };
  herb_llm_rerank: { used: boolean; provider: string; model?: string; fallback_used?: boolean };
  no_match_message?: string | null;
  requires_human_confirm: boolean;
  disclaimer: string;
}

export const doctorApi = {
  aiChat(message: string) {
    return apiRequest<AIChatResult>({ method: "POST", url: "/api/doctor/ai-chat", data: { message }, timeout: 90_000 });
  },
  aiChatStream(message: string, onEvent: (event: ChatStreamEvent<AIChatResult>) => void, signal?: AbortSignal) {
    return postChatStream<AIChatResult>("/api/doctor/ai-chat/stream", message, onEvent, signal);
  },
  aiChatHistory() {
    return apiRequest<AIChatHistoryItem[]>({ method: "GET", url: "/api/doctor/ai-chat/history" });
  },
  deleteAIChat(chatId: number) {
    return apiRequest<{ deleted: boolean; chat_id: number }>({ method: "DELETE", url: `/api/doctor/ai-chat/${chatId}` });
  },
  listAIMemories() {
    return apiRequest<LongTermMemory[]>({ method: "GET", url: "/api/doctor/ai-memory" });
  },
  rememberAI(content: string) {
    return apiRequest<{ memory_id: number }>({ method: "POST", url: "/api/doctor/ai-memory", data: { content } });
  },
  forgetAI(memoryId: number) {
    return apiRequest<{ deleted: boolean; memory_id: number }>({ method: "DELETE", url: `/api/doctor/ai-memory/${memoryId}` });
  },
  listPatients() {
    return apiRequest<PatientProfile[]>({ method: "GET", url: "/api/doctor/patients" });
  },
  createPatient(payload: PatientCreatePayload) {
    return apiRequest<PatientProfile>({ method: "POST", url: "/api/doctor/patients", data: payload });
  },
  getPatientDetail(patientId: number) {
    return apiRequest<PatientProfile>({ method: "GET", url: `/api/doctor/patients/${patientId}` });
  },
  updatePatient(patientId: number, payload: Partial<PatientCreatePayload>) {
    return apiRequest<PatientProfile>({ method: "PATCH", url: `/api/doctor/patients/${patientId}`, data: payload });
  },
  createPrescription(payload: PrescriptionCreatePayload) {
    return apiRequest<Prescription>({ method: "POST", url: "/api/doctor/prescriptions", data: payload });
  },
  getPrescriptionSuggestions(payload: { patient_id: number; symptoms?: string; query?: string }) {
    return apiRequest<PrescriptionSuggestionsResponse>({ method: "POST", url: "/api/doctor/prescription-suggestions", data: payload, timeout: 90_000 });
  },
  getPrescriptionDetail(prescriptionId: number) {
    return apiRequest<Prescription>({ method: "GET", url: `/api/doctor/prescriptions/${prescriptionId}` });
  },
  updatePrescription(prescriptionId: number, payload: { symptom_context?: string; usage_method?: string; doctor_notes?: string }) {
    return apiRequest<Prescription>({ method: "PATCH", url: `/api/doctor/prescriptions/${prescriptionId}`, data: payload });
  },
  reviewPrescription(prescriptionId: number) {
    return apiRequest<ReviewResult>({ method: "POST", url: `/api/doctor/prescriptions/${prescriptionId}/review` });
  },
  resumeDoctorReview(traceId: string, payload: { approved: boolean; comment?: string }) {
    return apiRequest<ReviewResult>({
      method: "POST",
      url: `/api/agents/doctor/review/${traceId}/resume`,
      data: payload,
    });
  },
  listPendingActions() {
    return apiRequest<{ items: PendingHumanAction[] }>({ method: "GET", url: "/api/agents/doctor/review/pending" });
  },
  confirmPrescription(prescriptionId: number) {
    return apiRequest<Prescription>({ method: "POST", url: `/api/doctor/prescriptions/${prescriptionId}/confirm` });
  },
  publishPrescription(prescriptionId: number) {
    return apiRequest<Prescription>({ method: "POST", url: `/api/doctor/prescriptions/${prescriptionId}/publish` });
  },
  deidentifyPrescription(prescriptionId: number) {
    return apiRequest<Record<string, unknown>>({ method: "POST", url: `/api/doctor/prescriptions/${prescriptionId}/deidentify` });
  },
};
