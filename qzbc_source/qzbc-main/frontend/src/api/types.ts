export type Role = "doctor" | "patient" | "student";

export interface ApiEnvelope<T> {
  success: boolean;
  data: T;
  message?: string;
  error_code?: string;
  details?: Record<string, unknown>;
}

export interface ApiUser {
  id: number;
  username: string;
  role: Role;
  display_name?: string;
  is_active?: boolean;
  must_change_password?: boolean;
}

export interface LongTermMemory {
  id: number;
  category: string;
  category_label: string;
  memory_key: string;
  content: string;
  importance?: number;
  confidence?: number;
  source_mode?: string;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface AuthPayload {
  access_token: string;
  token_type: string;
  user: ApiUser;
}

export interface PatientProfile {
  id: number;
  username: string;
  display_name: string;
  age?: number | null;
  sex?: string | null;
  height_cm?: number | null;
  weight_kg?: number | null;
  allergy_history?: string;
  medical_history?: string;
  body_state?: string;
  current_symptoms?: string;
}

export interface PrescriptionItem {
  id?: number;
  herb_name: string;
  dose?: string | null;
  unit?: string | null;
  notes?: string | null;
}

export interface Prescription {
  id: number;
  patient_id: number;
  doctor_id: number;
  title: string;
  symptom_context?: string;
  patient_age_at_visit?: number | null;
  patient_sex_at_visit?: string | null;
  status: string;
  usage_method?: string | null;
  doctor_notes?: string | null;
  items: PrescriptionItem[];
  patient_instruction?: {
    content: string;
    safety_statement: string;
    doctor_note?: string;
    doctor_usage_method?: string;
    item_notes?: Array<{ herb_name: string; note: string }>;
  };
  diet_precaution?: {
    content: string;
    doctor_note?: string;
    doctor_usage_method?: string;
    ai_explanation?: string;
    food_interactions?: Array<{
      herb_name: string;
      food_items: string[];
      food_categories: string[];
      relation_label: string;
      message: string;
      source_id: string;
      source_record_id?: string;
      source_url?: string;
    }>;
    herb_interactions?: Array<{
      herb_name: string;
      incompatible_with: string;
      relation_label: string;
      source_id: string;
      source_record_id?: string;
      source_url?: string;
    }>;
    food_evidence_missing?: boolean;
    sources?: Array<{ source_id: string; url?: string }>;
    llm?: { provider: string; model: string; fallback_used: boolean; degraded: boolean };
  };
}

export interface EvidencePacket {
  packet_id?: string;
  trace_id?: string;
  role?: string;
  task_type?: string;
  visibility?: string;
  case_summary?: string;
  retrieved_evidence?: Array<Record<string, unknown>>;
  graph_reasoning_path?: Array<Record<string, unknown>>;
  graph_node_path?: string[];
  rule_findings?: Array<Record<string, unknown>>;
  memory_used?: Array<Record<string, unknown>>;
  tool_calls?: Array<Record<string, unknown>>;
  backend_summary?: Record<string, unknown>;
  degraded?: boolean;
  degraded_reasons?: string[];
  safety_decision?: Record<string, unknown>;
  human_confirm?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ReviewResult {
  prescription_id: number;
  risk_level?: string;
  review_summary?: string;
  risk_cards?: Array<Record<string, unknown>>;
  rule_findings?: Array<Record<string, unknown>>;
  evidence_packet?: EvidencePacket;
  evidence_packet_id?: string;
  trace_id?: string;
  thread_id?: string;
  graph_node_path?: string[];
  requires_human_confirm?: boolean;
  interrupted?: boolean;
  human_confirmed?: boolean;
  resume_action?: string | null;
  pending_action_status?: string | null;
  pending_action?: PendingHumanAction | null;
  prescription_status?: string;
  degraded?: boolean;
  fallback_used?: boolean;
  generated_response?: GeneratedResponsePayload;
  llm_result?: LLMResultPayload;
}

export interface GeneratedResponsePayload {
  title?: string;
  summary?: string;
  sections?: Array<{ heading?: string; content?: string; source_ids?: string[] }>;
  warnings?: string[];
  next_actions?: string[];
  disclaimer?: string;
}

export interface LLMResultPayload {
  provider?: string;
  model?: string;
  content?: string;
  safety_flags?: string[];
  metadata?: {
    fallback_used?: boolean;
    fallback_reason?: string | null;
    degraded?: boolean;
    latency_ms?: number;
    provider_unavailable?: boolean;
    parse_success?: boolean;
    schema_valid?: boolean;
    parser_error?: string | null;
    experiment_id?: string | null;
    token_usage?: Record<string, number>;
  };
  grounding?: {
    source_ids?: string[];
    used_evidence_count?: number;
    missing_evidence?: boolean;
  };
}

export interface AIChatResult {
  answer?: string;
  trace_id?: string;
  evidence_packet_id?: string;
  evidence_packet?: EvidencePacket;
  generated_response?: GeneratedResponsePayload;
  llm_result?: LLMResultPayload;
  degraded?: boolean;
  fallback_used?: boolean;
  chat_id?: number;
  created_at?: string;
}

export interface AIChatHistoryItem {
  id: number;
  role: "doctor" | "student";
  question: string;
  answer: string;
  trace_id?: string;
  evidence_packet?: EvidencePacket;
  generated_response?: GeneratedResponsePayload;
  llm_result?: LLMResultPayload;
  created_at: string;
}

export interface PendingHumanAction {
  action_id: string;
  trace_id: string;
  thread_id: string;
  graph_name: string;
  node_name: string;
  prescription_id: number;
  patient_id: number;
  doctor_id: number;
  evidence_packet_id: string;
  status: string;
  risk_level?: string | null;
  review_summary?: Record<string, unknown>;
  interrupt_payload?: Record<string, unknown>;
  resume_payload?: Record<string, unknown>;
  resume_comment?: string | null;
  created_at?: string;
  resumed_at?: string | null;
  expires_at?: string | null;
}

export interface TraceDetail {
  trace_id: string;
  summary: Record<string, unknown>;
  events: Array<Record<string, unknown>>;
  tool_calls: Array<Record<string, unknown>>;
  evidence_packets: Array<Record<string, unknown>>;
  pending_human_actions?: PendingHumanAction[];
  llm_generations?: Array<Record<string, unknown>>;
  safety_logs: Array<Record<string, unknown>>;
}

export interface LearningCase {
  id: number;
  title: string;
  anonymized_case: Record<string, unknown>;
  standard_analysis?: string | null;
  risk_points: Array<Record<string, unknown> | string>;
  evidence_chain: Array<Record<string, unknown>>;
}

export interface StudentAnalysis {
  id: number;
  title?: string;
  anonymized_case?: Record<string, unknown>;
  standard_analysis?: string;
  risk_points?: Array<Record<string, unknown> | string>;
  evidence_chain?: Array<Record<string, unknown>>;
  knowledge_cards?: Array<Record<string, unknown>>;
  quiz?: Array<Record<string, unknown>>;
  evidence_packet?: EvidencePacket;
  evidence_packet_id?: string;
  trace_id?: string;
  graph_node_path?: string[];
  generated_response?: GeneratedResponsePayload;
  llm_result?: LLMResultPayload;
}

export interface StudentLearningProgress {
  completed_training_count: number;
  submission_count: number;
  quiz_answered_count: number;
  wrong_quiz_count: number;
  quiz_accuracy?: number | null;
  wrong_quiz_case_ids: number[];
  review_suggestion: string;
  recent_records: Array<{
    answer_id: number;
    case_id: number;
    case_title: string;
    kind: string;
    correct?: boolean | null;
    created_at: string;
  }>;
}

export interface HarnessRun {
  run_id: number;
  phase: string;
  suite: string;
  status: string;
  total_cases: number;
  passed: number;
  failed: number;
  pass_rate: number;
  categories: Record<string, { passed: number; failed: number }>;
  metrics?: Record<string, unknown>;
  markdown_summary?: string;
  duration_ms?: number;
}

export interface HarnessReport {
  run_id: number;
  phase: string;
  suite: string;
  summary: Record<string, unknown>;
  competition_metrics: Record<string, unknown>;
  category_results: Record<string, unknown>;
  failed_cases: Array<Record<string, unknown>>;
  key_traces: string[];
  langgraph_node_summary: Array<Record<string, unknown>>;
  evidence_packet_examples: string[];
  markdown_summary: string;
}

export interface EvaluationResultItem {
  case_id: string;
  dataset: string;
  category: string;
  metric_name: string;
  severity: string;
  passed: boolean;
  actual: Record<string, unknown>;
  expected: Record<string, unknown>;
  trace_id?: string | null;
  evidence_packet_id?: string | null;
  failure_reason?: string | null;
  duration_ms?: number;
}

export interface EvaluationScores {
  overall_score: number;
  safety_score: number;
  evidence_score: number;
  agent_trace_score: number;
  llm_safety_score: number;
}

export interface EvaluationRun extends EvaluationScores {
  run_id: string;
  suite: string;
  dataset_version: string;
  status: string;
  total_cases: number;
  passed: number;
  failed: number;
  pass_rate: number;
  metrics: Record<string, Record<string, number>>;
  duration_ms?: number;
  created_at?: string;
  competition_summary?: string;
  markdown_report?: string;
  key_failures?: EvaluationResultItem[];
  results?: EvaluationResultItem[];
  ppt_ready_tables?: Array<Record<string, unknown>>;
  ppt_ready_charts?: Array<Record<string, unknown>>;
}

export interface AblationExperiment {
  name: string;
  label: string;
  target_metric: string;
  baseline_value: number;
  ablated_value: number;
  delta: number;
  score_drop: number;
  proves: string;
  run_id?: string;
  fallback_visible?: boolean;
  template?: Record<string, unknown>;
  ollama?: Record<string, unknown>;
}

export interface AblationReport {
  ablation_id: string;
  status: string;
  suite: string;
  baseline_run_id: string;
  experiment_count: number;
  baseline: EvaluationScores & { run_id: string; total_cases: number; pass_rate: number };
  experiments: AblationExperiment[];
  competition_summary: string;
  duration_ms: number;
  ppt_ready_tables: Array<Record<string, unknown>>;
  ppt_ready_charts: Array<Record<string, unknown>>;
}

export interface Phase9SourceItem {
  source_id: string;
  name: string;
  homepage?: string;
  repository_url?: string;
  license?: string;
  status?: string;
  approval_status?: "approved" | "pending" | "rejected" | string;
  allowed_use?: string;
  downloaded?: boolean;
  download_status?: string;
  can_process?: boolean;
  license_status?: string;
  commit_hash?: string;
  downloaded_at?: string;
  imported_count?: number;
  expected_content?: string[];
  notes?: string;
}

export interface Phase9SourceRegistryReport {
  summary: Record<string, unknown>;
  sources: Phase9SourceItem[];
  reports: {
    download_report?: Record<string, unknown>;
    license_report?: Record<string, unknown>;
  };
}

export interface Phase9DataPipelineReport {
  summary: Record<string, unknown>;
  processed_counts: Record<string, number>;
  content_counts: Record<string, number>;
  content_stats: Record<string, unknown>;
  content_quality: {
    metrics?: Record<string, unknown>;
    critical_issues?: Array<Record<string, unknown>>;
    high_issues?: Array<Record<string, unknown>>;
    warnings?: Array<Record<string, unknown>>;
    [key: string]: unknown;
  };
  reports: {
    download_report?: Record<string, unknown>;
    license_report?: Record<string, unknown>;
    processing_report?: Record<string, unknown>;
    quality_report?: Record<string, unknown>;
  };
}

export interface DemoStatus {
  environment: {
    app_env: string;
    phase: string;
    version: string;
  };
  backend: {
    api_alive: boolean;
    database_ok: boolean;
    alembic_current: boolean;
  };
  accounts: Record<"doctor_demo_exists" | "patient_demo_exists" | "student_demo_exists", boolean>;
  bindings: {
    doctor_patient_binding_exists: boolean;
  };
  demo_data: Record<string, boolean>;
  content: Record<string, number>;
  llm: {
    enabled: boolean;
    provider: string;
    active_provider?: string;
    configured_provider?: string;
    status: string;
    configured?: boolean;
    reachable?: boolean;
    degraded?: boolean;
    fallback_available: boolean;
    fallback_provider?: string;
    model?: string | null;
    latency_ms?: number | null;
    last_error?: string | null;
    providers?: ProviderHealthPayload[];
  };
  lightrag: {
    enabled: boolean;
    backend: string;
    status: string;
    fallback_available: boolean;
    mode?: string;
    docs_count?: number;
  };
  harness: {
    last_run_id?: number | null;
    last_phase?: string | null;
    last_pass_rate?: number | null;
  };
  demo_reset?: {
    last_reset_id?: string | null;
    last_status?: string | null;
    last_reset_at?: string | null;
  };
  observability?: {
    latest_trace_id?: string | null;
    latest_evidence_packet_id?: string | null;
  };
  readiness: {
    ready: boolean;
    blocking_issues: string[];
    warnings: string[];
  };
}

export interface ProviderHealthPayload {
  name: string;
  configured: boolean;
  reachable: boolean;
  status: string;
  latency_ms?: number | null;
  model?: string | null;
  last_error?: string | null;
}

export interface LLMExperimentItem {
  case_id: string;
  role: Role;
  task_type: string;
  severity: string;
  passed: boolean;
  provider: string;
  model?: string;
  latency_ms?: number;
  fallback_used?: boolean;
  fallback_reason?: string | null;
  parse_success?: boolean;
  schema_valid?: boolean;
  parser_error?: string | null;
  safety_flags?: string[];
  source_ids?: string[];
  expected_checks?: Record<string, boolean>;
  result?: LLMResultPayload;
}

export interface LLMExperimentReport {
  experiment_id: string;
  provider: string;
  model: string;
  suite: string;
  total_cases: number;
  success: number;
  failed: number;
  fallback_count: number;
  parse_success_rate: number;
  schema_valid_rate: number;
  safety_pass_rate: number;
  avg_latency_ms: number;
  categories: Record<string, Record<string, number>>;
  items: LLMExperimentItem[];
}

export interface DemoResetResult {
  reset_id: string;
  status: string;
  created: Record<string, number>;
  demo_ids: Record<string, number>;
  accounts: Record<string, { username: string; password: string }>;
}

export interface DemoSmokeCheck {
  name: string;
  passed: boolean;
  severity: "critical" | "high" | "medium" | "low";
  details: string;
  metadata?: Record<string, unknown>;
}

export interface DemoSmokeResult {
  smoke_id: string;
  passed: boolean;
  total_checks: number;
  passed_checks: number;
  failed_checks: number;
  checks: DemoSmokeCheck[];
  warnings: string[];
}
