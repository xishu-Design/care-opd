import { apiRequest } from "./http";

export interface LightRAGHealth {
  enabled?: boolean;
  backend?: string;
  mode?: string;
  status?: string;
  working_dir?: string;
  input_dir?: string;
  docs_count?: number;
  docs_indexed?: number;
  last_index_at?: string;
  index_status?: string;
  degraded?: boolean;
  degraded_reason?: string;
  [key: string]: unknown;
}

export interface LightRAGReports {
  doc_build_report?: Record<string, unknown>;
  index_report?: Record<string, unknown>;
  query_report?: Record<string, unknown>;
}

export type KnowledgeNodeType = "herb" | "formula" | "source" | "case" | "efficacy" | "symptom" | "syndrome" | "concept" | "unknown";

export interface KnowledgeGraphNode {
  id: string;
  label: string;
  type: KnowledgeNodeType;
  description: string;
  source_ids: string[];
  visibility: string[];
}

export interface KnowledgeGraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  source_ids: string[];
  evidence_count?: number;
}

export interface KnowledgeGraphResponse {
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  meta: {
    backend: string;
    degraded: boolean;
    fallback_used?: boolean;
    truncated?: boolean;
    filtered_nodes?: number;
    max_depth?: number;
    max_nodes?: number;
    snapshot_id?: string;
    degraded_reason?: string;
    lightrag_status?: "available" | "unavailable";
    lightrag_anchor_nodes?: number;
    lightrag_raw_nodes?: number;
    lightrag_raw_edges?: number;
    relationship_backend?: string;
    query_correction?: string;
  };
}

export interface KnowledgeGraphLabels {
  items: Array<{ label: string; entity_type: KnowledgeNodeType; source_ids: string[]; visibility: string[] }>;
  backend: string;
  degraded: boolean;
  fallback_used?: boolean;
}

export const lightragApi = {
  getHealth() {
    return apiRequest<LightRAGHealth>({ method: "GET", url: "/api/lightrag/health" });
  },
  getReport() {
    return apiRequest<LightRAGReports>({ method: "GET", url: "/api/lightrag/report" });
  },
  buildDocs() {
    return apiRequest<Record<string, unknown>>({ method: "POST", url: "/api/lightrag/build-docs" });
  },
  reindex() {
    return apiRequest<Record<string, unknown>>({ method: "POST", url: "/api/lightrag/reindex" });
  },
  getGraphLabels(query = "", limit = 50) {
    return apiRequest<KnowledgeGraphLabels>({ method: "GET", url: "/api/lightrag/graph/labels", params: { q: query, limit } });
  },
  getGraph(label = "", maxDepth = 2, maxNodes = 200) {
    return apiRequest<KnowledgeGraphResponse>({ method: "GET", url: "/api/lightrag/graph", params: { label, max_depth: maxDepth, max_nodes: maxNodes } });
  },
};
