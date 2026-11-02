import { apiRequest } from "./http";

export interface KnowledgeCatalogEntry {
  entry_id: string;
  source_id: string;
  entry_type: string;
  title: string;
  keywords: string[];
  aliases: string[];
  component_names: string[];
  source_title: string;
  source_type: string;
  visibility: string[];
  license: string;
  license_note: string;
  commit_hash: string;
  source_records: Array<{ source_id: string; source_record_id: string; source_file: string }>;
  content_preview: string;
  functions: string[];
  symptoms: string[];
  syndromes: string[];
  chpa_efficacies: string[];
  chpa_symptoms: string[];
  chpa_relation_note: string;
  chpa_translation_review_status: string;
  indirect_functions: string[];
  indirect_symptoms: string[];
  indirect_syndromes: string[];
  related_formulas: Array<{ formula_name: string; functions: string[]; symptoms: string[]; syndromes: string[]; source_ids: string[] }>;
  formula_relation_note: string;
  formula_relations: string[];
  claim_status: string[];
  incompatible_herbs: Array<{ herb_name: string; category: string; relation_type: string; source_id: string; source_record_id: string }>;
}

export interface KnowledgeCatalogDetail extends KnowledgeCatalogEntry {
  content: string;
  metadata: {
    source_ids: string[];
    source_records: Array<{ source_id: string; source_record_id: string; source_file: string }>;
    original_source: string;
    repository_url: string;
    license: string;
    license_note: string;
    commit_hash: string;
    processing_script: string;
    processing_version: string;
    evidence_level: string;
    visibility: string[];
    incompatible_herbs: Array<{ herb_name: string; category: string; relation_type: string; source_id: string; source_record_id: string }>;
    aliases: string[];
    component_names: string[];
    indirect_functions: string[];
    indirect_symptoms: string[];
    indirect_syndromes: string[];
    chpa_efficacies: string[];
    chpa_symptoms: string[];
    chpa_relation_note: string;
    chpa_translation_review_status: string;
    related_formulas: Array<{ formula_name: string; functions: string[]; symptoms: string[]; syndromes: string[]; source_ids: string[] }>;
    formula_relation_note: string;
    claims: Array<{ predicate: string; object_label: string; source_id: string; source_record_id: string; source_revision: string; license_status: string; distribution_scope: string; content_risk_level: string; review_status: string; source_excerpt: string }>;
  };
}

export interface KnowledgeCatalogPage {
  items: KnowledgeCatalogEntry[];
  page: number;
  page_size: number;
  total: number;
}

export interface KnowledgeSource {
  source_id: string;
  source_title: string;
  source_type: string;
  source_version: string;
  source_year: string;
  evidence_level: string;
  license_note: string;
}

export const knowledgeApi = {
  listEntries(params: { entryType?: string; query?: string; sourceId?: string; page?: number; pageSize?: number }) {
    return apiRequest<KnowledgeCatalogPage>({
      method: "GET",
      url: "/api/knowledge/entries",
      params: {
        entry_type: params.entryType,
        q: params.query || undefined,
        source_id: params.sourceId || undefined,
        page: params.page || 1,
        page_size: params.pageSize || 20,
      },
    });
  },
  getEntry(sourceId: string, entryId: string) {
    return apiRequest<KnowledgeCatalogDetail>({ method: "GET", url: `/api/knowledge/entries/${sourceId}/${entryId}` });
  },
  listSources() {
    return apiRequest<KnowledgeSource[]>({ method: "GET", url: "/api/knowledge/sources" });
  },
};
