import type { KnowledgeGraphEdge, KnowledgeGraphNode, KnowledgeNodeType } from "../../api/lightragApi";

export const NODE_TYPES: KnowledgeNodeType[] = ["herb", "formula", "efficacy", "symptom", "syndrome", "source", "case", "concept", "unknown"];

export const NODE_TYPE_LABELS: Record<KnowledgeNodeType, string> = {
  herb: "药材",
  formula: "方剂",
  source: "来源",
  case: "案例",
  efficacy: "中医功效",
  symptom: "中医症状",
  syndrome: "中医证候",
  concept: "概念",
  unknown: "其他",
};

export interface GraphSelection {
  node: KnowledgeGraphNode;
  relations: KnowledgeGraphEdge[];
}

export type GraphLayoutMode = "focus" | "overview";

export const GRAPH_LAYOUT_LABELS: Record<GraphLayoutMode, string> = {
  focus: "聚焦关系",
  overview: "关系概览",
};

export const RELATION_LABELS: Record<string, string> = {
  HAS_TCM_FUNCTION: "具有中医功效",
  ASSOCIATED_WITH_TCM_SYMPTOM: "关联症状",
  ASSOCIATED_WITH_TCM_SYNDROME: "关联证候",
  HAS_CHPA_EFFICACY: "CHPA 功效统计关联",
  ASSOCIATED_WITH_CHPA_SYMPTOM: "CHPA 症状统计关联",
  TRADITIONAL_INCOMPATIBILITY_18: "十八反（传统配伍禁忌）",
  TRADITIONAL_INCOMPATIBILITY_19: "十九畏（传统配伍禁忌）",
  CONTAINS: "包含药材",
  DERIVED_FROM: "来源于",
  REFERENCES: "引用",
  DESCRIBES: "描述",
};
