import { PageHeader } from "../../components/common/PageHeader";
import { KnowledgeGraphView } from "../../components/knowledgeGraph";

export function StudentKnowledgeGraphPage() {
  return <><PageHeader title="学习知识图谱" subtitle="浏览学生可见的脱敏案例、药材、方剂与来源关系" /><KnowledgeGraphView initialLabel="黄芪" /></>;
}
