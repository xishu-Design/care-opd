import { PageHeader } from "../../components/common/PageHeader";
import { KnowledgeGraphView } from "../../components/knowledgeGraph";

export function DoctorKnowledgeGraphPage() {
  return <><PageHeader title="中医知识图谱" subtitle="浏览来源可追溯的药材、方剂、组成关系与传统配伍禁忌" /><KnowledgeGraphView initialLabel="黄芪" /></>;
}
