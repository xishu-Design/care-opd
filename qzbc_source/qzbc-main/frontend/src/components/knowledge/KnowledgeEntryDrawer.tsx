import { useQuery } from "@tanstack/react-query";
import { Alert, Descriptions, Drawer, List, Space, Spin, Tag, Typography } from "antd";

import { knowledgeApi, type KnowledgeCatalogEntry } from "../../api/knowledgeApi";

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String).filter(Boolean) : [];
}

const entryTypeLabels: Record<string, string> = {
  herb: "药材",
  formula: "方剂",
  contraindication: "禁忌",
  diet_precaution: "饮食注意",
  public_education: "科普内容",
};

const predicateLabels: Record<string, string> = {
  HAS_TCM_FUNCTION: "中医功效",
  HAS_EFFICACY: "中医功效",
  HAS_FUNCTION: "中医功效",
  ASSOCIATED_WITH_TCM_SYMPTOM: "关联症状",
  ASSOCIATED_WITH_TCM_SYNDROME: "关联证候",
  HAS_CHPA_EFFICACY: "CHPA 功效统计关联",
  ASSOCIATED_WITH_CHPA_SYMPTOM: "CHPA 症状统计关联",
  TRADITIONAL_INCOMPATIBILITY_18: "十八反（传统配伍禁忌）",
  TRADITIONAL_INCOMPATIBILITY_19: "十九畏（传统配伍禁忌）",
  CONTAINS: "包含药材",
  REFERENCES_FORMULA: "关联方剂",
};

export function KnowledgeEntryDrawer({ entry, onClose }: { entry: KnowledgeCatalogEntry | null; onClose: () => void }) {
  const detail = useQuery({
    queryKey: ["knowledge", "entry", entry?.source_id, entry?.entry_id],
    queryFn: () => knowledgeApi.getEntry(entry!.source_id, entry!.entry_id),
    enabled: Boolean(entry),
  });
  const data = detail.data;
  const metadata = data?.metadata;
  const aliases = stringList(data?.aliases);
  const componentNames = stringList(data?.component_names);
  const functions = stringList(data?.functions);
  const symptoms = stringList(data?.symptoms);
  const chpaEfficacies = stringList(data?.chpa_efficacies);
  const chpaSymptoms = stringList(data?.chpa_symptoms);
  const indirectSyndromes = stringList(data?.indirect_syndromes);
  const indirectSymptoms = stringList(data?.indirect_symptoms);
  const indirectFunctions = stringList(data?.indirect_functions);
  const incompatibilities = data?.incompatible_herbs || [];

  return (
    <Drawer
      rootClassName="knowledge-detail-drawer"
      title={<span>{entry?.title || "知识条目"}</span>}
      open={Boolean(entry)}
      onClose={onClose}
      width={520}
    >
      {detail.isLoading ? <div className="drawer-loading"><Spin /></div> : null}
      {detail.isError ? <Alert type="error" showIcon message="无法加载条目详情" description="请确认当前账号有该来源的查看权限后重试。" /> : null}
      {data ? (
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="条目类型">{entryTypeLabels[data.entry_type] || data.entry_type}</Descriptions.Item>
            <Descriptions.Item label="来源"><Tag color="blue">{data.source_id}</Tag> {data.source_title}</Descriptions.Item>
            {data.entry_type === "herb" ? <Descriptions.Item label="配伍禁忌药材">{incompatibilities.length ? <Space wrap>{incompatibilities.map((relation) => <Tag color="volcano" key={`${relation.category}:${relation.herb_name}`}>{relation.herb_name}（{relation.category}）</Tag>)}</Space> : "当前标准未收录相关禁配关系"}</Descriptions.Item> : null}
            {aliases.length ? <Descriptions.Item label="别名">{aliases.join("、")}</Descriptions.Item> : null}
            {componentNames.length ? <Descriptions.Item label="组成记录">{componentNames.join("、")}</Descriptions.Item> : null}
            {functions.length ? <Descriptions.Item label="功效">{functions.filter((value) => !chpaEfficacies.includes(value)).join("、") || "暂无人工整理功效"}</Descriptions.Item> : null}
            {symptoms.length ? <Descriptions.Item label="关联症状">{symptoms.filter((value) => !chpaSymptoms.includes(value)).join("、") || "暂无人工整理症状"}</Descriptions.Item> : null}
            {chpaEfficacies.length ? <Descriptions.Item label="CHPA 功效统计关联">{chpaEfficacies.join("、")}</Descriptions.Item> : null}
            {chpaSymptoms.length ? <Descriptions.Item label="CHPA 症状统计关联">{chpaSymptoms.join("、")}</Descriptions.Item> : null}
            {data.chpa_relation_note ? <Descriptions.Item label="CHPA 数据说明">{data.chpa_relation_note}</Descriptions.Item> : null}
            {indirectSyndromes.length ? <Descriptions.Item label="方剂关联证候">{indirectSyndromes.join("、")}</Descriptions.Item> : null}
            {indirectSymptoms.length ? <Descriptions.Item label="方剂关联疾病/症状">{indirectSymptoms.join("、")}</Descriptions.Item> : null}
            {indirectFunctions.length ? <Descriptions.Item label="关联方剂功效">{indirectFunctions.join("、")}</Descriptions.Item> : null}
            {data.formula_relation_note ? <Descriptions.Item label="关系说明">{data.formula_relation_note}</Descriptions.Item> : null}
          </Descriptions>
          <Typography.Paragraph className="pre-wrap knowledge-source-content">{data.content || "暂无来源内容"}</Typography.Paragraph>
          <List
            size="small"
            header="原始记录定位"
            dataSource={metadata?.source_records || []}
            locale={{ emptyText: "未提供原始记录定位" }}
            renderItem={(record) => <List.Item>{record.source_file || "来源文件"} / {record.source_record_id || "记录"}</List.Item>}
          />
          <List
            size="small"
            header="来源方剂关系"
            dataSource={data.related_formulas || []}
            locale={{ emptyText: "暂无来源方剂关系" }}
            renderItem={(relation) => <List.Item><Space direction="vertical" size={2}><Typography.Text>{relation.formula_name}</Typography.Text><Typography.Text type="secondary">证候：{stringList(relation.syndromes).join("、") || "未提供"}；症状/疾病：{stringList(relation.symptoms).join("、") || "未提供"}</Typography.Text></Space></List.Item>}
          />
          <List
            size="small"
            header="结构化知识声明"
            dataSource={metadata?.claims || []}
            locale={{ emptyText: "暂无可追溯声明" }}
            renderItem={(claim) => <List.Item><Space direction="vertical" size={2}><Typography.Text>{predicateLabels[claim.predicate] || "知识关联"}：{claim.object_label}</Typography.Text><Typography.Text type="secondary">数据来源：{claim.source_id}</Typography.Text></Space></List.Item>}
          />
          {metadata?.repository_url ? <Typography.Link href={metadata.repository_url} target="_blank" rel="noreferrer">打开来源仓库</Typography.Link> : null}
        </Space>
      ) : null}
    </Drawer>
  );
}
