import { useQuery } from "@tanstack/react-query";
import { Button, Input, Select, Space, Table, Tag, Typography, type TableProps } from "antd";
import { Search } from "lucide-react";
import { useState } from "react";

import { knowledgeApi, type KnowledgeCatalogEntry } from "../../api/knowledgeApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { KnowledgeEntryDrawer } from "../../components/knowledge/KnowledgeEntryDrawer";

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String).filter(Boolean) : [];
}

export function KnowledgeCatalogPage({ entryType, title, subtitle }: { entryType: "herb" | "formula"; title: string; subtitle: string }) {
  const [draftQuery, setDraftQuery] = useState("");
  const [query, setQuery] = useState("");
  const [sourceId, setSourceId] = useState<string>();
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<KnowledgeCatalogEntry | null>(null);
  const entries = useQuery({ queryKey: ["knowledge", "entries", entryType, query, sourceId, page], queryFn: () => knowledgeApi.listEntries({ entryType, query, sourceId, page, pageSize: 15 }) });
  const sources = useQuery({ queryKey: ["knowledge", "sources"], queryFn: knowledgeApi.listSources, staleTime: 60_000 });

  const columns: TableProps<KnowledgeCatalogEntry>["columns"] = [
    {
      title: entryType === "herb" ? "药材名称" : "方剂名称",
      dataIndex: "title",
      width: 260,
      fixed: "left",
      render: (value: string, row) => {
        const aliases = stringList(row.aliases);
        return <Space direction="vertical" size={0}><Typography.Text strong>{value}</Typography.Text>{aliases.length ? <Typography.Text type="secondary">别名：{aliases.join("、")}</Typography.Text> : null}</Space>;
      },
    },
    entryType === "formula" ? { title: "中药组成", dataIndex: "component_names", width: 320, render: (values: unknown) => { const names = stringList(values); return names.length ? names.join("、") : "来源未提供组成记录"; } } : { title: "功效", dataIndex: "functions", width: 260, render: (values: unknown) => { const functions = stringList(values); return functions.length ? functions.join("、") : "来源未提供功效字段"; } },
    entryType === "formula" ? { title: "中医功效", dataIndex: "functions", width: 240, render: (values: unknown) => { const functions = stringList(values); return functions.length ? functions.join("、") : "来源未提供功效字段"; } } : { title: "关联证候（经方剂）", dataIndex: "indirect_syndromes", width: 300, render: (values: unknown) => { const syndromes = stringList(values); return syndromes.length ? <Space wrap>{syndromes.slice(0, 8).map((value) => <Tag key={value} color="green">{value}</Tag>)}</Space> : "暂无来源方剂关联"; } },
    ...(entryType === "herb" ? [{ title: "关联疾病/症状（经方剂）", dataIndex: "indirect_symptoms", width: 300, render: (values: unknown) => { const symptoms = stringList(values); return symptoms.length ? symptoms.slice(0, 8).join("、") : "暂无来源方剂关联"; } }] : []),
    ...(entryType === "formula" ? [{ title: "中医证候/疾病术语", width: 320, render: (_: unknown, row: KnowledgeCatalogEntry) => [stringList(row.syndromes).join("、"), stringList(row.symptoms).join("、")].filter(Boolean).join("；") || "来源未提供术语关联" }] : []),
    { title: "数据来源", dataIndex: "source_title", width: 220, render: (value: string, row) => <Space direction="vertical" size={0}><Typography.Text>{value}</Typography.Text><Typography.Text type="secondary">{row.source_id}</Typography.Text></Space> },
    ...(entryType === "herb" ? [{
      title: "配伍禁忌药材",
      dataIndex: "incompatible_herbs",
      width: 300,
      render: (values: unknown) => {
        const relations = Array.isArray(values) ? values as KnowledgeCatalogEntry["incompatible_herbs"] : [];
        return relations.length
          ? <Space wrap>{relations.map((relation) => <Tag color="volcano" key={`${relation.category}:${relation.herb_name}`}>{relation.herb_name}（{relation.category}）</Tag>)}</Space>
          : <Typography.Text type="secondary">当前标准未收录相关禁配关系</Typography.Text>;
      },
    }] : []),
    { title: "操作", width: 90, fixed: "right", render: (_, row) => <Button type="link" onClick={() => setSelected(row)}>详情</Button> },
  ];

  if (entries.isLoading) return <LoadingState />;
  if (entries.isError) return <ErrorState error={entries.error} />;
  if (!entries.data) return <LoadingState />;
  const data = entries.data;
  return (
    <>
      <PageHeader title={title} subtitle={subtitle} />
      <section className="data-workbench">
        <div className="data-filter-bar">
          <Input
            value={draftQuery}
            allowClear
            prefix={<Search size={16} />}
            placeholder={entryType === "herb" ? "搜索药材名称、别名或来源" : "搜索方剂名称、组成或来源"}
            onChange={(event) => setDraftQuery(event.target.value)}
            onPressEnter={() => { setPage(1); setQuery(draftQuery.trim()); }}
          />
          <Select
            allowClear
            placeholder="全部来源"
            value={sourceId}
            onChange={(value) => { setPage(1); setSourceId(value); }}
            options={(sources.data || []).map((source) => ({ value: source.source_id, label: source.source_title }))}
          />
          <Button type="primary" icon={<Search size={16} />} onClick={() => { setPage(1); setQuery(draftQuery.trim()); }}>查询</Button>
        </div>
        <Table
          className="data-workbench-table"
          rowKey={(row) => `${row.source_id}:${row.entry_id}`}
          size="middle"
          columns={columns}
          dataSource={data.items}
          scroll={{ x: entryType === "formula" ? 1600 : 1580 }}
          pagination={{ current: data.page, pageSize: data.page_size, total: data.total, showSizeChanger: false, onChange: setPage, showTotal: (total) => `共 ${total} 条` }}
        />
      </section>
      <KnowledgeEntryDrawer entry={selected} onClose={() => setSelected(null)} />
    </>
  );
}
