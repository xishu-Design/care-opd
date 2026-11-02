import { useQuery } from "@tanstack/react-query";
import { Drawer, Table, Tag, Typography, type TableProps } from "antd";
import { useState } from "react";

import { knowledgeApi, type KnowledgeSource } from "../../api/knowledgeApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";

export function KnowledgeSourcePage() {
  const sources = useQuery({ queryKey: ["knowledge", "sources"], queryFn: knowledgeApi.listSources });
  const [selected, setSelected] = useState<KnowledgeSource | null>(null);
  const columns: TableProps<KnowledgeSource>["columns"] = [
    { title: "来源标识", dataIndex: "source_id", width: 240, render: (value: string) => <Typography.Text strong>{value}</Typography.Text> },
    { title: "来源名称", dataIndex: "source_title", width: 260 },
    { title: "来源类型", dataIndex: "source_type", width: 150 },
    { title: "版本", dataIndex: "source_version", width: 120, render: (value: string) => value || "-" },
    { title: "年份", dataIndex: "source_year", width: 100, render: (value: string) => value || "-" },
    { title: "证据标记", dataIndex: "evidence_level", width: 160, render: (value: string) => <Tag color="blue">{value || "未标注"}</Tag> },
    { title: "许可证记录", dataIndex: "license_note", width: 360, ellipsis: true, render: (value: string) => value || "未记录" },
    { title: "操作", width: 90, fixed: "right", render: (_, row) => <Typography.Link onClick={() => setSelected(row)}>详情</Typography.Link> },
  ];
  if (sources.isLoading) return <LoadingState />;
  if (sources.isError) return <ErrorState error={sources.error} />;
  return (
    <>
      <PageHeader title="数据来源与许可" subtitle="仅登记许可明确、可追溯的知识来源；公开可访问不等同于可用于临床证据。" />
      <section className="data-workbench">
        <Table className="data-workbench-table" rowKey="source_id" columns={columns} dataSource={sources.data} scroll={{ x: 1450 }} pagination={{ pageSize: 15, showSizeChanger: false }} />
      </section>
      <Drawer rootClassName="knowledge-detail-drawer" title={selected?.source_title || "数据来源"} open={Boolean(selected)} onClose={() => setSelected(null)} width={480}>
        {selected ? <><Typography.Paragraph><Typography.Text strong>source_id：</Typography.Text>{selected.source_id}</Typography.Paragraph><Typography.Paragraph><Typography.Text strong>来源类型：</Typography.Text>{selected.source_type}</Typography.Paragraph><Typography.Paragraph><Typography.Text strong>证据标记：</Typography.Text>{selected.evidence_level || "未标注"}</Typography.Paragraph><Typography.Paragraph className="pre-wrap"><Typography.Text strong>许可证记录：</Typography.Text>{selected.license_note || "未记录"}</Typography.Paragraph></> : null}
      </Drawer>
    </>
  );
}
