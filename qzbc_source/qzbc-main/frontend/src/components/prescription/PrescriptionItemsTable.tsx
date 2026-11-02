import { Table } from "antd";

import type { PrescriptionItem } from "../../api/types";

export function PrescriptionItemsTable({ items = [] }: { items?: PrescriptionItem[] }) {
  return (
    <Table
      size="small"
      rowKey={(row, index) => String(row.id || `${row.herb_name}-${index}`)}
      dataSource={items}
      pagination={false}
      columns={[
        { title: "药材", dataIndex: "herb_name" },
        { title: "剂量", dataIndex: "dose" },
        { title: "单位", dataIndex: "unit" },
        { title: "备注", dataIndex: "notes" },
      ]}
    />
  );
}
