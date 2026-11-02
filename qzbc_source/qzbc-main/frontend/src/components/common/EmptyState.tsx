import { Empty } from "antd";

export function EmptyState({ label = "暂无数据", description }: { label?: string; description?: string }) {
  return <Empty description={description ? <span>{label}<br />{description}</span> : label} />;
}
