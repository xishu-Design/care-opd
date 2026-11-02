import { Progress, Space, Typography } from "antd";

import { percent } from "../../utils/format";


export function MetricBars({ items }: { items: Array<{ label: string; value: number; color?: string }> }) {
  return (
    <Space direction="vertical" size={12} style={{ width: "100%" }}>
      {items.map((item) => (
        <div key={item.label} className="metric-bar-row">
          <div className="metric-bar-label">
            <Typography.Text>{item.label}</Typography.Text>
            <Typography.Text strong>{percent(item.value)}</Typography.Text>
          </div>
          <Progress percent={Math.round(item.value * 100)} showInfo={false} strokeColor={item.color || "#2f7d62"} trailColor="#e8eeeb" />
        </div>
      ))}
    </Space>
  );
}
