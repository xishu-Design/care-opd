import { Spin } from "antd";

export function LoadingState({ label = "加载中" }: { label?: string }) {
  return (
    <div style={{ padding: 32, textAlign: "center" }}>
      <Spin />
      <div className="muted" style={{ marginTop: 12 }}>
        {label}
      </div>
    </div>
  );
}
