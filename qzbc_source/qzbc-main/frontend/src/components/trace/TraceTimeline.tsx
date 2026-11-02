import { Space, Tag, Timeline, Typography } from "antd";

import { StatusTag } from "../common/StatusTag";

export function TraceTimeline({ events = [] }: { events?: Array<Record<string, unknown>> }) {
  return (
    <Timeline
      items={events.map((event) => ({
        color: event.status === "failed" || event.status === "blocked" ? "red" : event.status === "interrupted" ? "orange" : "green",
        children: (
          <div>
            <Typography.Text strong>{String(event.node_name || "-")}</Typography.Text>{" "}
            <StatusTag value={String(event.event_type || event.status || "")} />
            <div className="muted">{String(event.status || "")}</div>
            <TraceMetadata event={event} />
          </div>
        ),
      }))}
    />
  );
}

function TraceMetadata({ event }: { event: Record<string, unknown> }) {
  const metadata = isRecord(event.metadata) ? event.metadata : {};
  const output = isRecord(event.output_summary) ? event.output_summary : {};
  const backend = metadata.backend || output.backend;
  const fallback = metadata.fallback_used ?? output.fallback_used;
  const filtered = metadata.filtered_count;
  if (!backend && fallback === undefined && filtered === undefined) {
    return null;
  }
  return (
    <Space size={6} wrap style={{ marginTop: 4 }}>
      {backend ? <Tag>{String(backend)}</Tag> : null}
      {fallback !== undefined ? <Tag color={fallback ? "gold" : "default"}>fallback={String(Boolean(fallback))}</Tag> : null}
      {filtered !== undefined ? <Tag color={Number(filtered) > 0 ? "gold" : "default"}>filtered={String(filtered)}</Tag> : null}
    </Space>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
