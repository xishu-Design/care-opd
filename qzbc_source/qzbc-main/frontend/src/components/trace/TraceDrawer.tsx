import { useQuery } from "@tanstack/react-query";
import { Descriptions, Drawer, Tabs } from "antd";

import { harnessApi } from "../../api/harnessApi";
import { useUiStore } from "../../stores/uiStore";
import { asText } from "../../utils/format";
import { ErrorState } from "../common/ErrorState";
import { LoadingState } from "../common/LoadingState";
import { ToolCallList } from "../evidence/ToolCallList";
import { TraceTimeline } from "./TraceTimeline";

export function TraceDrawer() {
  const open = useUiStore((state) => state.traceDrawerOpen);
  const traceId = useUiStore((state) => state.currentTraceId);
  const close = useUiStore((state) => state.closeTrace);
  const query = useQuery({
    queryKey: ["trace", traceId],
    queryFn: () => harnessApi.getTrace(traceId!),
    enabled: open && Boolean(traceId),
  });

  return (
    <Drawer width={760} title="链路追踪" open={open} onClose={close}>
      {query.isLoading ? <LoadingState /> : query.isError ? <ErrorState error={query.error} /> : query.data ? (
        <Tabs
          items={[
            {
              key: "summary",
              label: "概要",
              children: (
                <Descriptions size="small" column={1} bordered>
                  <Descriptions.Item label="trace_id">{query.data.trace_id}</Descriptions.Item>
                  <Descriptions.Item label="summary">{asText(query.data.summary)}</Descriptions.Item>
                  <Descriptions.Item label="pending">{asText(query.data.pending_human_actions)}</Descriptions.Item>
                  <Descriptions.Item label="llm_generations">{query.data.llm_generations?.length || 0}</Descriptions.Item>
                </Descriptions>
              ),
            },
            { key: "events", label: "节点", children: <TraceTimeline events={query.data.events} /> },
            { key: "tools", label: "工具", children: <ToolCallList items={query.data.tool_calls} /> },
            { key: "llm", label: "LLM", children: <pre className="json-box">{JSON.stringify(query.data.llm_generations || [], null, 2)}</pre> },
            { key: "packets", label: "证据包", children: <pre className="json-box">{JSON.stringify(query.data.evidence_packets, null, 2)}</pre> },
          ]}
        />
      ) : null}
    </Drawer>
  );
}
