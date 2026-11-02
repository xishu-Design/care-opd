import { Descriptions, Drawer, Tabs, Typography } from "antd";

import { useUiStore } from "../../stores/uiStore";
import { asText, shortId } from "../../utils/format";
import { EvidenceSourceList } from "./EvidenceSourceList";
import { GraphNodePath } from "./GraphNodePath";
import { RuleFindingList } from "./RuleFindingList";
import { ToolCallList } from "./ToolCallList";

export function EvidenceDrawer() {
  const open = useUiStore((state) => state.evidenceDrawerOpen);
  const packet = useUiStore((state) => state.currentEvidencePacket);
  const close = useUiStore((state) => state.closeEvidence);
  const graphNodes = Array.isArray(packet?.graph_node_path) ? (packet.graph_node_path as string[]) : [];

  return (
    <Drawer width={720} title="证据包" open={open} onClose={close}>
      {!packet ? null : (
        <Tabs
          items={[
            {
              key: "base",
              label: "基础",
              children: (
                <Descriptions size="small" column={1} bordered>
                  <Descriptions.Item label="packet_id">{shortId(String(packet.packet_id || ""))}</Descriptions.Item>
                  <Descriptions.Item label="trace_id">{shortId(String(packet.trace_id || ""))}</Descriptions.Item>
                  <Descriptions.Item label="visibility">{asText(packet.visibility)}</Descriptions.Item>
                  <Descriptions.Item label="human_confirm">{asText(packet.human_confirm)}</Descriptions.Item>
                  <Descriptions.Item label="backend_summary">{asText(packet.backend_summary)}</Descriptions.Item>
                </Descriptions>
              ),
            },
            {
              key: "rules",
              label: "规则",
              children: <RuleFindingList items={(packet.rule_findings as Array<Record<string, unknown>>) || []} />,
            },
            {
              key: "sources",
              label: "证据",
              children: <EvidenceSourceList items={(packet.retrieved_evidence as Array<Record<string, unknown>>) || []} />,
            },
            {
              key: "graph",
              label: "图谱",
              children: graphNodes.length ? <GraphNodePath nodes={graphNodes} /> : <Typography.Text>{asText(packet.graph_reasoning_path)}</Typography.Text>,
            },
            {
              key: "tools",
              label: "工具",
              children: <ToolCallList items={(packet.tool_calls as Array<Record<string, unknown>>) || []} />,
            },
            {
              key: "raw",
              label: "原始 JSON",
              children: <pre className="json-box">{JSON.stringify(packet, null, 2)}</pre>,
            },
          ]}
        />
      )}
    </Drawer>
  );
}
