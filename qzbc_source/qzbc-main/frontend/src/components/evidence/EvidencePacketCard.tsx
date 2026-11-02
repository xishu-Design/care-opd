import { Button, Card, Descriptions, Space, Tag } from "antd";
import { Network, Search } from "lucide-react";

import type { EvidencePacket } from "../../api/types";
import { useUiStore } from "../../stores/uiStore";
import { shortId } from "../../utils/format";

export function EvidencePacketCard({ packet, traceId }: { packet?: EvidencePacket | null; traceId?: string | null }) {
  const openEvidence = useUiStore((state) => state.openEvidence);
  const openTrace = useUiStore((state) => state.openTrace);
  if (!packet) return null;
  return (
    <Card
      title="证据包"
      extra={
        <Space>
          <Button icon={<Search size={16} />} onClick={() => openEvidence(packet)}>
            证据链
          </Button>
          {traceId ? (
            <Button icon={<Network size={16} />} onClick={() => openTrace(traceId)}>
              Trace
            </Button>
          ) : null}
        </Space>
      }
    >
      <Descriptions size="small" column={2}>
        <Descriptions.Item label="packet_id">{shortId(packet.packet_id)}</Descriptions.Item>
        <Descriptions.Item label="visibility">
          <Tag color={packet.visibility === "doctor_internal" ? "gold" : packet.visibility === "patient_safe" ? "green" : "blue"}>
            {packet.visibility || "-"}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="degraded">{String(Boolean(packet.degraded))}</Descriptions.Item>
        <Descriptions.Item label="sources">{packet.retrieved_evidence?.length || 0}</Descriptions.Item>
        <Descriptions.Item label="rules">{packet.rule_findings?.length || 0}</Descriptions.Item>
        <Descriptions.Item label="tools">{packet.tool_calls?.length || 0}</Descriptions.Item>
      </Descriptions>
    </Card>
  );
}
