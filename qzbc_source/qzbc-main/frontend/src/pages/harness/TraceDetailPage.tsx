import { useQuery } from "@tanstack/react-query";
import { Button, Card, Descriptions, Input, List, Space, Typography } from "antd";
import { Search } from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { harnessApi } from "../../api/harnessApi";
import { ErrorState } from "../../components/common/ErrorState";
import { LoadingState } from "../../components/common/LoadingState";
import { PageHeader } from "../../components/common/PageHeader";
import { StatusTag } from "../../components/common/StatusTag";
import { ToolCallList } from "../../components/evidence/ToolCallList";
import { TraceTimeline } from "../../components/trace/TraceTimeline";
import { asText, shortId } from "../../utils/format";

export function TraceDetailPage() {
  const navigate = useNavigate();
  const params = useParams();
  const [input, setInput] = useState(params.traceId || "");
  const list = useQuery({ queryKey: ["harness", "traces"], queryFn: harnessApi.listTraces });
  const detail = useQuery({
    queryKey: ["harness", "trace", params.traceId],
    queryFn: () => harnessApi.getTrace(params.traceId!),
    enabled: Boolean(params.traceId),
  });

  return (
    <>
      <PageHeader title="链路追踪" subtitle="智能体链路、MCP 工具调用、证据包和人工确认事件。" />
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        <Card>
          <Space.Compact style={{ width: "100%" }}>
            <Input value={input} onChange={(event) => setInput(event.target.value)} placeholder="trace_id" />
            <Button type="primary" icon={<Search size={16} />} onClick={() => input && navigate(`/harness/traces/${input}`)}>
              查看
            </Button>
          </Space.Compact>
        </Card>
        {detail.isLoading ? <LoadingState /> : detail.isError ? <ErrorState error={detail.error} /> : detail.data ? (
          <div className="two-column">
            <Space direction="vertical" size={16}>
              <Card title="概要">
                <Descriptions column={1} size="small" bordered>
                  <Descriptions.Item label="trace_id">{detail.data.trace_id}</Descriptions.Item>
                  <Descriptions.Item label="summary">{asText(detail.data.summary)}</Descriptions.Item>
                  <Descriptions.Item label="pending">{asText(detail.data.pending_human_actions)}</Descriptions.Item>
                  <Descriptions.Item label="llm_generations">{detail.data.llm_generations?.length || 0}</Descriptions.Item>
                </Descriptions>
              </Card>
              <Card title="工具调用"><ToolCallList items={detail.data.tool_calls} /></Card>
            </Space>
            <Space direction="vertical" size={16}>
              <Card title="节点时间线"><TraceTimeline events={detail.data.events} /></Card>
              <Card title="LLM 生成记录">
                <Descriptions column={1} size="small" bordered>
                  {(detail.data.llm_generations || []).map((item, index) => (
                    <Descriptions.Item key={`${String(item.request_id || index)}`} label={String(item.provider || `LLM ${index + 1}`)}>
                      {asText({
                        model: item.model,
                        task_type: item.task_type,
                        fallback_used: item.fallback_used,
                        degraded: item.degraded,
                        source_ids: item.source_ids,
                        safety_flags: item.safety_flags,
                        latency_ms: item.latency_ms,
                      })}
                    </Descriptions.Item>
                  ))}
                  {detail.data.llm_generations?.length ? null : <Descriptions.Item label="empty">-</Descriptions.Item>}
                </Descriptions>
              </Card>
              <Card title="证据包"><pre className="json-box">{JSON.stringify(detail.data.evidence_packets, null, 2)}</pre></Card>
            </Space>
          </div>
        ) : null}
        <Card title="最近链路">
          {list.isLoading ? <LoadingState /> : list.isError ? <ErrorState error={list.error} /> : (
            <List
              dataSource={list.data?.items || []}
              renderItem={(item) => (
                <List.Item actions={[<Button key="open" onClick={() => navigate(`/harness/traces/${item.trace_id}`)}>打开</Button>]}>
                  <Space direction="vertical" size={4}>
                    <Typography.Text strong>{shortId(item.trace_id)}</Typography.Text>
                    <Space wrap>
                      <StatusTag value={item.status} />
                      <Typography.Text type="secondary">{item.role} / {item.task_type}</Typography.Text>
                    </Space>
                  </Space>
                </List.Item>
              )}
            />
          )}
        </Card>
      </Space>
    </>
  );
}
