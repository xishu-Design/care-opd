import { Alert, Button, Input, Popconfirm, Space, Tag, Typography } from "antd";
import { MessageSquareQuote, Send, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { TextAreaRef } from "antd/es/input/TextArea";

import { patientApi } from "../../api/patientApi";
import type { GeneratedResponsePayload, LLMResultPayload } from "../../api/types";
import { SelectableAnswer } from "../../components/chat/SelectableAnswer";
import { LongTermMemoryPanel } from "../../components/chat/LongTermMemoryPanel";
import { PageHeader } from "../../components/common/PageHeader";
import { errorMessage } from "../../utils/errors";

interface ChatItem {
  id?: number;
  question: string;
  answer: string;
  createdAt: string;
  allowed?: boolean;
  traceId?: string;
  evidence?: Record<string, unknown>;
  generated?: GeneratedResponsePayload;
  llm?: LLMResultPayload;
  status?: string;
  streaming?: boolean;
  stopped?: boolean;
  contextTurns?: number;
  availableContextTurns?: number;
  contextMode?: string;
  followUp?: boolean;
  businessContextTypes?: string[];
}

function displayAnswer(data: { safe_reply?: string; review_summary?: string; generated_response?: GeneratedResponsePayload }) {
  const generated = data.generated_response;
  const sections = (generated?.sections || []).map((section) => [section.heading, section.content].filter(Boolean).join("\n")).filter(Boolean);
  return [generated?.summary, ...sections].filter(Boolean).join("\n\n") || data.safe_reply || data.review_summary || "已收到。";
}

function contextStatus(evidence?: Record<string, unknown>) {
  const summary = evidence?.backend_summary;
  if (!summary || typeof summary !== "object") return {};
  const values = summary as Record<string, unknown>;
  const count = Number(values.recent_turn_count || 0);
  const available = Number(values.available_recent_turn_count || 0);
  return {
    contextTurns: Number.isFinite(count) ? count : 0,
    availableContextTurns: Number.isFinite(available) ? available : 0,
    contextMode: String(values.context_mode || ""),
    followUp: values.follow_up_detected === true,
    businessContextTypes: Array.isArray(values.business_context_types) ? values.business_context_types.map(String) : [],
  };
}

export function PatientEducationChatPage() {
  const [message, setMessage] = useState("");
  const [items, setItems] = useState<ChatItem[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [requestError, setRequestError] = useState<string>();
  const [followUpQuote, setFollowUpQuote] = useState("");
  const [deletingId, setDeletingId] = useState<number>();
  const [memoryRefresh, setMemoryRefresh] = useState(0);
  const abortController = useRef<AbortController | null>(null);
  const composerRef = useRef<TextAreaRef>(null);

  useEffect(() => {
    patientApi.educationHistory()
      .then((history) => setItems(history.map((item) => ({
        id: item.id,
        question: item.question,
        answer: item.answer,
        createdAt: item.created_at,
        allowed: item.allowed,
        traceId: item.trace_id,
        evidence: item.evidence_packet,
        generated: item.generated_response,
        llm: item.llm_result,
        ...contextStatus(item.evidence_packet),
      }))))
      .catch(() => setRequestError("历史问答加载失败，请稍后刷新重试。"));
    return () => abortController.current?.abort();
  }, []);

  const updateLatest = (update: (item: ChatItem) => ChatItem) => {
    setItems((previous) => previous.map((item, index) => index === previous.length - 1 ? update(item) : item));
  };

  const sendMessage = async () => {
    const typedMessage = message.trim();
    if (!typedMessage || isStreaming) return;
    const question = followUpQuote
      ? `针对你上一条回答中的这段内容：\n“${followUpQuote}”\n\n${typedMessage}`
      : typedMessage;
    setRequestError(undefined);
    setMessage("");
    setFollowUpQuote("");
    setIsStreaming(true);
    const controller = new AbortController();
    abortController.current = controller;
    setItems((previous) => [...previous, { question, answer: "", createdAt: new Date().toISOString(), status: "正在连接安全科普服务…", streaming: true }]);
    try {
      await patientApi.educationChatStream(question, (event) => {
        if (event.type === "status") {
          updateLatest((item) => ({ ...item, status: event.message }));
        }
        if (event.type === "answer") {
          updateLatest((item) => ({ ...item, answer: item.answer + event.text, status: "正在生成回答…" }));
        }
        if (event.type === "error") {
          setRequestError(event.message);
          updateLatest((item) => ({ ...item, answer: item.answer || event.message, allowed: false, status: undefined, streaming: false }));
        }
        if (event.type === "complete") {
          const data = event.result;
          updateLatest((item) => ({
            ...item,
            id: data.chat_id,
            answer: item.answer || displayAnswer(data),
            createdAt: data.created_at || item.createdAt,
            allowed: data.allowed,
            traceId: data.trace_id,
            evidence: data.evidence_packet,
            generated: data.generated_response,
            llm: data.llm_result,
            ...contextStatus(data.evidence_packet),
            status: undefined,
            streaming: false,
          }));
          setMemoryRefresh((value) => value + 1);
        }
      }, controller.signal);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      const message = errorMessage(error);
      setRequestError(message);
      updateLatest((item) => ({ ...item, answer: item.answer || message, allowed: false, status: undefined, streaming: false }));
    } finally {
      if (abortController.current === controller) {
        abortController.current = null;
        setIsStreaming(false);
      }
    }
  };

  const stopGeneration = () => {
    abortController.current?.abort();
    abortController.current = null;
    setIsStreaming(false);
    updateLatest((item) => ({
      ...item,
      answer: item.answer || "回答已停止生成。",
      status: undefined,
      streaming: false,
      stopped: true,
    }));
  };

  const prepareFollowUp = (quote: string) => {
    setFollowUpQuote(quote);
    window.setTimeout(() => composerRef.current?.focus(), 0);
  };

  const deleteChat = async (chatId: number) => {
    setDeletingId(chatId);
    setRequestError(undefined);
    try {
      await patientApi.deleteEducationChat(chatId);
      setItems((previous) => previous.filter((item) => item.id !== chatId));
    } catch (error) {
      setRequestError(errorMessage(error));
    } finally {
      setDeletingId(undefined);
    }
  };

  const formatTime = (value: string) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return new Intl.DateTimeFormat("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(date);
  };

  return (
    <>
      <PageHeader title="中医药科普问答" subtitle="诊断、开方、改药、加量和停药由后端 Safety Guard 拦截。" />
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        {requestError ? <Alert type="error" showIcon message={requestError} /> : null}
        <LongTermMemoryPanel api={patientApi} refreshSignal={memoryRefresh} />
        <section className="patient-chat-shell" aria-label="中医药科普对话">
          <div className="patient-chat-messages">
            {items.length ? items.map((item, index) => (
              <div className="patient-chat-turn" key={item.id || `${item.question}-${index}`}>
                <div className="patient-message patient-message-user">
                  <Typography.Text>{item.question}</Typography.Text>
                  <Typography.Text className="chat-message-time">{formatTime(item.createdAt)}</Typography.Text>
                </div>
                <div className={`patient-message patient-message-assistant${item.allowed === false ? " patient-message-warning" : ""}`}>
                  {item.answer ? <SelectableAnswer text={item.answer} onFollowUp={prepareFollowUp} /> : null}
                  {item.status ? <Typography.Text type="secondary" className="patient-stream-status">{item.status}</Typography.Text> : null}
                  <div className="patient-message-meta">
                    <Tag color={item.stopped ? "default" : item.allowed === false ? "gold" : "green"}>{item.stopped ? "已停止" : item.allowed === false ? "安全提示" : item.streaming ? "生成中" : "科普回答"}</Tag>
                    {!item.streaming ? <Typography.Text type="secondary">{formatTime(item.createdAt)}</Typography.Text> : null}
                    {item.llm?.provider ? <Typography.Text type="secondary">{item.llm.provider === "ollama" ? "本地 Ollama" : item.llm.provider}{item.llm.model ? ` · ${item.llm.model}` : ""}</Typography.Text> : null}
                    {item.llm?.grounding?.missing_evidence === true && item.llm?.provider !== "patient_safety_policy"
                      ? <Tag color="red">无可追溯证据 · 不再作为上下文依据</Tag>
                      : null}
                    {item.businessContextTypes?.includes("published_prescriptions") ? <Tag color="blue">已关联本人当前处方</Tag> : null}
                    {item.businessContextTypes?.includes("patient_self") ? <Tag color="cyan">已关联本人健康档案</Tag> : null}
                    {item.contextTurns ? <Tag color="default">已使用 {item.contextTurns} 轮相关上下文{item.followUp ? " · 追问" : ""}</Tag> : null}
                    {!item.contextTurns && item.contextMode === "new_topic" && item.availableContextTurns ? <Tag color="blue">新话题 · 未引用旧回答</Tag> : null}
                    {item.id && !item.streaming ? (
                      <Popconfirm title="删除这条问答？" description="删除后不会再作为后续上下文。" okText="删除" cancelText="取消" onConfirm={() => deleteChat(item.id!)}>
                        <Button type="text" danger size="small" loading={deletingId === item.id} icon={<Trash2 size={14} />}>删除</Button>
                      </Popconfirm>
                    ) : null}
                  </div>
                </div>
              </div>
            )) : <div className="patient-chat-empty"><Typography.Text type="secondary">输入一个中医药科普问题，回答将经过安全检查。</Typography.Text></div>}
          </div>
          <div className="patient-chat-composer">
          {followUpQuote ? (
            <div className="follow-up-quote">
              <div><MessageSquareQuote size={15} /> <span>追问选中内容</span></div>
              <Typography.Text ellipsis={{ tooltip: followUpQuote }}>“{followUpQuote}”</Typography.Text>
              <Button type="text" size="small" onClick={() => setFollowUpQuote("")}>取消</Button>
            </div>
          ) : null}
          <Input.TextArea
            ref={composerRef}
            rows={3}
            maxLength={1500}
            showCount
            placeholder={followUpQuote ? "输入你想针对这段内容追问的问题…" : "请输入中医药科普问题"}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onPressEnter={(event) => {
              if (!event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
          />
          {isStreaming ? (
            <Button danger icon={<Square size={15} fill="currentColor" />} onClick={stopGeneration} style={{ marginTop: 12 }}>停止生成</Button>
          ) : (
            <Button type="primary" icon={<Send size={16} />} disabled={!message.trim()} onClick={sendMessage} style={{ marginTop: 12 }}>发送</Button>
          )}
          </div>
        </section>
      </Space>
    </>
  );
}
