import { Alert, Button, Input, Popconfirm, Space, Tag, Typography } from "antd";
import { MessageSquareQuote, Send, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { TextAreaRef } from "antd/es/input/TextArea";

import { doctorApi } from "../../api/doctorApi";
import { studentApi } from "../../api/studentApi";
import type { AIChatHistoryItem, AIChatResult, GeneratedResponsePayload, LLMResultPayload, Role } from "../../api/types";
import { SelectableAnswer } from "../../components/chat/SelectableAnswer";
import { LongTermMemoryPanel } from "../../components/chat/LongTermMemoryPanel";
import { PageHeader } from "../../components/common/PageHeader";
import { errorMessage } from "../../utils/errors";

interface ChatItem {
  id?: number;
  question: string;
  answer: string;
  createdAt: string;
  traceId?: string;
  generated?: GeneratedResponsePayload;
  llm?: LLMResultPayload;
  contextTurns?: number;
  availableContextTurns?: number;
  contextMode?: string;
  followUp?: boolean;
  loading?: boolean;
  status?: string;
  stopped?: boolean;
}

interface RoleAIChatPageProps {
  role: Extract<Role, "doctor" | "student">;
}

const pageCopy = {
  doctor: {
    title: "医生专业 AI 问答",
    subtitle: "结合规范中药知识、知识图谱关系和医生端历史记忆生成专业回答，结果仅作临床决策辅助。",
    empty: "可询问中医证候、治法、方剂组成、药材功效及禁忌关系。",
    tag: "专业辅助",
    placeholder: "例如：四君子汤对应什么证候？其组成和治法依据是什么？",
  },
  student: {
    title: "中医药教学 AI 问答",
    subtitle: "按照概念解释、辨析思路和复习建议组织回答，并结合你的历史问答辅助学习。",
    empty: "可询问中医基础概念、证候辨析、方剂与药材知识。",
    tag: "教学回答",
    placeholder: "例如：请用教学方式解释脾气虚证与脾阳虚证的区别。",
  },
} as const;

function displayAnswer(data: AIChatResult) {
  const generated = data.generated_response;
  const sections = (generated?.sections || []).map((section) => [section.heading, section.content].filter(Boolean).join("\n")).filter(Boolean);
  return [generated?.summary, ...sections].filter(Boolean).join("\n\n") || data.answer || "当前没有生成可显示的回答。";
}

function historyItem(item: AIChatHistoryItem): ChatItem {
  const context = contextStatus(item.evidence_packet);
  return {
    id: item.id,
    question: item.question,
    answer: item.answer,
    createdAt: item.created_at,
    traceId: item.trace_id,
    generated: item.generated_response,
    llm: item.llm_result,
    ...context,
  };
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
  };
}

function formatTime(value: string) {
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
}

export function RoleAIChatPage({ role }: RoleAIChatPageProps) {
  const copy = pageCopy[role];
  const api = role === "doctor" ? doctorApi : studentApi;
  const [message, setMessage] = useState("");
  const [items, setItems] = useState<ChatItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [requestError, setRequestError] = useState<string>();
  const [followUpQuote, setFollowUpQuote] = useState("");
  const [deletingId, setDeletingId] = useState<number>();
  const [memoryRefresh, setMemoryRefresh] = useState(0);
  const abortController = useRef<AbortController | null>(null);
  const composerRef = useRef<TextAreaRef>(null);

  useEffect(() => {
    api.aiChatHistory()
      .then((history) => setItems(history.map(historyItem)))
      .catch(() => setRequestError("历史问答加载失败，请稍后刷新重试。"));
    return () => abortController.current?.abort();
  }, [api]);

  const updateLatest = (update: (item: ChatItem) => ChatItem) => {
    setItems((previous) => previous.map((item, index) => index === previous.length - 1 ? update(item) : item));
  };

  const sendMessage = async () => {
    const typedMessage = message.trim();
    if (!typedMessage || loading) return;
    const question = followUpQuote
      ? `针对你上一条回答中的这段内容：\n“${followUpQuote}”\n\n${typedMessage}`
      : typedMessage;
    const pendingTime = new Date().toISOString();
    setRequestError(undefined);
    setMessage("");
    setFollowUpQuote("");
    setLoading(true);
    const controller = new AbortController();
    abortController.current = controller;
    setItems((previous) => [...previous, { question, answer: "", createdAt: pendingTime, loading: true, status: "正在连接流式问答服务…" }]);
    try {
      await api.aiChatStream(question, (event) => {
        if (event.type === "status") updateLatest((item) => ({ ...item, status: event.message }));
        if (event.type === "answer") updateLatest((item) => ({ ...item, answer: item.answer + event.text, status: "正在生成回答…" }));
        if (event.type === "error") {
          setRequestError(event.message);
          updateLatest((item) => ({ ...item, answer: item.answer || event.message, loading: false, status: undefined }));
        }
        if (event.type === "complete") {
          const result = event.result;
          const context = contextStatus(result.evidence_packet);
          updateLatest((item) => ({
            ...item,
            id: result.chat_id,
            answer: item.answer || displayAnswer(result),
            createdAt: result.created_at || item.createdAt,
            traceId: result.trace_id,
            generated: result.generated_response,
            llm: result.llm_result,
            ...context,
            loading: false,
            status: undefined,
          }));
          setMemoryRefresh((value) => value + 1);
        }
      }, controller.signal);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      const text = errorMessage(error);
      setRequestError(text);
      updateLatest((item) => ({ ...item, answer: item.answer || text, loading: false, status: undefined }));
    } finally {
      if (abortController.current === controller) {
        abortController.current = null;
        setLoading(false);
      }
    }
  };

  const stopGeneration = () => {
    abortController.current?.abort();
    abortController.current = null;
    setLoading(false);
    updateLatest((item) => ({
      ...item,
      answer: item.answer || "回答已停止生成。",
      loading: false,
      status: undefined,
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
      await api.deleteAIChat(chatId);
      setItems((previous) => previous.filter((item) => item.id !== chatId));
    } catch (error) {
      setRequestError(errorMessage(error));
    } finally {
      setDeletingId(undefined);
    }
  };

  return (
    <>
      <PageHeader title={copy.title} subtitle={copy.subtitle} />
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        {requestError ? <Alert type="error" showIcon message={requestError} /> : null}
        <LongTermMemoryPanel api={api} refreshSignal={memoryRefresh} />
        <section className="patient-chat-shell" aria-label={copy.title}>
          <div className="patient-chat-messages">
            {items.length ? items.map((item, index) => (
              <div className="patient-chat-turn" key={item.id || `${item.question}-${index}`}>
                <div className="patient-message patient-message-user">
                  <Typography.Paragraph className="pre-wrap">{item.question}</Typography.Paragraph>
                  <Typography.Text className="chat-message-time">{formatTime(item.createdAt)}</Typography.Text>
                </div>
                <div className="patient-message patient-message-assistant">
                  {item.answer ? <SelectableAnswer text={item.answer} onFollowUp={prepareFollowUp} /> : null}
                  {item.status ? <Typography.Text type="secondary" className="patient-stream-status">{item.status}</Typography.Text> : null}
                  <div className="patient-message-meta">
                    <Tag color={item.stopped ? "default" : role === "doctor" ? "blue" : "green"}>{item.stopped ? "已停止" : item.loading ? "生成中" : copy.tag}</Tag>
                    {!item.loading ? <Typography.Text type="secondary">{formatTime(item.createdAt)}</Typography.Text> : null}
                    {item.llm?.provider ? <Typography.Text type="secondary">{item.llm.provider === "ollama" ? "本地 Ollama" : item.llm.provider}{item.llm.model ? ` · ${item.llm.model}` : ""}</Typography.Text> : null}
                    {item.contextTurns ? <Tag color="default">已使用 {item.contextTurns} 轮相关上下文{item.followUp ? " · 追问" : ""}</Tag> : null}
                    {!item.contextTurns && item.contextMode === "new_topic" && item.availableContextTurns ? <Tag color="blue">新话题 · 未引用旧回答</Tag> : null}
                    {item.id && !item.loading ? (
                      <Popconfirm title="删除这条问答？" description="删除后不会再作为后续上下文。" okText="删除" cancelText="取消" onConfirm={() => deleteChat(item.id!)}>
                        <Button type="text" danger size="small" loading={deletingId === item.id} icon={<Trash2 size={14} />}>删除</Button>
                      </Popconfirm>
                    ) : null}
                  </div>
                </div>
              </div>
            )) : (
              <div className="patient-chat-empty">
                <Typography.Text type="secondary">{copy.empty}</Typography.Text>
              </div>
            )}
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
              placeholder={followUpQuote ? "输入你想针对这段内容追问的问题…" : copy.placeholder}
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              onPressEnter={(event) => {
                if (!event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
            />
            {loading ? (
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
