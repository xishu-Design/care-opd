import { Alert, Button, Collapse, Empty, Input, List, Popconfirm, Space, Tag, Typography } from "antd";
import { Brain, Plus, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import type { LongTermMemory } from "../../api/types";
import { errorMessage } from "../../utils/errors";

interface MemoryApi {
  listAIMemories: () => Promise<LongTermMemory[]>;
  rememberAI: (content: string) => Promise<unknown>;
  forgetAI: (memoryId: number) => Promise<unknown>;
}

interface LongTermMemoryPanelProps {
  api: MemoryApi;
  refreshSignal?: number;
}

export function LongTermMemoryPanel({ api, refreshSignal }: LongTermMemoryPanelProps) {
  const [items, setItems] = useState<LongTermMemory[]>([]);
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  const load = useCallback(async () => {
    try {
      setItems(await api.listAIMemories());
    } catch (cause) {
      setError(errorMessage(cause));
    }
  }, [api]);

  useEffect(() => { void load(); }, [load, refreshSignal]);

  const remember = async () => {
    const value = content.trim();
    if (!value) return;
    setLoading(true);
    setError(undefined);
    try {
      await api.rememberAI(value);
      setContent("");
      await load();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setLoading(false);
    }
  };

  const forget = async (memoryId: number) => {
    setLoading(true);
    setError(undefined);
    try {
      await api.forgetAI(memoryId);
      await load();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Collapse
      items={[{
        key: "long-term-memory",
        label: <Space><Brain size={16} /><span>AI 长期记忆</span><Tag>{items.length} 条</Tag></Space>,
        children: (
          <Space direction="vertical" size={12} style={{ width: "100%" }}>
            <Alert
              type="info"
              showIcon
              message="这里只保存稳定的回答偏好、学习目标或专业关注点。过敏史、病史、诊断、处方和剂量请写入对应结构化资料。"
            />
            {error ? <Alert type="error" showIcon message={error} /> : null}
            <Space.Compact style={{ width: "100%" }}>
              <Input
                maxLength={500}
                value={content}
                placeholder="例如：请记住以后回答时先给结论，再分点说明"
                onChange={(event) => setContent(event.target.value)}
                onPressEnter={() => void remember()}
              />
              <Button type="primary" icon={<Plus size={15} />} loading={loading} disabled={!content.trim()} onClick={() => void remember()}>记住</Button>
            </Space.Compact>
            {items.length ? (
              <List
                size="small"
                dataSource={items}
                renderItem={(item) => (
                  <List.Item
                    actions={[
                      <Popconfirm key="forget" title="忘记这条长期记忆？" okText="忘记" cancelText="取消" onConfirm={() => void forget(item.id)}>
                        <Button type="text" danger size="small" icon={<Trash2 size={14} />}>忘记</Button>
                      </Popconfirm>,
                    ]}
                  >
                    <List.Item.Meta
                      title={<Space><Tag color="green">{item.category_label}</Tag>{item.revision > 1 ? <Tag>已更新 {item.revision - 1} 次</Tag> : null}</Space>}
                      description={<Typography.Text>{item.content}</Typography.Text>}
                    />
                  </List.Item>
                )}
              />
            ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="还没有长期记忆；普通问答不会自动写入这里。" />}
          </Space>
        ),
      }]}
    />
  );
}
