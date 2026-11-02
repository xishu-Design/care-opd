import { Alert, Card, Descriptions, Divider, List, Space, Tag, Typography } from "antd";

import type { Prescription } from "../../api/types";

type DietData = NonNullable<Prescription["diet_precaution"]>;

export function DietPrecautionCard({ data }: { data?: DietData }) {
  const foodInteractions = data?.food_interactions || [];
  const herbInteractions = data?.herb_interactions || [];
  const llm = data?.llm;
  return (
    <Card title="饮食注意与配伍提醒">
      <Descriptions size="small" column={1} bordered>
        <Descriptions.Item label="医生饮食备注（原文）">{data?.doctor_note || "医生未填写饮食备注"}</Descriptions.Item>
        <Descriptions.Item label="AI 通俗说明">
          <Typography.Paragraph className="pre-wrap" style={{ marginBottom: 4 }}>
            {data?.ai_explanation || data?.content || "暂无扩展说明"}
          </Typography.Paragraph>
          {llm ? (
            <Space size={6} wrap>
              <Tag color={llm.provider === "ollama" && !llm.fallback_used ? "green" : "gold"}>
                {llm.provider === "ollama" && !llm.fallback_used ? "Ollama 生成" : "模型暂不可用，已用安全规则说明"}
              </Tag>
              <Typography.Text type="secondary">{llm.model}</Typography.Text>
              {llm.provider !== "ollama" || llm.fallback_used ? (
                <Typography.Text type="secondary">医生原文和数据库禁忌仍正常显示，仅 AI 通俗扩写改用固定模板。</Typography.Text>
              ) : null}
            </Space>
          ) : null}
        </Descriptions.Item>
      </Descriptions>

      <Divider orientation="left">处方药材与食物</Divider>
      {foodInteractions.length ? (
        <List
          size="small"
          dataSource={foodInteractions}
          renderItem={(item) => (
            <List.Item>
              <div>
                <Space wrap>
                  <Typography.Text strong>{item.herb_name}</Typography.Text>
                  <Typography.Text>与</Typography.Text>
                  {item.food_items.map((food) => <Tag color="orange" key={food}>{food}</Tag>)}
                  <Tag>{item.relation_label}</Tag>
                </Space>
                <Typography.Paragraph type="secondary" style={{ margin: "6px 0 0" }}>{item.message}</Typography.Paragraph>
                {item.source_url ? <a href={item.source_url} target="_blank" rel="noreferrer">查看来源：{item.source_id}</a> : null}
              </div>
            </List.Item>
          )}
        />
      ) : (
        <Alert
          type="info"
          showIcon
          message="当前可追溯资料未收录本方与具体水果、蔬菜或肉类的食忌记录"
          description="这不表示绝对没有相互作用。请优先遵循医生备注；对某种具体食物不确定时，请咨询开方医生或药师。"
        />
      )}

      <Divider orientation="left">处方药材与其他药材</Divider>
      {herbInteractions.length ? (
        <List
          size="small"
          dataSource={herbInteractions}
          renderItem={(item) => (
            <List.Item>
              <Space wrap>
                <Typography.Text strong>{item.herb_name}</Typography.Text>
                <Typography.Text>— 不宜与 —</Typography.Text>
                <Tag color="volcano">{item.incompatible_with}</Tag>
                <Tag color="volcano">{item.relation_label}</Tag>
                {item.source_url ? <a href={item.source_url} target="_blank" rel="noreferrer">来源</a> : null}
              </Space>
            </List.Item>
          )}
        />
      ) : (
        <Alert
          type="info"
          showIcon
          message="当前十八反、十九畏结构化标准未收录本方药材的相关禁配关系"
          description="未命中只代表当前标准中没有匹配记录，不代表可与所有药物任意同用。"
        />
      )}
      <Alert
        style={{ marginTop: 16 }}
        type="warning"
        showIcon
        message="请勿仅凭本页自行增减药、改变服法或停药；同时服用西药、保健品或其他中药时，请主动告知医生或药师。"
      />
    </Card>
  );
}
