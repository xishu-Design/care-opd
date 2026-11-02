import { Button, Form, Input, Space } from "antd";
import { MinusCircle, Plus } from "lucide-react";

export interface ReviewAnswerValues {
  answer_text: string;
  proposed_herbs: string[];
  has_risk: boolean;
  risk_level?: string;
}

export function ReviewAnswerForm({ loading, onSubmit, mode = "review" }: { loading?: boolean; onSubmit: (values: ReviewAnswerValues) => void; mode?: "review" | "composition" }) {
  const isComposition = mode === "composition";
  return (
    <Form layout="vertical" initialValues={{ proposed_herbs: [""] }} onFinish={(values) => onSubmit({ answer_text: String(values.answer_text || ""), has_risk: false, risk_level: undefined, proposed_herbs: (values.proposed_herbs || []).map((item: string) => item.trim()).filter(Boolean) })}>
      <Form.List name="proposed_herbs">
        {(fields, { add, remove }) => <Form.Item label={isComposition ? "你识别出的药材组成（仅填写药材名称）" : "学生候选方剂（仅填写药材名称）"} required>
          <Space direction="vertical" style={{ width: "100%" }}>
            {fields.map((field) => <Space.Compact key={field.key}><Form.Item {...field} noStyle rules={[{ required: true, message: "请填写药材名称" }]}><Input placeholder="药材名称" /></Form.Item><Button aria-label="移除药材" icon={<MinusCircle size={16} />} onClick={() => remove(field.name)} /></Space.Compact>)}
            <Button icon={<Plus size={16} />} onClick={() => add("")}>添加药材</Button>
          </Space>
        </Form.Item>}
      </Form.List>
      <Form.Item name="answer_text" label={isComposition ? "组成识别理由" : "拟方理由"} rules={[{ required: true, message: "请简要说明拟方理由" }]}>
        <Input.TextArea rows={5} />
      </Form.Item>
      <Button type="primary" htmlType="submit" loading={loading}>{isComposition ? "提交组成对照" : "提交答案"}</Button>
    </Form>
  );
}
