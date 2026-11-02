import { Alert, Card, Descriptions, Typography } from "antd";

type ItemNote = { herb_name: string; note: string };

export function PatientInstructionCard({
  content,
  statement,
  doctorNote,
  doctorUsage,
  itemNotes = [],
}: {
  content?: string;
  statement?: string;
  doctorNote?: string;
  doctorUsage?: string;
  itemNotes?: ItemNote[];
}) {
  const hasDoctorFields = Boolean(doctorNote || doctorUsage || itemNotes.length);
  return (
    <Card title="服药说明">
      {hasDoctorFields ? (
        <Descriptions size="small" column={1} bordered>
          <Descriptions.Item label="医生确认的处方用法">{doctorUsage || "医生尚未填写，请联系医生或药师确认"}</Descriptions.Item>
          <Descriptions.Item label="医生备注原文">{doctorNote || "医生未填写备注"}</Descriptions.Item>
          {itemNotes.length ? (
            <Descriptions.Item label="单味药备注">
              {itemNotes.map((item) => <div key={`${item.herb_name}-${item.note}`}>{item.herb_name}：{item.note}</div>)}
            </Descriptions.Item>
          ) : null}
        </Descriptions>
      ) : (
        <Typography.Paragraph className="pre-wrap">{content || "暂无说明"}</Typography.Paragraph>
      )}
      {statement ? <Alert type="info" showIcon message={statement} /> : null}
    </Card>
  );
}
