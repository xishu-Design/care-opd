import { Tag } from "antd";

const statusColors: Record<string, string> = {
  draft: "default",
  reviewing: "gold",
  doctor_confirmed: "green",
  published_to_patient: "blue",
  deidentified_for_learning: "purple",
  pending: "gold",
  approved: "green",
  rejected: "red",
  checkpoint_lost: "volcano",
  success: "green",
  blocked: "red",
  interrupted: "gold",
  resumed: "blue",
  failed: "red",
};

const riskColors: Record<string, string> = {
  red: "red",
  orange: "orange",
  yellow: "gold",
  green: "green",
};

export function StatusTag({ value, kind = "status" }: { value?: string | null; kind?: "status" | "risk" | "role" }) {
  if (!value) return <Tag>-</Tag>;
  const color = kind === "risk" ? riskColors[value] || "default" : kind === "role" ? "cyan" : statusColors[value] || "default";
  return <Tag color={color}>{value}</Tag>;
}
