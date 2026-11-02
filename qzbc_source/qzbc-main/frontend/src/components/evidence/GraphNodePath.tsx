import { Steps } from "antd";

export function GraphNodePath({ nodes = [] }: { nodes?: string[] }) {
  if (!nodes.length) return null;
  return <Steps size="small" direction="vertical" current={nodes.length - 1} items={nodes.map((title) => ({ title }))} />;
}
