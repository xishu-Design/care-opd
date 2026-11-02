import { useQuery } from "@tanstack/react-query";
import { Alert, Button, Empty, Space, Spin, Typography } from "antd";
import cytoscape, { type Core } from "cytoscape";
import { useEffect, useMemo, useRef, useState } from "react";

import { lightragApi, type KnowledgeGraphEdge, type KnowledgeGraphNode, type KnowledgeNodeType } from "../../api/lightragApi";
import { graphLayout } from "./graphLayout";
import { NODE_COLORS, graphStyles } from "./graphStyles";
import { NODE_TYPES, RELATION_LABELS, type GraphLayoutMode, type GraphSelection } from "./graphTypes";
import { KnowledgeGraphLegend } from "./KnowledgeGraphLegend";
import { KnowledgeGraphNodeDrawer } from "./KnowledgeGraphNodeDrawer";
import { KnowledgeGraphToolbar } from "./KnowledgeGraphToolbar";

export function KnowledgeGraphView({ initialLabel = "", compact = false, initialMode = "focus" }: { initialLabel?: string; compact?: boolean; initialMode?: GraphLayoutMode }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const [query, setQuery] = useState(initialLabel);
  const [label, setLabel] = useState(initialLabel);
  const [activeTypes, setActiveTypes] = useState<KnowledgeNodeType[]>(NODE_TYPES);
  const [selection, setSelection] = useState<GraphSelection | null>(null);
  const [layoutWarning, setLayoutWarning] = useState("");
  const [mode, setMode] = useState<GraphLayoutMode>(initialMode);
  const [focusLimit, setFocusLimit] = useState(30);
  const [expandedNodeIds, setExpandedNodeIds] = useState<string[]>([]);
  const mobile = typeof window !== "undefined" && window.matchMedia("(max-width: 700px)").matches;
  const maxNodes = mobile ? 80 : compact || mode === "focus" ? 120 : 200;
  const labels = useQuery({ queryKey: ["lightrag", "labels", query], queryFn: () => lightragApi.getGraphLabels(query, 50), staleTime: 15_000 });
  const graph = useQuery({ queryKey: ["lightrag", "graph", label, maxNodes], queryFn: () => lightragApi.getGraph(label, 2, maxNodes) });
  const typeFilteredNodes = useMemo(() => (graph.data?.nodes || []).filter((node) => activeTypes.includes(node.type)), [activeTypes, graph.data]);
  const rootId = useMemo(() => {
    const exact = typeFilteredNodes.find((node) => node.label === label);
    return exact?.id || typeFilteredNodes[0]?.id;
  }, [label, typeFilteredNodes]);
  const focusNodeIds = useMemo(
    () => selectFocusNodeIds(typeFilteredNodes, graph.data?.edges || [], rootId, expandedNodeIds, mobile ? Math.min(focusLimit, 22) : focusLimit),
    [expandedNodeIds, focusLimit, graph.data?.edges, mobile, rootId, typeFilteredNodes],
  );
  const visibleNodes = useMemo(
    () => mode === "overview" ? typeFilteredNodes : typeFilteredNodes.filter((node) => focusNodeIds.has(node.id)),
    [focusNodeIds, mode, typeFilteredNodes],
  );
  const graphElements = useMemo(() => {
    const nodeIds = new Set(visibleNodes.map((node) => node.id));
    const degree = new Map<string, number>();
    const edges = (graph.data?.edges || []).filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
    edges.forEach((edge) => {
      degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
      degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
    });
    return {
      nodes: visibleNodes.map((node) => ({
        classes: [
          node.id === rootId ? "root-node" : "",
          mode === "focus" || node.id === rootId || (degree.get(node.id) || 0) >= 5 ? "labeled-node" : "",
          expandedNodeIds.includes(node.id) ? "expanded-node" : "",
        ].filter(Boolean).join(" "),
        data: {
          ...node,
          displayLabel: shortGraphLabel(node.label, node.id === rootId ? 16 : 12),
          color: NODE_COLORS[node.type],
          textColor: "#17211d",
          size: node.id === rootId && mode === "focus" ? 62 : Math.min(44, 27 + Math.min(degree.get(node.id) || 0, 4) * 4),
        },
      })),
      edges: edges.map((edge) => ({ data: { ...edge, displayRelation: RELATION_LABELS[edge.label] || "知识关联" } })),
    };
  }, [graph.data?.edges, mode, rootId, visibleNodes]);

  useEffect(() => {
    if (!containerRef.current || !graph.data || graph.data.nodes.length === 0) return;
    cyRef.current?.destroy();
    const cy = cytoscape({
      container: containerRef.current,
      elements: [...graphElements.nodes, ...graphElements.edges],
      style: graphStyles,
      layout: { name: "preset" },
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.22,
    });
    try {
      cy.layout(graphLayout(mode, rootId)).run();
      setLayoutWarning("");
    } catch (error) {
      // A graph visualization failure must not crash the prescription review.
      setLayoutWarning(error instanceof Error ? error.message : "当前浏览器无法运行知识图谱布局");
    }
    cy.on("tap", "node", (event) => {
      const selected = graph.data!.nodes.find((node) => node.id === event.target.id()) as KnowledgeGraphNode | undefined;
      if (!selected) return;
      if (mode === "focus" && selected.id !== rootId) {
        setExpandedNodeIds((current) => current.includes(selected.id) ? current : [...current, selected.id]);
        setFocusLimit((current) => Math.min(current + 12, 80));
      }
      cy.elements().addClass("faded");
      event.target.closedNeighborhood().removeClass("faded");
      event.target.addClass("focused");
      setSelection({ node: selected, relations: graph.data!.edges.filter((edge) => edge.source === selected.id || edge.target === selected.id) });
    });
    cy.on("tap", (event) => {
      if (event.target === cy) {
        cy.elements().removeClass("faded focused");
        setSelection(null);
      }
    });
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => { cy.resize(); cy.fit(undefined, 28); });
    observer?.observe(containerRef.current);
    cyRef.current = cy;
    return () => { observer?.disconnect(); cy.destroy(); cyRef.current = null; };
  }, [graph.data, graphElements, mode, rootId]);

  useEffect(() => {
    setFocusLimit(30);
    setExpandedNodeIds([]);
    setSelection(null);
  }, [label, mode]);

  const resetLayout = () => { cyRef.current?.elements().removeClass("faded focused"); cyRef.current?.layout(graphLayout(mode, rootId)).run(); setSelection(null); };
  const collapseFocus = () => { setFocusLimit(30); setExpandedNodeIds([]); setSelection(null); };
  return (
    <div className={compact ? "knowledge-graph compact" : "knowledge-graph"}>
      <KnowledgeGraphToolbar
        query={query}
        options={(labels.data?.items || []).map((item) => item.label)}
        activeTypes={activeTypes}
        mode={mode}
        onQueryChange={setQuery}
        onSearch={() => setLabel(query.trim())}
        onTypesChange={setActiveTypes}
        onModeChange={setMode}
        onFit={() => cyRef.current?.fit(undefined, 28)}
        onReset={resetLayout}
        onZoomIn={() => cyRef.current?.zoom({ level: Math.min((cyRef.current?.zoom() || 1) * 1.2, 3), renderedPosition: { x: 300, y: 220 } })}
        onZoomOut={() => cyRef.current?.zoom({ level: Math.max((cyRef.current?.zoom() || 1) / 1.2, 0.2), renderedPosition: { x: 300, y: 220 } })}
      />
      <div className="knowledge-graph-meta"><KnowledgeGraphLegend /></div>
      <div className="knowledge-graph-view-status">
        <Typography.Text type="secondary">
          {mode === "focus"
            ? `聚焦视图显示 ${visibleNodes.length}/${typeFilteredNodes.length} 个已载入节点；点击任一节点可展开它的相邻关系。`
            : `关系概览显示 ${visibleNodes.length} 个节点；标签会在缩放或选中节点时显示，避免文字重叠。`}
        </Typography.Text>
        {mode === "focus" ? (
          <Space size={8}>
            <Button size="small" disabled={visibleNodes.length >= typeFilteredNodes.length || focusLimit >= 80} onClick={() => setFocusLimit((current) => Math.min(current + 15, 80))}>展开更多</Button>
            <Button size="small" disabled={focusLimit === 30 && expandedNodeIds.length === 0} onClick={collapseFocus}>收起到中心</Button>
          </Space>
        ) : null}
      </div>
      {layoutWarning ? <Alert type="warning" showIcon message="知识图谱布局暂不可用" description={`审方结果仍可正常查看。错误：${layoutWarning}`} /> : null}
      {graph.data?.meta.degraded ? (
        <Alert
          type="warning"
          showIcon
          message="LightRAG 图查询失败，已使用本地可追溯图谱"
          description={`药材、方剂和审核关系仍可浏览。本次回退原因：${graphFallbackReason(graph.data.meta.degraded_reason)}。检索后端：${graph.data.meta.backend}`}
        />
      ) : graph.data?.meta.degraded_reason === "lightrag_no_relationships_for_query" ? (
        <Alert
          type="info"
          showIcon
          message="LightRAG 已命中词条，本次关系由本地审核图谱补充"
          description={`LightRAG 服务正常，返回 ${graph.data.meta.lightrag_raw_nodes || 0} 个节点、${graph.data.meta.lightrag_raw_edges || 0} 条关系；页面连线来自同一项目内有来源记录的药材、方剂与配伍禁忌数据。`}
        />
      ) : null}
      {graph.data?.meta.query_correction ? (
        <Alert
          type="info"
          showIcon
          message={`未收录“${label}”，已按“${graph.data.meta.query_correction}”展示`}
          description="这是输入纠正提示；图谱只展示已收录且来源可追溯的规范药材。"
        />
      ) : null}
      {graph.data?.meta.truncated ? <Alert type="info" showIcon message={`图谱数据较大，本次已载入当前词条周围的前 ${maxNodes} 个候选节点；可继续搜索具体药材或方剂查看其他关系。`} /> : null}
      <div
        className="knowledge-graph-canvas-shell"
        data-node-count={visibleNodes.length}
        data-edge-count={graph.data?.edges.length || 0}
        data-backend={graph.data?.meta.backend || ""}
        data-degraded={String(Boolean(graph.data?.meta.degraded))}
      >
        {graph.isLoading ? <div className="knowledge-graph-state"><Spin tip="正在加载知识图谱" /></div> : null}
        {graph.isError ? (
          <div className="knowledge-graph-state">
            <Alert
              type="error"
              showIcon
              message="知识图谱加载失败"
              description="请点击重试；如果持续失败，请确认后端已经重新启动。"
              action={<Button size="small" onClick={() => graph.refetch()}>重试</Button>}
            />
          </div>
        ) : null}
        {!graph.isLoading && !graph.isError && graph.data?.nodes.length === 0 ? <div className="knowledge-graph-state"><Empty description="没有找到可见且来源明确的图谱节点" /></div> : null}
        <div ref={containerRef} className="knowledge-graph-canvas" style={{ visibility: graph.data?.nodes.length ? "visible" : "hidden" }} />
      </div>
      <KnowledgeGraphNodeDrawer selection={selection} onClose={() => { cyRef.current?.elements().removeClass("faded focused"); setSelection(null); }} />
    </div>
  );
}

function shortGraphLabel(label: string, limit: number) {
  return label.length > limit ? `${label.slice(0, limit - 1)}...` : label;
}

function graphFallbackReason(reason?: string) {
  if (reason === "lightrag_results_filtered_by_provenance") return "LightRAG 返回内容未通过来源与角色可见性校验";
  if (reason === "lightrag_graph_query_failed") return "LightRAG 服务连接或图查询异常";
  return "LightRAG 没有返回可用图数据";
}

function selectFocusNodeIds(
  nodes: KnowledgeGraphNode[],
  edges: KnowledgeGraphEdge[],
  rootId: string | undefined,
  expandedNodeIds: string[],
  limit: number,
) {
  const available = new Set(nodes.map((node) => node.id));
  if (!rootId || !available.has(rootId)) return new Set(nodes.slice(0, limit).map((node) => node.id));
  const degree = new Map<string, number>();
  const adjacency = new Map<string, Array<{ id: string; priority: number }>>();
  edges.forEach((edge) => {
    if (!available.has(edge.source) || !available.has(edge.target)) return;
    degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
    const priority = relationPriority(edge.label);
    adjacency.set(edge.source, [...(adjacency.get(edge.source) || []), { id: edge.target, priority }]);
    adjacency.set(edge.target, [...(adjacency.get(edge.target) || []), { id: edge.source, priority }]);
  });
  const selected = new Set<string>([rootId]);
  const expanded = expandedNodeIds.filter((id) => available.has(id) && id !== rootId);
  expanded.forEach((id) => selected.add(id));
  const centers = [...expanded.slice().reverse(), rootId];
  for (const center of centers) {
    const neighbors = [...(adjacency.get(center) || [])].sort((a, b) => a.priority - b.priority || (degree.get(b.id) || 0) - (degree.get(a.id) || 0) || a.id.localeCompare(b.id, "zh-CN"));
    const centerBudget = center === rootId ? limit : 12;
    let added = 0;
    for (const neighbor of neighbors) {
      if (selected.size >= limit || added >= centerBudget) break;
      if (!selected.has(neighbor.id)) {
        selected.add(neighbor.id);
        added += 1;
      }
    }
  }
  return selected;
}

function relationPriority(label: string) {
  if (label.startsWith("TRADITIONAL_INCOMPATIBILITY")) return 0;
  if (label.includes("EFFICACY") || label.includes("FUNCTION")) return 1;
  if (label.includes("SYNDROME")) return 2;
  if (label.includes("SYMPTOM")) return 3;
  if (label === "CONTAINS") return 4;
  return 5;
}
