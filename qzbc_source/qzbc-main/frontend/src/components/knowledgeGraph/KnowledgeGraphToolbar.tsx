import { AutoComplete, Button, Checkbox, Input, Segmented, Space, Tooltip } from "antd";
import { Focus, RotateCcw, Search, ZoomIn, ZoomOut } from "lucide-react";

import type { KnowledgeNodeType } from "../../api/lightragApi";
import { GRAPH_LAYOUT_LABELS, NODE_TYPES, NODE_TYPE_LABELS, type GraphLayoutMode } from "./graphTypes";

export function KnowledgeGraphToolbar({
  query,
  options,
  activeTypes,
  mode,
  onQueryChange,
  onSearch,
  onTypesChange,
  onModeChange,
  onFit,
  onReset,
  onZoomIn,
  onZoomOut,
}: {
  query: string;
  options: string[];
  activeTypes: KnowledgeNodeType[];
  mode: GraphLayoutMode;
  onQueryChange: (value: string) => void;
  onSearch: () => void;
  onTypesChange: (types: KnowledgeNodeType[]) => void;
  onModeChange: (mode: GraphLayoutMode) => void;
  onFit: () => void;
  onReset: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}) {
  return (
    <div className="knowledge-graph-toolbar">
      <Space.Compact style={{ minWidth: 280, flex: "1 1 340px" }}>
        <AutoComplete
          value={query}
          options={options.map((value) => ({ value }))}
          onChange={onQueryChange}
          onSelect={(value) => { onQueryChange(value); setTimeout(onSearch, 0); }}
          style={{ width: "100%" }}
        >
          <Input placeholder="搜索药材、方剂、案例或来源" onPressEnter={onSearch} allowClear />
        </AutoComplete>
        <Button type="primary" icon={<Search size={16} />} onClick={onSearch}>查询</Button>
      </Space.Compact>
      <Checkbox.Group
        options={NODE_TYPES.map((type) => ({ label: NODE_TYPE_LABELS[type], value: type }))}
        value={activeTypes}
        onChange={(values) => onTypesChange(values as KnowledgeNodeType[])}
      />
      <Segmented
        aria-label="图谱布局"
        value={mode}
        options={Object.entries(GRAPH_LAYOUT_LABELS).map(([value, label]) => ({ value, label }))}
        onChange={(value) => onModeChange(value as GraphLayoutMode)}
      />
      <Space.Compact>
        <Tooltip title="放大"><Button aria-label="放大" icon={<ZoomIn size={16} />} onClick={onZoomIn} /></Tooltip>
        <Tooltip title="缩小"><Button aria-label="缩小" icon={<ZoomOut size={16} />} onClick={onZoomOut} /></Tooltip>
        <Tooltip title="适配视图"><Button aria-label="适配视图" icon={<Focus size={16} />} onClick={onFit} /></Tooltip>
        <Tooltip title="重置布局"><Button aria-label="重置布局" icon={<RotateCcw size={16} />} onClick={onReset} /></Tooltip>
      </Space.Compact>
    </div>
  );
}
