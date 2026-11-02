import type { LayoutOptions } from "cytoscape";

import type { GraphLayoutMode } from "./graphTypes";

export function graphLayout(mode: GraphLayoutMode, rootId?: string): LayoutOptions {
  if (mode === "focus") {
    return {
      name: "breadthfirst",
      animate: false,
      fit: true,
      padding: 88,
      directed: false,
      circle: true,
      spacingFactor: 1.65,
      avoidOverlap: true,
      roots: rootId ? [rootId] : undefined,
    };
  }
  return {
    name: "cose",
    animate: false,
    fit: true,
    padding: 58,
    nodeRepulsion: () => 18000,
    nodeOverlap: 36,
    idealEdgeLength: () => 150,
    edgeElasticity: () => 90,
    nestingFactor: 1.2,
    gravity: 0.12,
    numIter: 1600,
    componentSpacing: 120,
    randomize: true,
  };
}
