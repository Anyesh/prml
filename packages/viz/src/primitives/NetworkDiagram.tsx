import type { KeyboardEvent as ReactKeyboardEvent } from 'react';

import { useFrame } from '../Plot.js';
import { useResolvedTokens } from '../hooks.js';

export type NetworkNodeShape = 'circle' | 'square';

export interface NetworkNode {
  readonly id: string;
  /** Data coordinates, projected through the enclosing Plot's frame like every other primitive. */
  readonly x: number;
  readonly y: number;
  /** Drawn inside the node. Keep it to a few characters; nothing is wrapped or shrunk to fit. */
  readonly label?: string;
  /** Square is the factor-graph convention for a factor node; circle is a variable. */
  readonly shape?: NetworkNodeShape;
  /** Shaded fill, the book's convention for a variable whose value has been observed. */
  readonly observed?: boolean;
  readonly color?: string;
  /** Radius in pixels, or half-side for a square. */
  readonly radius?: number;
  readonly dimmed?: boolean;
}

export interface NetworkEdge {
  readonly from: string;
  readonly to: string;
  /** Overrides the diagram-wide `directed`. An arrowhead is drawn at the `to` end. */
  readonly directed?: boolean;
  readonly label?: string;
  readonly color?: string;
  readonly width?: number;
  readonly dash?: boolean;
  readonly dimmed?: boolean;
}

export interface NetworkDiagramProps {
  nodes: readonly NetworkNode[];
  edges: readonly NetworkEdge[];
  /** Default for edges that do not set their own. */
  directed?: boolean;
  nodeRadius?: number;
  onSelect?: (id: string) => void;
  onHover?: (id: string | null) => void;
  /** Accessible name for a node; defaults to its label, then its id. */
  nodeLabel?: (node: NetworkNode) => string;
}

const DEFAULT_RADIUS = 16;
const DEFAULT_ARROW_SIZE = 8;
const DIMMED_OPACITY = 0.3;

/** Distance in pixels from a node's centre to its boundary along the unit direction (ux, uy). */
export function boundaryOffset(shape: NetworkNodeShape, radius: number, ux: number, uy: number): number {
  if (shape === 'circle') return radius;
  const spread = Math.max(Math.abs(ux), Math.abs(uy));
  if (spread === 0) return radius;
  return radius / spread;
}

export interface EdgeGeometry {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  /** Three points of the arrowhead triangle, or null when the edge is undirected. */
  readonly arrow: readonly (readonly [number, number])[] | null;
}

interface EdgeEndpoint {
  px: number;
  py: number;
  shape: NetworkNodeShape;
  radius: number;
}

/** Pixel endpoints trimmed to both node boundaries, so a line never runs under a node's fill. */
export function edgeGeometry(
  from: EdgeEndpoint,
  to: EdgeEndpoint,
  directed: boolean,
  arrowSize: number = DEFAULT_ARROW_SIZE,
): EdgeGeometry {
  const dx = to.px - from.px;
  const dy = to.py - from.py;
  const dist = Math.hypot(dx, dy);
  const [ux, uy] = dist === 0 ? [0, 0] : [dx / dist, dy / dist];

  const fromOffset = boundaryOffset(from.shape, from.radius, ux, uy);
  const toOffset = boundaryOffset(to.shape, to.radius, -ux, -uy);

  const x1 = from.px + ux * fromOffset;
  const y1 = from.py + uy * fromOffset;
  const x2 = to.px - ux * toOffset;
  const y2 = to.py - uy * toOffset;

  if (!directed) {
    return { x1, y1, x2, y2, arrow: null };
  }

  const backX = x2 - ux * arrowSize;
  const backY = y2 - uy * arrowSize;
  const perpX = -uy;
  const perpY = ux;
  const halfWidth = arrowSize / 2;

  return {
    x1,
    y1,
    x2,
    y2,
    arrow: [
      [x2, y2],
      [backX + perpX * halfWidth, backY + perpY * halfWidth],
      [backX - perpX * halfWidth, backY - perpY * halfWidth],
    ],
  };
}

export interface LayeredNode {
  readonly id: string;
  readonly layer: number;
  readonly index: number;
  readonly x: number;
  readonly y: number;
}

export interface LayeredLayoutBox {
  readonly x0: number;
  readonly x1: number;
  readonly y0: number;
  readonly y1: number;
}

const DEFAULT_LAYOUT_BOX: LayeredLayoutBox = { x0: 0, x1: 1, y0: 0, y1: 1 };

function evenlySpread(index: number, count: number, start: number, end: number): number {
  if (count <= 1) return (start + end) / 2;
  return start + (index / (count - 1)) * (end - start);
}

/**
 * Positions for a feed-forward network: `counts` is units per layer. Layers are spread evenly
 * across the x box, and within a layer nodes are spread evenly down the y box and centred, so a
 * layer of one unit sits on the box's vertical midline.
 */
export function layeredLayout(counts: readonly number[], box: LayeredLayoutBox = DEFAULT_LAYOUT_BOX): LayeredNode[] {
  const nodes: LayeredNode[] = [];
  for (let layer = 0; layer < counts.length; layer++) {
    const x = evenlySpread(layer, counts.length, box.x0, box.x1);
    const count = counts[layer] ?? 0;
    for (let index = 0; index < count; index++) {
      const y = evenlySpread(index, count, box.y0, box.y1);
      nodes.push({ id: `L${layer}N${index}`, layer, index, x, y });
    }
  }
  return nodes;
}

interface ProjectedNode {
  readonly node: NetworkNode;
  readonly px: number;
  readonly py: number;
  readonly shape: NetworkNodeShape;
  readonly radius: number;
}

interface NetworkNodeGlyphProps {
  projected: ProjectedNode;
  fill: string;
  stroke: string;
  labelColor: string;
  accessibleName: string;
  onSelect?: ((id: string) => void) | undefined;
  onHover?: ((id: string | null) => void) | undefined;
}

function NetworkNodeGlyph({ projected, fill, stroke, labelColor, accessibleName, onSelect, onHover }: NetworkNodeGlyphProps) {
  const { node, px, py, shape, radius } = projected;
  const interactive = Boolean(onSelect);

  function handleKeyDown(event: ReactKeyboardEvent<SVGGElement>) {
    if (!onSelect) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onSelect(node.id);
    }
  }

  return (
    <g
      transform={`translate(${px}, ${py})`}
      role={onSelect ? 'button' : 'img'}
      aria-label={accessibleName}
      tabIndex={interactive ? 0 : -1}
      opacity={node.dimmed ? DIMMED_OPACITY : 1}
      onKeyDown={interactive ? handleKeyDown : undefined}
      onClick={onSelect ? () => onSelect(node.id) : undefined}
      onPointerEnter={onHover ? () => onHover(node.id) : undefined}
      onPointerLeave={onHover ? () => onHover(null) : undefined}
    >
      {shape === 'square' ? (
        <rect x={-radius} y={-radius} width={radius * 2} height={radius * 2} fill={fill} stroke={stroke} strokeWidth={1.5} />
      ) : (
        <circle r={radius} fill={fill} stroke={stroke} strokeWidth={1.5} />
      )}
      {node.label ? (
        <text
          textAnchor="middle"
          dominantBaseline="middle"
          fill={labelColor}
          style={{ font: 'var(--prml-text-xs) var(--prml-font-sans)', pointerEvents: 'none' }}
        >
          {node.label}
        </text>
      ) : null}
    </g>
  );
}

interface NetworkEdgeGlyphProps {
  edge: NetworkEdge;
  geometry: EdgeGeometry;
  stroke: string;
  labelColor: string;
  plateColor: string;
}

function NetworkEdgeGlyph({ edge, geometry, stroke, labelColor, plateColor }: NetworkEdgeGlyphProps) {
  return (
    <g opacity={edge.dimmed ? DIMMED_OPACITY : 1}>
      <line
        x1={geometry.x1}
        y1={geometry.y1}
        x2={geometry.x2}
        y2={geometry.y2}
        stroke={stroke}
        strokeWidth={edge.width ?? 1.5}
        strokeDasharray={edge.dash ? '4 3' : undefined}
      />
      {geometry.arrow ? <polygon points={geometry.arrow.map(([x, y]) => `${x},${y}`).join(' ')} fill={stroke} /> : null}
      {edge.label ? (
        <g
          transform={`translate(${(geometry.x1 + geometry.x2) / 2}, ${(geometry.y1 + geometry.y2) / 2})`}
          style={{ pointerEvents: 'none' }}
        >
          {/* A message label sits on the edge it belongs to, so it must knock out the stroke
              beneath it or the line reads straight through the glyphs. */}
          <text
            textAnchor="middle"
            dominantBaseline="middle"
            stroke={plateColor}
            strokeWidth={3.5}
            strokeLinejoin="round"
            fill="none"
            style={{ font: 'var(--prml-text-xs) var(--prml-font-sans)' }}
          >
            {edge.label}
          </text>
          <text
            textAnchor="middle"
            dominantBaseline="middle"
            fill={labelColor}
            style={{ font: 'var(--prml-text-xs) var(--prml-font-sans)' }}
          >
            {edge.label}
          </text>
        </g>
      ) : null}
    </g>
  );
}

/**
 * Node-link diagrams for directed Bayesian networks, undirected Markov random fields, and
 * bipartite factor graphs, plus the directed chains and lattices of chapter 13. Positions are
 * explicit data coordinates rather than a layout this primitive computes, because the shape a
 * graphical model needs to convey (which nodes are plates, which are conditioned on which) is
 * the reader's argument, not something a generic layout could recover from the edge list alone.
 *
 * SVG rather than canvas for the same reason as `ScatterField`: nodes are what a reader clicks,
 * and their labels are small text that has to hit the pixel grid.
 */
export function NetworkDiagram({
  nodes,
  edges,
  directed = false,
  nodeRadius = DEFAULT_RADIUS,
  onSelect,
  onHover,
  nodeLabel,
}: NetworkDiagramProps) {
  const frame = useFrame();
  const tokens = useResolvedTokens();

  const projected = new Map<string, ProjectedNode>();
  for (const node of nodes) {
    // Two nodes sharing an id would both draw at the last one's position, which is a wrong
    // picture rather than a missing one, so it must not pass silently.
    if (projected.has(node.id)) {
      throw new Error(`@prml/viz: NetworkDiagram has two nodes with id "${node.id}"`);
    }
    const [px, py] = frame.toPx(node.x, node.y);
    projected.set(node.id, {
      node,
      px,
      py,
      shape: node.shape ?? 'circle',
      radius: node.radius ?? nodeRadius,
    });
  }

  return (
    <g>
      <g>
        {edges.map((edge, index) => {
          const from = projected.get(edge.from);
          if (!from) throw new Error(`@prml/viz: NetworkDiagram edge references unknown node "${edge.from}"`);
          const to = projected.get(edge.to);
          if (!to) throw new Error(`@prml/viz: NetworkDiagram edge references unknown node "${edge.to}"`);

          const geometry = edgeGeometry(from, to, edge.directed ?? directed);

          return (
            <NetworkEdgeGlyph
              key={`${edge.from}-${edge.to}-${index}`}
              edge={edge}
              geometry={geometry}
              stroke={edge.color ?? tokens.color.inkMuted}
              labelColor={tokens.color.ink}
              plateColor={tokens.color.plotBg}
            />
          );
        })}
      </g>
      <g>
        {nodes.map((node) => {
          const glyph = projected.get(node.id)!;
          const accessibleName = nodeLabel ? nodeLabel(node) : (node.label ?? node.id);
          return (
            <NetworkNodeGlyph
              key={node.id}
              projected={glyph}
              fill={node.observed ? tokens.color.inkFaint : tokens.color.plotBg}
              stroke={node.color ?? tokens.color.borderStrong}
              labelColor={tokens.color.ink}
              accessibleName={accessibleName}
              onSelect={onSelect}
              onHover={onHover}
            />
          );
        })}
      </g>
    </g>
  );
}
