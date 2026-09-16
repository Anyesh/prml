import { useMemo, useState } from 'react';
import { type CartNode, evalGrid, fitRegressionTree, linspace, predictRegressionTree } from '@prml/math';
import { Axes, Heatmap, NetworkDiagram, Plot, ScatterField, sequentialScale, useResolvedTokens, type NetworkEdge, type NetworkNode } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { treeRegressionData } from './data.js';
import '../widgets.css';

export const title = 'Growing a regression tree one split at a time';
export const caption =
  'Step forward to add one more level of splits. The left panel is the partition of input space; the right panel is the same tree drawn as a diagram, root at the top.';
export const figure = '14.6';

const MAX_DEPTH = 4;
const DOMAIN: readonly [number, number] = [-2, 2];
const GRID_AXIS = linspace(DOMAIN[0], DOMAIN[1], 60);
const { X, t } = treeRegressionData(60);

interface Layout {
  readonly nodes: NetworkNode[];
  readonly edges: NetworkEdge[];
}

/** Standard tidy-tree layout: leaves get consecutive x positions left to right, an internal node sits above the midpoint of its children, depth increases downward. */
function layoutTree(node: CartNode<number>, depth: number, counter: { next: number }, path: string): Layout {
  if (node.kind === 'leaf') {
    const x = counter.next++;
    return {
      nodes: [{ id: path, x, y: -depth, label: node.value.toFixed(2), shape: 'circle' }],
      edges: [],
    };
  }
  const left = layoutTree(node.left, depth + 1, counter, `${path}L`);
  const right = layoutTree(node.right, depth + 1, counter, `${path}R`);
  const leftX = left.nodes.find((n) => n.id === `${path}L`)!.x;
  const rightX = right.nodes.find((n) => n.id === `${path}R`)!.x;
  const label = `x${node.featureIndex + 1} <= ${node.threshold.toFixed(2)}`;
  return {
    nodes: [
      { id: path, x: (leftX + rightX) / 2, y: -depth, label, shape: 'square' },
      ...left.nodes,
      ...right.nodes,
    ],
    edges: [{ from: path, to: `${path}L` }, { from: path, to: `${path}R` }, ...left.edges, ...right.edges],
  };
}

function countLeaves(node: CartNode<number>): number {
  return node.kind === 'leaf' ? 1 : countLeaves(node.left) + countLeaves(node.right);
}

export default function DecisionTreeGrowth() {
  const [depth, setDepth] = useState(0);
  const tokens = useResolvedTokens();

  const tree = useMemo(() => fitRegressionTree(X, t, { maxDepth: depth, minLeafSize: 3 }), [depth]);
  const grid = useMemo(() => evalGrid(GRID_AXIS, GRID_AXIS, (x1, x2) => predictRegressionTree(tree, [x1, x2])), [tree]);
  const layout = useMemo(() => layoutTree(tree, 0, { next: 0 }, 'n'), [tree]);
  const leafCount = countLeaves(tree);

  const values = grid.values.flat();
  const colorAt = sequentialScale([Math.min(...values), Math.max(...values)]);
  const treeWidth = Math.max(1, layout.nodes.filter((n) => n.shape === 'circle').length - 1);

  return (
    <div>
      <div className="widget-grid">
        <Plot width={300} height={300} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Input space partitioned into regions, coloured by the tree's prediction">
          <Heatmap data={grid} interpolator={colorAt} opacity={0.65} />
          <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
          <ScatterField points={X.map((x) => ({ x: x[0]!, y: x[1]!, color: tokens.color.ink, size: 2.5 }))} />
        </Plot>
        <Plot width={300} height={300} xDomain={[-0.5, treeWidth + 0.5]} yDomain={[-MAX_DEPTH - 0.5, 0.5]} label="The tree, root at top, leaves at bottom">
          <NetworkDiagram nodes={layout.nodes} edges={layout.edges} directed nodeRadius={20} />
        </Plot>
      </div>
      <StepThrough step={depth} stepCount={MAX_DEPTH + 1} onStep={setDepth} labels={Array.from({ length: MAX_DEPTH + 1 }, (_, d) => `depth ${d}`)} />
      <p className="widget-readout">
        Depth {depth}: {leafCount} leaves. Notice how little the partition changes once depth passes 2, the point
        past which every further split is chasing noise.
      </p>
    </div>
  );
}
