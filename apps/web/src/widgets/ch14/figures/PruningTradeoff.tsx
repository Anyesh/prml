import { fitRegressionTree, pruneRegressionTree } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { treeRegressionData } from '../data.js';
import '../../widgets.css';

const { X, t } = treeRegressionData(60);
const FULL_TREE = fitRegressionTree(X, t, { maxDepth: 6, minLeafSize: 2 });

function totalSse(node: ReturnType<typeof fitRegressionTree>): number {
  if (node.kind === 'leaf') return node.stat.sumTSq - (node.stat.sumT * node.stat.sumT) / node.nSamples;
  return totalSse(node.left) + totalSse(node.right);
}

function countLeaves(node: ReturnType<typeof fitRegressionTree>): number {
  return node.kind === 'leaf' ? 1 : countLeaves(node.left) + countLeaves(node.right);
}

const LAMBDAS = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3];
const POINTS = LAMBDAS.map((lambda) => {
  const pruned = pruneRegressionTree(FULL_TREE, lambda);
  return { leaves: countLeaves(pruned), sse: totalSse(pruned) };
});

export default function PruningTradeoff() {
  const tokens = useResolvedTokens();
  const maxLeaves = Math.max(...POINTS.map((p) => p.leaves));
  const maxSse = Math.max(...POINTS.map((p) => p.sse));

  return (
    <Plot height={200} xDomain={[0, maxLeaves + 1]} yDomain={[0, maxSse * 1.1]} label="Residual sum of squares against number of leaves, across pruning strengths">
      <Axes x={{ label: 'leaves |T|' }} y={{ label: 'residual SSE' }} grid />
      <Curve points={POINTS.map((p) => [p.leaves, p.sse] as const)} color={tokens.color.inkMuted} width={1.5} dash="dotted" />
      <ScatterField points={POINTS.map((p) => ({ x: p.leaves, y: p.sse, color: tokens.color.accent, size: 4 }))} />
    </Plot>
  );
}
