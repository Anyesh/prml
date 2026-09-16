import { NetworkDiagram, Plot, useResolvedTokens } from '@prml/viz';

const LABELS = ['x1', 'x2', 'x3', 'x4'];

export default function ChainGraphs() {
  const tokens = useResolvedTokens();
  const y = 0.7;
  const xs = [0.15, 0.4, 0.65, 0.9];

  const nodes = LABELS.map((label, i) => ({ id: `x${i}`, x: xs[i]!, y, label }));
  const firstOrderEdges = xs.slice(1).map((_, i) => ({ from: `x${i}`, to: `x${i + 1}`, directed: true }));
  const secondOrderExtra = xs.slice(2).map((_, i) => ({
    from: `x${i}`,
    to: `x${i + 2}`,
    directed: true,
    color: tokens.series[1]!,
    dash: true,
  }));

  return (
    <Plot width={420} height={140} xDomain={[0, 1]} yDomain={[0, 1]} label="First- and second-order Markov chains">
      <NetworkDiagram nodes={nodes} edges={[...firstOrderEdges, ...secondOrderExtra]} nodeRadius={14} />
    </Plot>
  );
}
