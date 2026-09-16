import { NetworkDiagram, Plot } from '@prml/viz';

export default function FactorGraphChain() {
  const zXs = [0.2, 0.5, 0.8];
  const factorXs = [0.05, 0.35, 0.65];
  const y = 0.5;

  const zNodes = zXs.map((x, i) => ({ id: `z${i}`, x, y, label: `z${i + 1}`, shape: 'circle' as const }));
  const factorNodes = factorXs.map((x, i) => ({
    id: `f${i}`,
    x,
    y,
    label: i === 0 ? 'h' : `f${i + 1}`,
    shape: 'square' as const,
  }));

  const edges = [
    { from: 'f0', to: 'z0' },
    { from: 'z0', to: 'f1' },
    { from: 'f1', to: 'z1' },
    { from: 'z1', to: 'f2' },
    { from: 'f2', to: 'z2' },
  ];

  return (
    <Plot width={360} height={140} xDomain={[0, 1]} yDomain={[0, 1]} label="Simplified factor graph for the hidden Markov model">
      <NetworkDiagram nodes={[...factorNodes, ...zNodes]} edges={edges} nodeRadius={13} />
    </Plot>
  );
}
