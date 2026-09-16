import { NetworkDiagram, Plot, useResolvedTokens } from '@prml/viz';

const XS = [0.15, 0.45, 0.75];

export default function FactorialHmmDependence() {
  const tokens = useResolvedTokens();
  const topY = 0.2;
  const bottomY = 0.5;
  const obsY = 0.85;

  const topNodes = XS.map((x, i) => ({ id: `a${i}`, x, y: topY, label: `z(1)${i + 1}` }));
  const bottomNodes = XS.map((x, i) => ({ id: `b${i}`, x, y: bottomY, label: `z(2)${i + 1}` }));
  const obsNodes = XS.map((x, i) => ({ id: `x${i}`, x, y: obsY, label: `x${i + 1}`, observed: true }));

  const chainEdges = [0, 1].flatMap((i) => [
    { from: `a${i}`, to: `a${i + 1}`, directed: true },
    { from: `b${i}`, to: `b${i + 1}`, directed: true },
  ]);
  const emissionEdges = XS.map((_, i) => [
    { from: `a${i}`, to: `x${i}`, directed: true },
    { from: `b${i}`, to: `x${i}`, directed: true },
  ]).flat();

  // The path highlighted green in PRML fig 13.20: head-to-head at the observed nodes,
  // head-to-tail at the unobserved z(2) nodes, so it is not blocked and d-separation
  // fails between the two chains once x is observed.
  const colliderPath = [
    { from: 'b0', to: 'x0', directed: true, color: tokens.color.accent, width: 2.5 },
    { from: 'a0', to: 'x0', directed: true, color: tokens.color.accent, width: 2.5 },
  ];

  return (
    <Plot width={360} height={220} xDomain={[0, 1]} yDomain={[0, 1]} label="Two latent chains sharing observed nodes in a factorial HMM">
      <NetworkDiagram
        nodes={[...topNodes, ...bottomNodes, ...obsNodes]}
        edges={[...chainEdges, ...emissionEdges, ...colliderPath]}
        nodeRadius={12}
      />
    </Plot>
  );
}
