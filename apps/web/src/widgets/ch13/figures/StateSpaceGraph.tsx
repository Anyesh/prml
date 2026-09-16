import { NetworkDiagram, Plot } from '@prml/viz';

export interface StateSpaceGraphProps {
  /** Square, unshaded nodes for a discrete latent chain (HMM); circles otherwise (LDS). */
  discreteLatent?: boolean;
}

export default function StateSpaceGraph({ discreteLatent = false }: StateSpaceGraphProps) {
  const xs = [0.15, 0.4, 0.65, 0.9];
  const latentY = 0.3;
  const obsY = 0.75;

  const latentNodes = xs.map((x, i) => ({
    id: `z${i}`,
    x,
    y: latentY,
    label: `z${i + 1}`,
    shape: discreteLatent ? ('square' as const) : ('circle' as const),
  }));
  const obsNodes = xs.map((x, i) => ({ id: `x${i}`, x, y: obsY, label: `x${i + 1}`, observed: true }));

  const chainEdges = xs.slice(1).map((_, i) => ({ from: `z${i}`, to: `z${i + 1}`, directed: true }));
  const emissionEdges = xs.map((_, i) => ({ from: `z${i}`, to: `x${i}`, directed: true }));

  return (
    <Plot width={420} height={200} xDomain={[0, 1]} yDomain={[0, 1]} label="A Markov chain of latent variables, one per observation">
      <NetworkDiagram nodes={[...latentNodes, ...obsNodes]} edges={[...chainEdges, ...emissionEdges]} nodeRadius={14} />
    </Plot>
  );
}
