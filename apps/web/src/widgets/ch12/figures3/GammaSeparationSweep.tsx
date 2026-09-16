import { kernelPcaFit, kernelPcaRbfKernel, kernelPcaTrainingProjections, linspace } from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { ringsData } from '../data.js';

import '../../widgets.css';

const DATA = ringsData();
const N_PER_RING = 24;
const GAMMAS = linspace(0.02, 1, 40);

function separationFraction(gamma: number): number {
  const model = kernelPcaFit(DATA, kernelPcaRbfKernel(gamma), 1);
  const proj = kernelPcaTrainingProjections(model).map((row) => row[0]!);
  const inner = proj.slice(0, N_PER_RING);
  const outer = proj.slice(N_PER_RING);
  const innerSign = inner.filter((v) => v > 0).length >= inner.length / 2 ? 1 : -1;
  const correct =
    inner.filter((v) => Math.sign(v) === innerSign).length + outer.filter((v) => Math.sign(v) !== innerSign).length;
  return correct / (inner.length + outer.length);
}

const CURVE = GAMMAS.map((g) => [g, separationFraction(g)] as const);

export default function GammaSeparationSweep() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={180} xDomain={[0, 1]} yDomain={[0.4, 1.02]} label="Fraction of points correctly separated by the sign of the first kernel component, against kernel width">
      <Axes x={{ label: 'gamma' }} y={{ label: 'fraction correctly separated' }} grid />
      <Curve points={CURVE} color={tokens.color.accent} width={2} />
      <Rule y={1} color={tokens.color.borderStrong} dash />
    </Plot>
  );
}
