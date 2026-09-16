import { pcg32, standardNormal, univariateGaussianVbFit } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const RNG = pcg32(20261001);
const TRUE_MU = 1.4;
const TRUE_SIGMA = 0.6;
const DATA = Array.from({ length: 12 }, () => TRUE_MU + TRUE_SIGMA * standardNormal(RNG));

const PRIOR = { mu0: 0, lambda0: 1e-3, a0: 1e-3, b0: 1e-3 };
const HISTORY = univariateGaussianVbFit(DATA, PRIOR, { qMu: { mu: 0, lambda: 1 }, qTau: { a: 1, b: 1 } }, 10);

const MU_CURVE: (readonly [number, number])[] = HISTORY.map((h, i) => [i, h.qMu.mu]);
const STD_CURVE: (readonly [number, number])[] = HISTORY.map((h, i) => [i, 1 / Math.sqrt(h.qMu.lambda)]);

export default function UnivariateGaussianVbConvergence() {
  const tokens = useResolvedTokens();
  return (
    <Plot height={260} xDomain={[0, HISTORY.length - 1]} yDomain={[-0.2, 2]} label="q(mu)'s mean and standard deviation settling across coordinate-ascent sweeps">
      <Axes x={{ label: 'sweep' }} y={{ label: 'value' }} grid />
      <Curve points={MU_CURVE} color={tokens.color.accent} width={2} />
      <Curve points={STD_CURVE} color={tokens.color.inkMuted} width={2} dash="dashed" />
    </Plot>
  );
}
