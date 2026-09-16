import { designMatrix, pcg32, polynomialBasis, standardNormal, vlrFit } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const N = 12;
const TRUE_ORDER = 3;
const TRUE_WEIGHTS = [0.3, 1.2, -0.8, 0.5];
const NOISE = 0.3;
const BETA = 1 / (NOISE * NOISE);
const PRIOR = { a0: 1e-3, b0: 1e-3 };
const ITERS = 15;

function syntheticData() {
  const rng = pcg32(SEED);
  const xs = Array.from({ length: N }, (_, i) => -2 + (4 * i) / (N - 1));
  const targets = xs.map((x) => {
    const truePhi = polynomialBasis(TRUE_ORDER)(x);
    const mean = truePhi.reduce((s, v, i) => s + v * TRUE_WEIGHTS[i]!, 0);
    return mean + NOISE * standardNormal(rng);
  });
  return { xs, targets };
}

export default function AlphaConvergence() {
  const tokens = useResolvedTokens();
  const { xs, targets } = syntheticData();
  const design = designMatrix(xs, polynomialBasis(TRUE_ORDER));
  const fit = vlrFit(design, targets, BETA, PRIOR, ITERS);

  const points = fit.posteriorHistory.map((p, i) => [i, p.a / p.b] as const);

  return (
    <Plot height={200} xDomain={[0, ITERS - 1]} yDomain={[0, Math.max(...points.map((p) => p[1])) * 1.15]} label="E[alpha] across coordinate-ascent sweeps, fit at the true polynomial order">
      <Axes x={{ label: 'sweep' }} y={{ label: 'E[alpha]' }} grid />
      <Curve points={points} color={tokens.series[0]!} width={2} />
    </Plot>
  );
}
