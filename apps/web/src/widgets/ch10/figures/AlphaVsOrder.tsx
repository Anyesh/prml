import { designMatrix, pcg32, polynomialBasis, standardNormal, vlrFit } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const N = 12;
const TRUE_ORDER = 3;
const TRUE_WEIGHTS = [0.3, 1.2, -0.8, 0.5];
const NOISE = 0.3;
const BETA = 1 / (NOISE * NOISE);
const PRIOR = { a0: 1e-3, b0: 1e-3 };
const ITERS = 30;
const ORDERS = [1, 2, 3, 4, 5, 6, 7];

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

export default function AlphaVsOrder() {
  const tokens = useResolvedTokens();
  const { xs, targets } = syntheticData();

  const finalAlpha = ORDERS.map((order) => {
    const design = designMatrix(xs, polynomialBasis(order));
    const fit = vlrFit(design, targets, BETA, PRIOR, ITERS);
    const posterior = fit.posteriorHistory[fit.posteriorHistory.length - 1]!;
    return posterior.a / posterior.b;
  });

  const bars = ORDERS.map((order, i) => ({ at: order, value: finalAlpha[i]!, color: order === TRUE_ORDER ? tokens.color.accent : tokens.color.inkMuted }));

  return (
    <Plot height={200} xDomain={[0.5, 7.5]} yDomain={[0, Math.max(...finalAlpha) * 1.15]} label="Converged E[alpha] as the offered polynomial order grows past the true cubic">
      <Axes x={{ label: 'offered order M', ticks: ORDERS }} y={{ label: 'E[alpha]' }} grid />
      <Bars bars={bars} thickness={0.5} />
    </Plot>
  );
}
