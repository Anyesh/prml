import { designMatrix, pcg32, polynomialBasis, standardNormal, vlrFit } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const N = 12;
const TRUE_ORDER = 3;
const TRUE_WEIGHTS = [0.3, 1.2, -0.8, 0.5];
const OFFERED_ORDER = 7;
const NOISE = 0.3;
const BETA = 1 / (NOISE * NOISE);
const PRIOR = { a0: 1e-3, b0: 1e-3 };
const ITERS = 30;

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

export default function PosteriorWeightShrinkage() {
  const tokens = useResolvedTokens();
  const { xs, targets } = syntheticData();
  const design = designMatrix(xs, polynomialBasis(OFFERED_ORDER));
  const fit = vlrFit(design, targets, BETA, PRIOR, ITERS);
  const final = fit.posteriorHistory[fit.posteriorHistory.length - 1]!;

  const bars = final.mean.map((m, i) => ({
    at: i,
    value: Math.abs(m),
    color: i <= TRUE_ORDER ? tokens.color.accent : tokens.color.danger,
  }));

  return (
    <Plot height={200} xDomain={[-0.5, OFFERED_ORDER + 0.5]} yDomain={[0, Math.max(...bars.map((b) => b.value)) * 1.15]} label="Posterior mean weight magnitude by polynomial term, order 7 offered against a true cubic">
      <Axes x={{ label: 'term (x^i)', ticks: Array.from({ length: OFFERED_ORDER + 1 }, (_, i) => i) }} y={{ label: '|m_i|' }} grid />
      <Bars bars={bars} thickness={0.5} />
    </Plot>
  );
}
