import { designMatrix, pcg32, polynomialBasis, standardNormal, vlrFit } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const N = 12;
const TRUE_ORDER = 3;
const TRUE_WEIGHTS = [0.3, 1.2, -0.8, 0.5];
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

export default function FittedCurveWithData() {
  const tokens = useResolvedTokens();
  const { xs, targets } = syntheticData();
  const design = designMatrix(xs, polynomialBasis(TRUE_ORDER));
  const fit = vlrFit(design, targets, BETA, PRIOR, ITERS);
  const weights = fit.posteriorHistory[fit.posteriorHistory.length - 1]!.mean;

  const curveXs = Array.from({ length: 80 }, (_, i) => -2 + (4 * i) / 79);
  const curve = curveXs.map((x) => {
    const phi = polynomialBasis(TRUE_ORDER)(x);
    return [x, phi.reduce((s, v, i) => s + v * weights[i]!, 0)] as const;
  });

  return (
    <Plot height={220} xDomain={[-2, 2]} yDomain={[Math.min(...targets) - 0.5, Math.max(...targets) + 0.5]} label="The converged posterior mean fit against the training data it was fit to">
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <Curve points={curve} color={tokens.series[0]!} width={2} />
      <ScatterField points={xs.map((x, i) => ({ x, y: targets[i]!, color: tokens.color.ink, size: 4 }))} />
    </Plot>
  );
}
