import {
  designMatrix,
  maximumLikelihoodWeights,
  meanSquaredError,
  pcg32,
  polynomialBasis,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260101;
const NOISE_STD = 0.2;
const N = 20;
const M_VALUES = Array.from({ length: 10 }, (_, m) => m);

const rng = pcg32(SEED, 1);
const XS = Array.from({ length: N }, (_, i) => i / (N - 1));
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));

const testRng = pcg32(SEED, 2);
const TEST_XS = Array.from({ length: 100 }, (_, i) => i / 99);
const TEST_TS = TEST_XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(testRng));

function scoresAt(m: number) {
  const basis = polynomialBasis(m);
  const design = designMatrix(XS, basis);
  const w = maximumLikelihoodWeights(design, TS);
  const trainMse = meanSquaredError(design, TS, w);
  const testRms = Math.sqrt(meanSquaredError(designMatrix(TEST_XS, basis), TEST_TS, w));
  const logLikelihood = -0.5 * N * Math.log(2 * Math.PI) - 0.5 * N * Math.log(trainMse) - 0.5 * N;
  const aic = logLikelihood - (m + 1);
  return { testRms, aic };
}

const SCORES = M_VALUES.map(scoresAt);
const BEST_AIC_M = M_VALUES[SCORES.reduce((best, s, i) => (s.aic > SCORES[best]!.aic ? i : best), 0)]!;
const BEST_TEST_M = M_VALUES[SCORES.reduce((best, s, i) => (s.testRms < SCORES[best]!.testRms ? i : best), 0)]!;

export default function AicPicksModel() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 9]} yDomain={[0, 0.9]} label="True held-out RMS error by polynomial order">
        <Axes x={{ label: 'M' }} y={{ label: 'test RMS' }} grid />
        <Curve points={SCORES.map((s, i) => [i, s.testRms] as const)} color={tokens.series[0]!} width={2} />
        <Rule x={BEST_TEST_M} color={tokens.series[0]!} label={`best by test RMS: M=${BEST_TEST_M}`} />
      </Plot>
      <Plot height={220} xDomain={[0, 9]} yDomain={[Math.min(...SCORES.map((s) => s.aic)) - 2, Math.max(...SCORES.map((s) => s.aic)) + 2]} label="AIC score by polynomial order, computed from training data alone">
        <Axes x={{ label: 'M' }} y={{ label: 'AIC' }} grid />
        <Curve points={SCORES.map((s, i) => [i, s.aic] as const)} color={tokens.series[1]!} width={2} />
        <Rule x={BEST_AIC_M} color={tokens.series[1]!} label={`best by AIC: M=${BEST_AIC_M}`} />
        <Legend entries={[{ label: 'higher AIC is better, 1.73', color: tokens.series[1]!, mark: 'line' }]} placement="bottom-right" />
      </Plot>
      <p className="widget-readout">
        {`AIC, using only the ${N} training points, picks M=${BEST_AIC_M}; the true test RMS, using 100 held-out points, is minimised at M=${BEST_TEST_M}. Both land in the same low-error region without AIC ever touching the held-out set.`}
      </p>
    </div>
  );
}
