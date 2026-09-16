import { dataMean, pcg32, ppcaFitEM, ppcaMLE, standardNormal, type Mat, type PpcaParams } from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { ppcaDemoData } from '../data.js';

import '../../widgets.css';

const DATA = ppcaDemoData();
const MAX_ROUNDS = 8;
const INIT_SEED = 20260920;

function buildInitial(data: Mat): PpcaParams {
  const rng = pcg32(INIT_SEED);
  const mean = dataMean(data);
  return { mean, w: [[0.3 * standardNormal(rng)], [0.3 * standardNormal(rng)]], sigma2: 1 };
}

const FIT = ppcaFitEM(DATA, buildInitial(DATA), MAX_ROUNDS);
const TARGET = ppcaMLE(DATA, 1).sigma2;
const SIGMA_TRACE = FIT.paramsHistory.map((p, i) => [i, p.sigma2] as const);

export default function EmSigmaConvergence() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[0, MAX_ROUNDS]} yDomain={[0, 1.05]} label="Sigma squared across EM rounds, against the closed-form maximum-likelihood value">
      <Axes x={{ label: 'round' }} y={{ label: 'sigma^2' }} grid />
      <Curve points={SIGMA_TRACE} color={tokens.color.accent} width={2} />
      <Rule y={TARGET} color={tokens.color.borderStrong} dash label="closed-form" />
    </Plot>
  );
}
