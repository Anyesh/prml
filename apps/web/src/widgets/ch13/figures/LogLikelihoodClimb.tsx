import { hmmFitEM, type HmmGaussianParams } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { hmmDemo } from '../data.js';

const DEMO = hmmDemo(30);
const MAX_ITERS = 15;

const INITIAL: HmmGaussianParams = {
  pi: [1 / 3, 1 / 3, 1 / 3],
  A: [
    [0.5, 0.25, 0.25],
    [0.25, 0.5, 0.25],
    [0.25, 0.25, 0.5],
  ],
  components: [
    { mean: [1, 2], cov: [[2, 0], [0, 2]] },
    { mean: [3, 0], cov: [[2, 0], [0, 2]] },
    { mean: [-1, 3], cov: [[2, 0], [0, 2]] },
  ],
};

export default function LogLikelihoodClimb() {
  const tokens = useResolvedTokens();
  const fit = hmmFitEM(DEMO.data, INITIAL, MAX_ITERS);
  const points = fit.logLikelihoodHistory.map((ll, i) => [i + 1, ll] as const);
  const minLL = Math.min(...fit.logLikelihoodHistory);
  const maxLL = Math.max(...fit.logLikelihoodHistory);

  return (
    <Plot width={360} height={220} xDomain={[1, MAX_ITERS]} yDomain={[minLL - 1, maxLL + 1]} label="HMM log-likelihood across EM iterations">
      <Axes x={{ label: 'EM iteration' }} y={{ label: 'log p(X)' }} grid />
      <Curve points={points} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
