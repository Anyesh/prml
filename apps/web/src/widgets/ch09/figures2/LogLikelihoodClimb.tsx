import { gmmFitEM, kmeansInit, pcg32, type GmmParams } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { faithfulLikeData } from '../data.js';
import '../../widgets.css';

const DATA = faithfulLikeData();
const K = 2;
const SEED = 20260915;
const ROUNDS = 8;
const ISOTROPIC_COV = [
  [0.8, 0],
  [0, 0.8],
];

function computeHistory(): number[] {
  const rng = pcg32(SEED);
  const initialMeans = kmeansInit(rng, DATA, K);
  const initialParams: GmmParams = {
    components: initialMeans.map((mean) => ({ weight: 1 / K, mean, cov: ISOTROPIC_COV })),
  };
  return gmmFitEM(DATA, initialParams, ROUNDS).logLikelihoodHistory;
}

const HISTORY = computeHistory();

export default function LogLikelihoodClimb() {
  const tokens = useResolvedTokens();
  const points = HISTORY.map((v, i) => [i + 1, v] as const);
  const minY = Math.min(...HISTORY);
  const maxY = Math.max(...HISTORY);
  const pad = (maxY - minY) * 0.1 || 1;

  return (
    <Plot height={220} xDomain={[0.5, ROUNDS + 0.5]} yDomain={[minY - pad, maxY + pad]} label="Incomplete-data log-likelihood before each M-step, on the same run as the widget above">
      <Axes x={{ label: 'EM iteration' }} y={{ label: 'ln p(X|theta)' }} grid />
      <Curve points={points} color={tokens.color.accent} width={2} />
      <ScatterField points={points.map(([x, y]) => ({ x, y }))} color={tokens.color.accent} size={4} />
    </Plot>
  );
}
