import { expectedCompleteDataLogLikelihood, gmmEmStep, kmeansInit, pcg32, type GmmParams } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
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

function computeSeries() {
  const rng = pcg32(SEED);
  const initialMeans = kmeansInit(rng, DATA, K);
  let params: GmmParams = { components: initialMeans.map((mean) => ({ weight: 1 / K, mean, cov: ISOTROPIC_COV })) };
  const incomplete: number[] = [];
  const complete: number[] = [];
  for (let i = 0; i < ROUNDS; i++) {
    const step = gmmEmStep(DATA, params);
    incomplete.push(step.logLikelihood);
    complete.push(expectedCompleteDataLogLikelihood(DATA, step.responsibilities, params));
    params = step.params;
  }
  return { incomplete, complete };
}

const { incomplete, complete } = computeSeries();

export default function CompleteVsIncompleteLL() {
  const tokens = useResolvedTokens();
  const incompletePoints = incomplete.map((v, i) => [i + 1, v] as const);
  const completePoints = complete.map((v, i) => [i + 1, v] as const);
  const minY = Math.min(...incomplete, ...complete);
  const maxY = Math.max(...incomplete, ...complete);
  const pad = (maxY - minY) * 0.1 || 1;

  return (
    <Plot height={220} xDomain={[0.5, ROUNDS + 0.5]} yDomain={[minY - pad, maxY + pad]} label="Incomplete-data log-likelihood against the expected complete-data log-likelihood, per EM iteration">
      <Axes x={{ label: 'EM iteration' }} y={{ label: 'nats' }} grid />
      <Curve points={incompletePoints} color={tokens.series[0]!} width={2} />
      <Curve points={completePoints} color={tokens.series[1]!} width={2} dash="dashed" />
      <Legend
        entries={[
          { label: 'ln p(X|theta)', color: tokens.series[0]!, mark: 'dot' },
          { label: 'Q(theta,theta) = E[ln p(X,Z|theta)]', color: tokens.series[1]!, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
