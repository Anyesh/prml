import { gmmEStep, kmeansAssign, kmeansDistortion, kmeansFit, kmeansInit, pcg32, squaredDistance, type GmmParams } from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { faithfulLikeData } from '../data.js';
import '../../widgets.css';

const DATA = faithfulLikeData();
const K = 2;
const SEED = 20260915;
const LOG_EPSILON_MIN = -2.5;
const LOG_EPSILON_MAX = 0.7;
const SAMPLES = 40;

const MEANS = (() => {
  const rng = pcg32(SEED);
  const initial = kmeansInit(rng, DATA, K);
  return kmeansFit(DATA, initial, 10).means;
})();
const HARD_ASSIGNMENTS = kmeansAssign(DATA, MEANS);
const J = kmeansDistortion(DATA, MEANS, HARD_ASSIGNMENTS);

function responsibilityWeightedSum(epsilon: number): number {
  const cov = [
    [epsilon, 0],
    [0, epsilon],
  ];
  const params: GmmParams = { components: MEANS.map((mean) => ({ weight: 1 / K, mean, cov })) };
  const resp = gmmEStep(DATA, params);
  let total = 0;
  DATA.forEach((x, n) => {
    MEANS.forEach((mean, k) => {
      total += resp[n]![k]! * squaredDistance(x, mean);
    });
  });
  return total;
}

const CURVE = Array.from({ length: SAMPLES }, (_, i) => {
  const le = LOG_EPSILON_MIN + ((LOG_EPSILON_MAX - LOG_EPSILON_MIN) * i) / (SAMPLES - 1);
  return [le, responsibilityWeightedSum(10 ** le)] as const;
});

export default function KMeansAsLimit() {
  const tokens = useResolvedTokens();
  const maxY = Math.max(...CURVE.map(([, y]) => y), J) * 1.1;

  return (
    <Plot height={220} xDomain={[LOG_EPSILON_MIN, LOG_EPSILON_MAX]} yDomain={[0, maxY]} label="Responsibility-weighted squared distance as the shared variance shrinks, against the K-means distortion J">
      <Axes x={{ label: 'log10(epsilon)' }} y={{ label: 'sum r * ||x - mean||^2' }} grid />
      <Curve points={CURVE} color={tokens.color.accent} width={2} />
      <Rule y={J} color={tokens.color.success} label={`K-means J = ${J.toFixed(2)}`} />
    </Plot>
  );
}
