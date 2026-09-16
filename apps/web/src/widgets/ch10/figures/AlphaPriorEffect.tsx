import { kmeansInit, pcg32, standardNormal, vbGmmFit, type GaussianWishart, type Mat } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const TRUE_CLUSTERS: readonly (readonly [number, number])[] = [
  [-2.5, -2],
  [2.5, -1.5],
  [0, 2.8],
];
const POINTS_PER_CLUSTER = 18;
const K = 6;
const ROUNDS = 15;
const ALPHA0_VALUES = [0.001, 1, 10];

function syntheticData(): Mat {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  for (const [cx, cy] of TRUE_CLUSTERS) {
    for (let i = 0; i < POINTS_PER_CLUSTER; i++) points.push([cx + 0.7 * standardNormal(rng), cy + 0.7 * standardNormal(rng)]);
  }
  return points;
}

const DATA = syntheticData();

function survivingCount(alpha0: number): number {
  const rng = pcg32(SEED + 1);
  const means = kmeansInit(rng, DATA, K);
  const prior = { alpha0, beta0: 1, mean0: [0, 0], scale0: [[0.02, 0], [0, 0.02]], dof0: 2 };
  const initial = {
    alpha: new Array(K).fill(alpha0),
    components: means.map((m): GaussianWishart => ({ beta: prior.beta0, mean: m, scale: prior.scale0, dof: prior.dof0 })),
  };
  const fit = vbGmmFit(DATA, initial, prior, ROUNDS);
  const finalPosterior = fit.posteriorHistory[fit.posteriorHistory.length - 1]!;
  const total = finalPosterior.alpha.reduce((s, a) => s + a, 0);
  return finalPosterior.alpha.filter((a) => a / total > 0.05).length;
}

export default function AlphaPriorEffect() {
  const tokens = useResolvedTokens();
  const counts = ALPHA0_VALUES.map(survivingCount);
  const bars = counts.map((c, i) => ({ at: i, value: c, color: tokens.series[i % tokens.series.length]! }));

  return (
    <Plot height={200} xDomain={[-0.5, 2.5]} yDomain={[0, K + 0.5]} label="Number of components surviving as the Dirichlet prior concentration alpha0 changes">
      <Axes x={{ label: 'alpha0', ticks: [0, 1, 2], format: (v) => `${ALPHA0_VALUES[v] ?? ''}` }} y={{ label: 'components kept of 6' }} grid />
      <Bars bars={bars} thickness={0.5} />
    </Plot>
  );
}
