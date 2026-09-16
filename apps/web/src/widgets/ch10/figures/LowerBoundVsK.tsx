import { kmeansInit, pcg32, standardNormal, vbGmmFit, vbGmmLowerBound, type GaussianWishart, type Mat, type VbGmmPrior } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const TRUE_CLUSTERS: readonly (readonly [number, number])[] = [
  [-2.5, -2],
  [2.5, -1.5],
  [0, 2.8],
];
const POINTS_PER_CLUSTER = 18;
const K_VALUES = [1, 2, 3, 4, 5, 6];
const ROUNDS = 15;
const PRIOR: VbGmmPrior = { alpha0: 1e-3, beta0: 1, mean0: [0, 0], scale0: [[0.02, 0], [0, 0.02]], dof0: 2 };

function syntheticData(): Mat {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  for (const [cx, cy] of TRUE_CLUSTERS) {
    for (let i = 0; i < POINTS_PER_CLUSTER; i++) points.push([cx + 0.7 * standardNormal(rng), cy + 0.7 * standardNormal(rng)]);
  }
  return points;
}

const DATA = syntheticData();

function finalLowerBound(k: number): number {
  const rng = pcg32(SEED + k);
  const means = kmeansInit(rng, DATA, k);
  const initial = {
    alpha: new Array(k).fill(PRIOR.alpha0),
    components: means.map((m): GaussianWishart => ({ beta: PRIOR.beta0, mean: m, scale: PRIOR.scale0, dof: PRIOR.dof0 })),
  };
  const fit = vbGmmFit(DATA, initial, PRIOR, ROUNDS);
  const finalPosterior = fit.posteriorHistory[fit.posteriorHistory.length - 1]!;
  const finalR = fit.responsibilitiesHistory[fit.responsibilitiesHistory.length - 1]!;
  return vbGmmLowerBound(DATA, finalR, finalPosterior, PRIOR);
}

export default function LowerBoundVsK() {
  const tokens = useResolvedTokens();
  const bounds = K_VALUES.map(finalLowerBound);
  const best = Math.max(...bounds);
  const bars = K_VALUES.map((k, i) => ({ at: k, value: bounds[i]!, color: bounds[i]! === best ? tokens.color.accent : tokens.color.inkMuted }));
  const range = Math.max(...bounds) - Math.min(...bounds);

  return (
    <Plot height={220} xDomain={[0.5, 6.5]} yDomain={[Math.min(...bounds) - 0.05 * range, best + 0.05 * range]} label="Converged variational lower bound as a function of the number of offered components">
      <Axes x={{ label: 'K', ticks: K_VALUES }} y={{ label: 'L (lower bound)' }} grid />
      <Bars bars={bars} thickness={0.5} baseline={Math.min(...bounds) - 0.05 * range} />
    </Plot>
  );
}
