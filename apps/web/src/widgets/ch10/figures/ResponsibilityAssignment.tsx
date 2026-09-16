import { kmeansInit, pcg32, standardNormal, vbGmmFit, vbGmmResponsibilities, type GaussianWishart, type Mat, type VbGmmPrior } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const TRUE_CLUSTERS: readonly (readonly [number, number])[] = [
  [-2.5, -2],
  [2.5, -1.5],
  [0, 2.8],
];
const POINTS_PER_CLUSTER = 18;
const K = 6;
const ROUNDS = 12;
const DOMAIN: readonly [number, number] = [-6, 6];
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

export default function ResponsibilityAssignment() {
  const tokens = useResolvedTokens();
  const rng = pcg32(SEED + 1);
  const means = kmeansInit(rng, DATA, K);
  const initial = {
    alpha: new Array(K).fill(PRIOR.alpha0),
    components: means.map((m): GaussianWishart => ({ beta: PRIOR.beta0, mean: m, scale: PRIOR.scale0, dof: PRIOR.dof0 })),
  };
  const fit = vbGmmFit(DATA, initial, PRIOR, ROUNDS);
  const posterior = fit.posteriorHistory[fit.posteriorHistory.length - 1]!;
  const responsibilities = vbGmmResponsibilities(DATA, posterior);

  const points = DATA.map((p, n) => {
    const r = responsibilities[n]!;
    const best = r.indexOf(Math.max(...r));
    return { x: p[0]!, y: p[1]!, color: tokens.series[best % tokens.series.length]!, size: 4 };
  });

  return (
    <Plot width={280} height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Each point coloured by its most responsible surviving component">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
      <ScatterField points={points} />
    </Plot>
  );
}
