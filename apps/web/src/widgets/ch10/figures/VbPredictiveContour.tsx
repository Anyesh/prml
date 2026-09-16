import { evalGrid, kmeansInit, linspace, pcg32, standardNormal, vbGmmFit, vbGmmPredictiveLogPdf, type GaussianWishart, type Mat, type VbGmmPrior } from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
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
const GRID = linspace(DOMAIN[0], DOMAIN[1], 48);
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

export default function VbPredictiveContour() {
  const tokens = useResolvedTokens();
  const rng = pcg32(SEED + 1);
  const means = kmeansInit(rng, DATA, K);
  const initial = {
    alpha: new Array(K).fill(PRIOR.alpha0),
    components: means.map((m): GaussianWishart => ({ beta: PRIOR.beta0, mean: m, scale: PRIOR.scale0, dof: PRIOR.dof0 })),
  };
  const fit = vbGmmFit(DATA, initial, PRIOR, ROUNDS);
  const posterior = fit.posteriorHistory[fit.posteriorHistory.length - 1]!;

  const density = evalGrid(GRID, GRID, (x, y) => Math.exp(vbGmmPredictiveLogPdf([x, y], posterior)));
  let peak = 0;
  for (const row of density.values) for (const v of row) if (v > peak) peak = v;
  const fill = sequentialScale([0, peak]);

  return (
    <Plot width={280} height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="The Student-t predictive density (10.81) built from the converged posterior">
      <ContourField data={density} levelCount={6} color={tokens.color.inkFaint} fill={fill} z={-1} opacity={0.6} />
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
      <ScatterField points={DATA.map((p) => ({ x: p[0]!, y: p[1]!, color: tokens.color.ink, size: 3 }))} />
    </Plot>
  );
}
