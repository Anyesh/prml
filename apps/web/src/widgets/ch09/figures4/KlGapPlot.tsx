import { emKlGap, gmmEStep, linspace, pcg32, standardNormal, type GmmParams, type Mat } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260916;
const CLUSTER_A_MEAN = -2;
const CLUSTER_B_MEAN = 2;
const CLUSTER_SIZE = 6;
const FIXED_MEAN = -2.2;
const COV: number[][] = [[1]];
const THETA_OLD = 0.6;
const THETA_MIN = -4;
const THETA_MAX = 4;
const GRID_SAMPLES = 90;

const DATA: Mat = (() => {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  for (let i = 0; i < CLUSTER_SIZE; i++) points.push([CLUSTER_A_MEAN + 0.6 * standardNormal(rng)]);
  for (let i = 0; i < CLUSTER_SIZE; i++) points.push([CLUSTER_B_MEAN + 0.6 * standardNormal(rng)]);
  return points;
})();

function paramsAt(theta: number): GmmParams {
  return {
    components: [
      { weight: 0.5, mean: [FIXED_MEAN], cov: COV },
      { weight: 0.5, mean: [theta], cov: COV },
    ],
  };
}

const Q_OLD = gmmEStep(DATA, paramsAt(THETA_OLD));
const THETA_GRID = linspace(THETA_MIN, THETA_MAX, GRID_SAMPLES);
const GAP_CURVE = THETA_GRID.map((theta) => [theta, emKlGap(DATA, Q_OLD, paramsAt(theta))] as const);

export default function KlGapPlot() {
  const tokens = useResolvedTokens();
  const maxY = Math.max(...GAP_CURVE.map(([, y]) => y)) * 1.1;

  return (
    <Plot height={200} xDomain={[THETA_MIN, THETA_MAX]} yDomain={[0, maxY]} label="KL(q||p) as theta moves away from the point q was fitted to">
      <Axes x={{ label: 'theta' }} y={{ label: 'KL(q||p(Z|X,theta))' }} grid />
      <Curve points={GAP_CURVE} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
