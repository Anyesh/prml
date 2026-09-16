import { emLowerBound, gmmEStep, gmmLogLikelihood, linspace, pcg32, standardNormal, type GmmParams, type Mat } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
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
const LL_CURVE = THETA_GRID.map((t) => [t, gmmLogLikelihood(DATA, paramsAt(t))] as const);
const BOUND_CURVE = THETA_GRID.map((t) => [t, emLowerBound(DATA, Q_OLD, paramsAt(t))] as const);
const TOUCH_Y = gmmLogLikelihood(DATA, paramsAt(THETA_OLD));

export default function BoundTouchesCurve() {
  const tokens = useResolvedTokens();
  const ys = [...LL_CURVE.map(([, y]) => y), ...BOUND_CURVE.map(([, y]) => y)];
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = (maxY - minY) * 0.1;

  return (
    <Plot height={220} xDomain={[THETA_MIN, THETA_MAX]} yDomain={[minY - pad, maxY + pad]} label="The log-likelihood curve and one bound, touching at the theta the bound was built from">
      <Axes x={{ label: 'theta' }} y={{ label: 'nats' }} grid />
      <Curve points={LL_CURVE} color={tokens.color.danger} width={2} />
      <Curve points={BOUND_CURVE} color={tokens.color.accent} width={2} dash="dashed" />
      <ScatterField points={[{ x: THETA_OLD, y: TOUCH_Y, color: tokens.color.ink, size: 5 }]} />
      <Legend
        entries={[
          { label: 'ln p(X|theta)', color: tokens.color.danger, mark: 'line' },
          { label: 'L(q,theta)', color: tokens.color.accent, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
