import { emLowerBound, gmmEStep, gmmLogLikelihood, linspace, pcg32, standardNormal, type GmmParams, type Mat } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260916;
const CLUSTER_A_MEAN = -2;
const CLUSTER_B_MEAN = 2;
const CLUSTER_SIZE = 6;
const FIXED_MEAN = -2.2;
const COV: number[][] = [[1]];
const THETA_START = 0.6;
const THETA_MIN = -4;
const THETA_MAX = 4;
const GRID_SAMPLES = 90;
const ROUNDS = 3;

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

function argmaxOverGrid(values: readonly number[], grid: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i]! > values[best]!) best = i;
  return grid[best]!;
}

const THETA_GRID = linspace(THETA_MIN, THETA_MAX, GRID_SAMPLES);

const LL_HISTORY: number[] = (() => {
  const history: number[] = [];
  let theta = THETA_START;
  history.push(gmmLogLikelihood(DATA, paramsAt(theta)));
  for (let r = 0; r < ROUNDS; r++) {
    const responsibilities = gmmEStep(DATA, paramsAt(theta));
    const boundCurve = THETA_GRID.map((t) => emLowerBound(DATA, responsibilities, paramsAt(t)));
    theta = argmaxOverGrid(boundCurve, THETA_GRID);
    history.push(gmmLogLikelihood(DATA, paramsAt(theta)));
  }
  return history;
})();

export default function LogLikelihoodTrace() {
  const tokens = useResolvedTokens();
  const points = LL_HISTORY.map((v, i) => [i, v] as const);
  const minY = Math.min(...LL_HISTORY);
  const maxY = Math.max(...LL_HISTORY);
  const pad = (maxY - minY) * 0.15 || 1;

  return (
    <Plot height={200} xDomain={[-0.3, ROUNDS + 0.3]} yDomain={[minY - pad, maxY + pad]} label="Log-likelihood at theta after each M-step of the widget above">
      <Axes x={{ label: 'M-step', ticks: LL_HISTORY.map((_, i) => i) }} y={{ label: 'ln p(X|theta)' }} grid />
      <Curve points={points} color={tokens.color.accent} width={2} />
      <ScatterField points={points.map(([x, y]) => ({ x, y }))} color={tokens.color.accent} size={4.5} />
    </Plot>
  );
}
