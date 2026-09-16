import { pcg32, standardNormal, variationalLogisticFit } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const N_PER_CLASS = 14;
const PRIOR = { mean: [0, 0, 0], cov: [[4, 0, 0], [0, 4, 0], [0, 0, 4]] };
const ROUNDS = 8;

function syntheticData() {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < N_PER_CLASS; i++) {
    points.push([-1.3 + 0.7 * standardNormal(rng), -1 + 0.7 * standardNormal(rng)]);
    labels.push(0);
  }
  for (let i = 0; i < N_PER_CLASS; i++) {
    points.push([1.3 + 0.7 * standardNormal(rng), 1 + 0.7 * standardNormal(rng)]);
    labels.push(1);
  }
  return { points, labels };
}

const { points, labels } = syntheticData();
const DESIGN = points.map((p) => [1, p[0]!, p[1]!]);

export default function LogisticLowerBoundCurve() {
  const tokens = useResolvedTokens();
  const xiInit = new Array(DESIGN.length).fill(1);
  const fit = variationalLogisticFit(DESIGN, labels, PRIOR, xiInit, ROUNDS);
  const points_ = fit.lowerBoundHistory.map((v, i) => [i, v] as const);

  return (
    <Plot height={200} xDomain={[0, ROUNDS - 1]} yDomain={[Math.min(...fit.lowerBoundHistory), Math.max(...fit.lowerBoundHistory)]} label="The closed-form bound (10.164) across coordinate-ascent rounds">
      <Axes x={{ label: 'round' }} y={{ label: 'L(xi)' }} grid />
      <Curve points={points_} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
