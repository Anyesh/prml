import { fitLaplaceLogisticPosterior, laplaceLogisticPredictive, linspace, logisticPredict, pcg32, standardNormal } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const rng = pcg32(20260450);
const N_PER_CLASS = 8;
const XS0 = Array.from({ length: N_PER_CLASS }, () => -1.5 + 0.5 * standardNormal(rng));
const XS1 = Array.from({ length: N_PER_CLASS }, () => 1.5 + 0.5 * standardNormal(rng));
const DESIGN = [...XS0, ...XS1].map((x) => [1, x]);
const TARGETS = [...XS0.map(() => 0), ...XS1.map(() => 1)];

const POSTERIOR = fitLaplaceLogisticPosterior(DESIGN, TARGETS, {
  mean: [0, 0],
  covariance: [
    [9, 0],
    [0, 9],
  ],
});

const GRID = linspace(-8, 8, 200);

export default function PredictiveShrinkage1D() {
  const tokens = useResolvedTokens();
  const pointCurve = GRID.map((x) => [x, logisticPredict(POSTERIOR.mean, [1, x])] as const);
  const predictiveCurve = GRID.map((x) => [x, laplaceLogisticPredictive([1, x], POSTERIOR)] as const);

  return (
    <Plot height={220} xDomain={[-8, 8]} yDomain={[-0.02, 1.02]} label="A one-feature slice: point estimate against the marginalised predictive">
      <Axes x={{ label: 'x' }} y={{ label: 'p(C1)' }} grid />
      <Curve points={pointCurve} color={tokens.color.danger} width={2} />
      <Curve points={predictiveCurve} color={tokens.color.accent} width={2} />
      <ScatterField points={XS0.map((x, i) => ({ x, y: 0, id: `0-${i}` }))} color={tokens.series[0]} size={3} />
      <ScatterField points={XS1.map((x, i) => ({ x, y: 1, id: `1-${i}` }))} color={tokens.series[1]} size={3} />
      <Legend
        entries={[
          { label: 'σ(wMAPᵀφ)', color: tokens.color.danger, mark: 'line' },
          { label: 'marginalised predictive', color: tokens.color.accent, mark: 'line' },
        ]}
        placement="top-left"
      />
    </Plot>
  );
}
