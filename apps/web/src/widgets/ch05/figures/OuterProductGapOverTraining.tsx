import { exactHessian, outerProductHessian, unflattenWeights } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { HESSIAN_DATASET, HESSIAN_SPEC, trainHessianToy } from '../hessianToyNetwork';
import '../../widgets.css';

const CHECKPOINTS = [0, 2, 5, 10, 20, 40, 80, 150];

function frobeniusRelativeGap(a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a.length; j++) {
      const d = a[i]![j]! - b[i]![j]!;
      num += d * d;
      den += a[i]![j]! * a[i]![j]!;
    }
  }
  return Math.sqrt(num) / Math.sqrt(den);
}

const GAPS = CHECKPOINTS.map((steps) => {
  const weights = unflattenWeights(HESSIAN_SPEC, trainHessianToy(steps));
  const exact = exactHessian(HESSIAN_SPEC, weights, HESSIAN_DATASET, 'sumSquared');
  const approx = outerProductHessian(HESSIAN_SPEC, weights, HESSIAN_DATASET, 'sumSquared');
  return frobeniusRelativeGap(exact, approx);
});

export default function OuterProductGapOverTraining() {
  const tokens = useResolvedTokens();
  const points = CHECKPOINTS.map((steps, i) => [steps, GAPS[i]!] as const);

  return (
    <Plot height={220} xDomain={[0, 150]} yDomain={[0, Math.max(...GAPS) * 1.1]} label="Relative gap between the exact and outer-product Hessians as training progresses">
      <Axes x={{ label: 'training steps' }} y={{ label: '‖Hexact − Happrox‖ / ‖Hexact‖' }} grid />
      <Curve points={points} color={tokens.color.accent} width={2} />
      <ScatterField points={points.map(([x, y], i) => ({ x, y, id: i }))} color={tokens.color.accent} size={4} />
    </Plot>
  );
}
