import { eigSym, exactHessian, unflattenWeights } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { HESSIAN_DATASET, HESSIAN_SPEC, trainHessianToy } from '../hessianToyNetwork';
import '../../widgets.css';

const WEIGHTS = unflattenWeights(HESSIAN_SPEC, trainHessianToy(1200));
const HESSIAN = exactHessian(HESSIAN_SPEC, WEIGHTS, HESSIAN_DATASET, 'sumSquared');
const EIGEN = eigSym(HESSIAN);

// A log axis cannot reach zero, and a flat direction's eigenvalue is numerically zero, so the
// floor stands in for "too small to distinguish from flat" rather than for a real magnitude.
const FLOOR = 1e-4;

export default function EigenvalueSpectrum() {
  const tokens = useResolvedTokens();
  const bars = EIGEN.values.map((value, i) => ({
    at: i,
    value: Math.max(Math.abs(value), FLOOR),
    color: value >= 0 ? tokens.color.accent : tokens.color.danger,
  }));
  const top = Math.max(...bars.map((b) => b.value)) * 3;

  return (
    <Plot
      height={220}
      xDomain={[-0.5, EIGEN.values.length - 0.5]}
      yDomain={[FLOOR, top]}
      yScaleKind="log"
      label="Hessian eigenvalues at a trained minimum, sorted descending, on a logarithmic axis"
    >
      <Axes x={{ label: 'eigenvalue index' }} y={{ label: 'eigenvalue' }} grid />
      <Bars bars={bars} thickness={0.6} baseline={FLOOR} />
    </Plot>
  );
}
