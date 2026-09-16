import { eigSym, exactHessian, unflattenWeights } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { HESSIAN_DATASET, HESSIAN_SPEC, trainHessianToy } from '../hessianToyNetwork';
import '../../widgets.css';

const WEIGHTS = unflattenWeights(HESSIAN_SPEC, trainHessianToy(1200));
const HESSIAN = exactHessian(HESSIAN_SPEC, WEIGHTS, HESSIAN_DATASET, 'sumSquared');
const EIGEN = eigSym(HESSIAN);
const LOG_FLOOR = -4;

export default function EigenvalueSpectrum() {
  const tokens = useResolvedTokens();
  const bars = EIGEN.values.map((value, i) => ({
    at: i,
    value: Math.max(Math.log10(Math.max(Math.abs(value), 1e-8)), LOG_FLOOR),
    color: value >= 0 ? tokens.color.accent : tokens.color.danger,
  }));
  const top = Math.max(...bars.map((b) => b.value)) + 0.3;

  return (
    <Plot height={220} xDomain={[-0.5, EIGEN.values.length - 0.5]} yDomain={[LOG_FLOOR, top]} label="Hessian eigenvalues at a trained minimum, sorted descending">
      <Axes x={{ label: 'eigenvalue index' }} y={{ label: 'log₁₀ λ' }} grid />
      <Bars bars={bars} thickness={0.6} baseline={LOG_FLOOR} />
    </Plot>
  );
}
