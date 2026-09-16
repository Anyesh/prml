import { eigSym, exactHessian, outerProductHessian, unflattenWeights } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { HESSIAN_DATASET, HESSIAN_SPEC, trainHessianToy } from '../hessianToyNetwork';
import '../../widgets.css';

const WEIGHTS = unflattenWeights(HESSIAN_SPEC, trainHessianToy(15));
const EXACT_EIGEN = eigSym(exactHessian(HESSIAN_SPEC, WEIGHTS, HESSIAN_DATASET, 'sumSquared'));
const APPROX_EIGEN = eigSym(outerProductHessian(HESSIAN_SPEC, WEIGHTS, HESSIAN_DATASET, 'sumSquared'));

export default function OuterProductAlwaysPsd() {
  const tokens = useResolvedTokens();
  const smallestExact = Math.min(...EXACT_EIGEN.values);
  const smallestApprox = Math.min(...APPROX_EIGEN.values);
  const bars = [
    { at: 0, value: smallestExact, color: smallestExact >= 0 ? tokens.color.accent : tokens.color.danger },
    { at: 1, value: smallestApprox, color: smallestApprox >= -1e-9 ? tokens.color.accent : tokens.color.danger },
  ];
  const bound = Math.max(Math.abs(smallestExact), Math.abs(smallestApprox)) * 1.3;

  return (
    <Plot height={200} xDomain={[-0.5, 1.5]} yDomain={[-bound, bound]} label="Smallest eigenvalue of each Hessian, partway through training">
      <Axes x={{ label: 'exact (0) vs outer-product (1)' }} y={{ label: 'smallest λ' }} grid zeroLine />
      <Bars bars={bars} thickness={0.5} />
    </Plot>
  );
}
