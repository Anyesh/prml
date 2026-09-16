import { evalGrid, inverse, submatrix, symmetrise } from '@prml/math';
import { Axes, ContourField, CovarianceEllipse, Plot, useResolvedTokens } from '@prml/viz';
import { toyDescend, toyErrorAt, toyExactHessianAt } from '../toyErrorSurface';
import '../../widgets.css';

const CONVERGED = toyDescend([-3.5, 3.2], 0.12, 600);
const MINIMUM = CONVERGED[CONVERGED.length - 1]!;
const HESSIAN = symmetrise(submatrix(toyExactHessianAt(MINIMUM[0], MINIMUM[1]), [1, 3], [1, 3]));
const COVARIANCE = inverse(HESSIAN);
const HALF_WIDTH = 1.2;
const LOCAL_GRID = Array.from({ length: 61 }, (_, i) => MINIMUM[0]! - HALF_WIDTH + (2 * HALF_WIDTH * i) / 60);
const LOCAL_GRID_Y = Array.from({ length: 61 }, (_, i) => MINIMUM[1]! - HALF_WIDTH + (2 * HALF_WIDTH * i) / 60);
const LOCAL_SURFACE = evalGrid(LOCAL_GRID, LOCAL_GRID_Y, toyErrorAt);

export default function QuadraticApproxEllipse() {
  const tokens = useResolvedTokens();

  return (
    <Plot
      height={240}
      xDomain={[MINIMUM[0]! - HALF_WIDTH, MINIMUM[0]! + HALF_WIDTH]}
      yDomain={[MINIMUM[1]! - HALF_WIDTH, MINIMUM[1]! + HALF_WIDTH]}
      equalAspect
      label="Actual error contours near a minimum against the quadratic approximation's ellipse"
    >
      <ContourField data={LOCAL_SURFACE} levelCount={10} color={tokens.color.inkMuted} />
      <Axes x={{ label: 'w₁' }} y={{ label: 'w₂' }} grid />
      <CovarianceEllipse mean={[MINIMUM[0]!, MINIMUM[1]!]} cov={COVARIANCE} levels={[0.5]} color={tokens.color.accent} width={2} dash="dashed" />
    </Plot>
  );
}
