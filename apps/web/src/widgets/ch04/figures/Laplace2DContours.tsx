import { evalGrid, gammaPdf, laplaceApproximation, linspace } from '@prml/math';
import { Axes, ContourField, CovarianceEllipse, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SHAPE1 = 2;
const SHAPE2 = 5;
const GRID = linspace(0.1, 10, 90);

const LAPLACE = laplaceApproximation(
  (z) => [(SHAPE1 - 1) / z[0]! - 1, (SHAPE2 - 1) / z[1]! - 1],
  (z) => [
    [-(SHAPE1 - 1) / (z[0]! * z[0]!), 0],
    [0, -(SHAPE2 - 1) / (z[1]! * z[1]!)],
  ],
  [Math.max(SHAPE1 - 1, 0.5), Math.max(SHAPE2 - 1, 0.5)],
);

const DENSITY = evalGrid(GRID, GRID, (x, y) => gammaPdf(x, { shape: SHAPE1, rate: 1 }) * gammaPdf(y, { shape: SHAPE2, rate: 1 }));


export default function Laplace2DContours() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={280} xDomain={[0, 10]} yDomain={[0, 10]} equalAspect label="A skewed 2D density with its Laplace-fitted Gaussian ellipse">
      <ContourField data={DENSITY} levelCount={8} color={tokens.color.inkMuted} />
      <Axes x={{ label: 'z₁' }} y={{ label: 'z₂' }} grid />
      <CovarianceEllipse mean={LAPLACE.mode} cov={LAPLACE.covariance} levels={[0.8]} color={tokens.color.accent} width={2} dash="dashed" />
    </Plot>
  );
}
