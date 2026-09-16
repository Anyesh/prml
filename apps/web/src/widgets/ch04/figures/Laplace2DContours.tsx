import { evalGrid, gammaPdf, laplaceApproximation, linspace, mvnCovarianceEllipse } from '@prml/math';
import { Axes, ContourField, Curve, Plot, useResolvedTokens } from '@prml/viz';
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

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, angle: number): (readonly [number, number])[] {
  const n = 64;
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle)] as const;
  });
}

export default function Laplace2DContours() {
  const tokens = useResolvedTokens();
  const ellipse = mvnCovarianceEllipse({ mean: LAPLACE.mode, cov: LAPLACE.covariance }, 0.8);

  return (
    <Plot height={280} xDomain={[0, 10]} yDomain={[0, 10]} equalAspect label="A skewed 2D density with its Laplace-fitted Gaussian ellipse">
      <ContourField data={DENSITY} levelCount={8} color={tokens.color.inkMuted} />
      <Axes x={{ label: 'z₁' }} y={{ label: 'z₂' }} grid />
      <Curve points={ellipsePoints(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry, ellipse.angle)} color={tokens.color.accent} width={2} dash="dashed" />
    </Plot>
  );
}
