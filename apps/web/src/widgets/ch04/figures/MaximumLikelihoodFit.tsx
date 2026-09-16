import { fitSharedCovarianceGaussian, mvnCovarianceEllipse, sharedCovarianceLinearBoundary } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const CLASS1 = [
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
];
const CLASS2 = [
  [3, 3],
  [4, 3],
  [3, 4],
  [4, 4],
];

const FIT = fitSharedCovarianceGaussian([CLASS1, CLASS2]);
const BOUNDARY = sharedCovarianceLinearBoundary(FIT);
const DOMAIN: readonly [number, number] = [-1.5, 5.5];

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, angle: number): (readonly [number, number])[] {
  const n = 64;
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle)] as const;
  });
}

export default function MaximumLikelihoodFit() {
  const tokens = useResolvedTokens();
  const ellipse1 = mvnCovarianceEllipse({ mean: FIT.means[0]!, cov: FIT.covariance }, 0.8);
  const ellipse2 = mvnCovarianceEllipse({ mean: FIT.means[1]!, cov: FIT.covariance }, 0.8);
  const [w0, w1, w2] = [BOUNDARY.w0, BOUNDARY.w[0]!, BOUNDARY.w[1]!];
  const boundaryLine = DOMAIN.map((x) => [x, -(w0 + w1 * x) / w2] as const);

  return (
    <Plot height={260} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Eight points, their fitted shared-covariance Gaussians, and the resulting boundary">
      <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
      <Curve points={ellipsePoints(ellipse1.cx, ellipse1.cy, ellipse1.rx, ellipse1.ry, ellipse1.angle)} color={tokens.series[0]!} width={1.5} />
      <Curve points={ellipsePoints(ellipse2.cx, ellipse2.cy, ellipse2.rx, ellipse2.ry, ellipse2.angle)} color={tokens.series[1]!} width={1.5} />
      <Curve points={boundaryLine} color={tokens.color.ink} width={2} dash="dashed" />
      <ScatterField points={CLASS1.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `1-${i}` }))} color={tokens.series[0]} size={5} />
      <ScatterField points={CLASS2.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `2-${i}` }))} color={tokens.series[1]} size={5} />
    </Plot>
  );
}
