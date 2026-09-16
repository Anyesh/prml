import { flattenWeights, mvnCovarianceEllipse, submatrix, symmetrise } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { A_INVERSE, W_MAP } from '../bayesianToyNetwork';
import '../../widgets.css';

const PAIR: readonly [number, number] = [0, 1];
const FLAT = flattenWeights(W_MAP);
const CENTRE: [number, number] = [FLAT[PAIR[0]]!, FLAT[PAIR[1]]!];
const SUB_COV = symmetrise(submatrix(A_INVERSE, [...PAIR], [...PAIR]));
const ELLIPSE = mvnCovarianceEllipse({ mean: CENTRE, cov: SUB_COV }, 0.8);

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, angle: number): (readonly [number, number])[] {
  const n = 64;
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle)] as const;
  });
}

export default function PosteriorEllipseInWeightSpace() {
  const tokens = useResolvedTokens();
  const bound = Math.max(ELLIPSE.rx, ELLIPSE.ry) * 1.6;

  return (
    <Plot height={220} xDomain={[ELLIPSE.cx - bound, ELLIPSE.cx + bound]} yDomain={[ELLIPSE.cy - bound, ELLIPSE.cy + bound]} equalAspect label="Laplace-approximated posterior over two of the network's weights">
      <Axes x={{ label: 'w (index 0)' }} y={{ label: 'w (index 1)' }} grid />
      <Curve points={ellipsePoints(ELLIPSE.cx, ELLIPSE.cy, ELLIPSE.rx, ELLIPSE.ry, ELLIPSE.angle)} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
