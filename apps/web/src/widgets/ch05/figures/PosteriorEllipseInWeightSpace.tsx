import { flattenWeights, mvnCovarianceEllipse, submatrix, symmetrise } from '@prml/math';
import { Axes, CovarianceEllipse, Plot, useResolvedTokens } from '@prml/viz';
import { A_INVERSE, W_MAP } from '../bayesianToyNetwork';
import '../../widgets.css';

const PAIR: readonly [number, number] = [0, 1];
const FLAT = flattenWeights(W_MAP);
const CENTRE: [number, number] = [FLAT[PAIR[0]]!, FLAT[PAIR[1]]!];
const SUB_COV = symmetrise(submatrix(A_INVERSE, [...PAIR], [...PAIR]));
const MASS = 0.8;
const ELLIPSE = mvnCovarianceEllipse({ mean: CENTRE, cov: SUB_COV }, MASS);

export default function PosteriorEllipseInWeightSpace() {
  const tokens = useResolvedTokens();
  const bound = Math.max(ELLIPSE.rx, ELLIPSE.ry) * 1.6;

  return (
    <Plot height={220} xDomain={[ELLIPSE.cx - bound, ELLIPSE.cx + bound]} yDomain={[ELLIPSE.cy - bound, ELLIPSE.cy + bound]} equalAspect label="Laplace-approximated posterior over two of the network's weights">
      <Axes x={{ label: 'w (index 0)' }} y={{ label: 'w (index 1)' }} grid />
      <CovarianceEllipse mean={CENTRE} cov={SUB_COV} levels={[MASS]} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
