import { evalGrid, eigSym, linspace, mvnLogPdf } from '@prml/math';
import { Axes, ContourField, Plot, VectorField, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const MEAN = [0, 0];
const COV = [
  [3, 1.2],
  [1.2, 1],
];
const GRID = linspace(-4, 4, 70);
const EIGEN = eigSym(COV);
const [L1, L2] = EIGEN.values as [number, number];
const [U1, U2] = EIGEN.vectors as [number[], number[]];

const DENSITY = evalGrid(GRID, GRID, (x, y) => Math.exp(mvnLogPdf([x, y], { mean: MEAN, cov: COV })));

export default function EigenDecomposition() {
  const tokens = useResolvedTokens();
  const fill = sequentialScale([0, Math.exp(mvnLogPdf(MEAN, { mean: MEAN, cov: COV }))], tokens.sequential);
  return (
    <div>
      <Plot height={240} xDomain={[-4, 4]} yDomain={[-4, 4]} equalAspect label="Covariance ellipse with its two eigenvector axes">
        <ContourField data={DENSITY} levelCount={6} color={tokens.color.inkFaint} fill={fill} z={-1} />
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
        <VectorField origins={[MEAN as [number, number]]} field={() => [U1[0]!, U1[1]!] as const} color={tokens.color.danger} maxLength={40 * Math.sqrt(L1 / L1)} />
        <VectorField origins={[MEAN as [number, number]]} field={() => [U2[0]!, U2[1]!] as const} color={tokens.color.danger} maxLength={40 * Math.sqrt(L2 / L1)} />
      </Plot>
      <p className="widget-readout">
        {`Eigenvalues λ1=${L1.toFixed(2)}, λ2=${L2.toFixed(2)}. The contour is an ellipse with axes along u1 and u2 `}
        {`(2.45), lengths proportional to √λ1 and √λ2 (2.50): the direction of largest variance is the long axis, `}
        {'not an axis of the coordinate system.'}
      </p>
    </div>
  );
}
