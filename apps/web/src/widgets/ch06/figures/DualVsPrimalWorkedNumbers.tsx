import {
  designMatrix,
  dot,
  dualPredict,
  dualRidgeCoefficients,
  gramMatrix,
  kernelFromFeatureMap,
  polynomialBasis,
  regularisedWeights,
} from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const XS = [0.05, 0.2, 0.35, 0.55, 0.75, 0.9];
const TS = [0.42, -0.1, -0.55, -0.3, 0.5, 0.88];
const DEGREE = 2;
const LAMBDA = 0.05;
const QUERY_POINTS = [0.3, 0.85];

const PHI = polynomialBasis(DEGREE, { bias: true });
const DESIGN = designMatrix(XS, PHI);
const W = regularisedWeights(DESIGN, TS, LAMBDA);

const KERNEL = kernelFromFeatureMap((v: readonly number[]) => PHI(v[0]!));
const TRAIN_VECS = XS.map((x) => [x]);
const GRAM = gramMatrix(KERNEL, TRAIN_VECS);
const A = dualRidgeCoefficients(GRAM, TS, LAMBDA);

const PREDICTIONS = QUERY_POINTS.map((x) => ({
  x,
  primal: dot(PHI(x), W),
  dual: dualPredict(KERNEL, TRAIN_VECS, A, [x]),
}));

export default function DualVsPrimalWorkedNumbers() {
  const tokens = useResolvedTokens();
  const maxAbsA = Math.max(...A.map(Math.abs));

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[-0.5, 5.5]} yDomain={[-maxAbsA * 1.2, maxAbsA * 1.2]} label="The six dual coefficients a_n, one per training point">
        <Axes x={{ label: 'training point index n', ticks: [0, 1, 2, 3, 4, 5] }} y={{ label: 'a_n' }} grid zeroLine />
        <Bars bars={A.map((v, i) => ({ at: i, value: v, color: tokens.series[0]! }))} thickness={0.6} />
      </Plot>
      <p className="widget-readout">
        {`Degree ${DEGREE} polynomial basis (M = ${W.length}), lambda = ${LAMBDA}, over ${XS.length} points. ` +
          PREDICTIONS.map(
            (p) =>
              `At x = ${p.x}: weight space gives ${p.primal.toFixed(4)}, dual space gives ${p.dual.toFixed(4)}.`,
          ).join(' ')}
      </p>
    </div>
  );
}
