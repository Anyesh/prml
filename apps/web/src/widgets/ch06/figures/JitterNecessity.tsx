import { cholesky, gramMatrix, jitter, linspace, rbfKernel, svd } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const LENGTH_SCALE = 0.2;
const KERNEL = rbfKernel(LENGTH_SCALE);
const BASE_POINTS = linspace(0, 1, 9);
const DELTAS = linspace(-6, -0.3, 40).map((logDelta) => Math.pow(10, logDelta));

function pointsWithGap(delta: number) {
  return [...BASE_POINTS, BASE_POINTS[4]! + delta].map((x) => [x]);
}

function conditionNumber(matrix: readonly (readonly number[])[]): number {
  const s = svd(matrix).s;
  return s[0]! / s[s.length - 1]!;
}

const RESULTS = DELTAS.map((delta) => {
  const gram = gramMatrix(KERNEL, pointsWithGap(delta));
  const logCond = Math.log10(conditionNumber(gram));
  let holdsWithoutJitter = true;
  try {
    cholesky(gram);
  } catch {
    holdsWithoutJitter = false;
  }
  let holdsWithJitter = true;
  try {
    cholesky(jitter(gram, 1e-6));
  } catch {
    holdsWithJitter = false;
  }
  return { delta, logCond, holdsWithoutJitter, holdsWithJitter };
});

const FIRST_FAILURE = RESULTS.find((r) => !r.holdsWithoutJitter);

export default function JitterNecessity() {
  const tokens = useResolvedTokens();
  const curve = RESULTS.map((r) => [Math.log10(r.delta), r.logCond] as const);
  const maxLog = Math.max(...RESULTS.map((r) => r.logCond));

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-6, -0.3]} yDomain={[0, maxLog + 1]} label="Condition number of the Gram matrix as a tenth point closes in on an existing one">
        <Axes x={{ label: 'log10(gap between the two closest points)' }} y={{ label: 'condition number', format: (v) => `1e${Math.round(v)}` }} grid />
        {FIRST_FAILURE && (
          <Rule x={Math.log10(FIRST_FAILURE.delta)} color={tokens.color.danger} label="unjittered Cholesky starts failing here" />
        )}
        <Curve points={curve} color={tokens.color.accent} width={2} />
        <Legend entries={[{ label: 'condition number of K', color: tokens.color.accent, mark: 'line' }]} placement="top-left" />
      </Plot>
      <p className="widget-readout">
        {FIRST_FAILURE
          ? `Below a gap of ${FIRST_FAILURE.delta.toExponential(1)}, Cholesky on the raw Gram matrix throws outright: the two closest points have become numerically indistinguishable. Adding 1e-6 to the diagonal (jitter) keeps every single case in this sweep factorisable.`
          : 'No failures in this sweep, though the condition number is already climbing sharply.'}
      </p>
    </div>
  );
}
