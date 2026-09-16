import { linspace, normalPdf, studentTPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SIGMA2 = 1;
const NU = 3;
const SCALE2 = (SIGMA2 * (NU - 2)) / NU;
const T_PARAMS = { mu: 0, scale2: SCALE2, nu: NU };
const G_PARAMS = { mu: 0, sigma2: SIGMA2 };

const GRID = linspace(-8, 8, 240);
const PROBE = 4;
const RATIO = studentTPdf(PROBE, T_PARAMS) / normalPdf(PROBE, G_PARAMS);

export default function StudentTHeavyTail() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[-8, 8]} yDomain={[0.0001, 0.4]} label="Student's t against a Gaussian of the same variance">
        <Axes x={{ label: 'x' }} y={{ label: 'density' }} grid />
        <Curve points={GRID.map((x) => [x, normalPdf(x, G_PARAMS)] as const)} color={tokens.color.inkFaint} width={1.75} />
        <Curve points={GRID.map((x) => [x, studentTPdf(x, T_PARAMS)] as const)} color={tokens.color.accent} width={2} />
        <Legend
          entries={[
            { label: 'Gaussian, var=1', color: tokens.color.inkFaint, mark: 'line' },
            { label: `Student's t, ν=${NU}, same variance`, color: tokens.color.accent, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {`Both have variance 1. At x=${PROBE}, the t-density is ${RATIO.toFixed(1)}× the Gaussian's: an outlier that far out `}
        {'is unremarkable under t and a near-impossibility under the Gaussian, which is why t survives contamination.'}
      </p>
    </div>
  );
}
