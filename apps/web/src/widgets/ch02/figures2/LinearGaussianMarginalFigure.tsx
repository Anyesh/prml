import { linearGaussianMarginal, linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const PRIOR = { mean: [0], precision: [[1]] };
const LIKELIHOOD = { a: [[2]], b: [1], precision: [[4]] };
const MARGINAL = linearGaussianMarginal(PRIOR, LIKELIHOOD);

const GRID = linspace(-6, 6, 200);

export default function LinearGaussianMarginalFigure() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[-6, 6]} yDomain={[0, 0.5]} label="p(x) against the marginal p(y) after the linear-Gaussian map">
        <Axes x={{ label: 'value' }} y={{ label: 'density' }} grid />
        <Curve points={GRID.map((v) => [v, normalPdf(v, { mu: PRIOR.mean[0]!, sigma2: 1 / PRIOR.precision[0]![0]! })] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={GRID.map((v) => [v, normalPdf(v, { mu: MARGINAL.mean[0]!, sigma2: MARGINAL.cov[0]![0]! })] as const)} color={tokens.color.danger} width={2} />
        <Legend
          entries={[
            { label: 'p(x) = N(0, 1)', color: tokens.series[0]!, mark: 'line' },
            { label: 'p(y), marginalised', color: tokens.color.danger, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {`y = 2x + 1 + noise of variance 0.25. Equations 2.114-2.115 give p(y) mean `}
        {`${MARGINAL.mean[0]!.toFixed(2)} and variance ${MARGINAL.cov[0]![0]!.toFixed(2)}: the slope `}
        {'stretches x\'s own variance by 4 before the observation noise is even added.'}
      </p>
    </div>
  );
}
