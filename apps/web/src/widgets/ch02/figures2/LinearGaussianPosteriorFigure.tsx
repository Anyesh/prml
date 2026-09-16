import { linearGaussianPosterior, linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const PRIOR = { mean: [0], precision: [[1]] };
const LIKELIHOOD = { a: [[2]], b: [1], precision: [[4]] };
const OBSERVATIONS = [-2, 0, 2];
const POSTERIORS = OBSERVATIONS.map((y) => linearGaussianPosterior(PRIOR, LIKELIHOOD, [y]));

const GRID = linspace(-3, 3, 200);

export default function LinearGaussianPosteriorFigure() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[-3, 3]} yDomain={[0, 1.5]} label="p(x|y) for three observed values of y">
        <Axes x={{ label: 'x' }} y={{ label: 'density' }} grid />
        {POSTERIORS.map((post, i) => (
          <Curve
            key={i}
            points={GRID.map((x) => [x, normalPdf(x, { mu: post.mean[0]!, sigma2: post.cov[0]![0]! })] as const)}
            color={tokens.series[i]!}
            width={2}
          />
        ))}
        <Legend entries={OBSERVATIONS.map((y, i) => ({ label: `y=${y}`, color: tokens.series[i]!, mark: 'line' }))} placement="top-right" />
      </Plot>
      <p className="widget-readout">
        {`Equations 2.116-2.117: the three posteriors share one variance, ${POSTERIORS[0]!.cov[0]![0]!.toFixed(3)}, `}
        {'because it never depends on y at all, only the mean does. Observing y only ever moves where the belief sits, not how tight it is.'}
      </p>
    </div>
  );
}
