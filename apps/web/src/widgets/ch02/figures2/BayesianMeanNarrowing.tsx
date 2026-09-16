import { linspace, normalPdf, normalSample, pcg32 } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const TRUE_MU = 0.8;
const SIGMA2 = 0.1;
const PRIOR_MU0 = 0;
const PRIOR_SIGMA2_0 = SIGMA2;
const NS = [0, 1, 2, 10];

const rng = pcg32(20260916, 17);
const DATA = Array.from({ length: 10 }, () => normalSample(rng, { mu: TRUE_MU, sigma2: SIGMA2 }));

function posteriorAt(n: number) {
  if (n === 0) return { mu: PRIOR_MU0, sigma2: PRIOR_SIGMA2_0 };
  const sample = DATA.slice(0, n);
  const mlMean = sample.reduce((a, b) => a + b, 0) / n;
  const sigma2N = 1 / (1 / PRIOR_SIGMA2_0 + n / SIGMA2);
  const muN =
    (SIGMA2 / (n * PRIOR_SIGMA2_0 + SIGMA2)) * PRIOR_MU0 + (n * PRIOR_SIGMA2_0 / (n * PRIOR_SIGMA2_0 + SIGMA2)) * mlMean;
  return { mu: muN, sigma2: sigma2N };
}

const GRID = linspace(-1, 2, 200);

export default function BayesianMeanNarrowing() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[-1, 2]} yDomain={[0, 6]} label="Posterior over the mean after 0, 1, 2, and 10 observations">
        <Axes x={{ label: 'µ' }} y={{ label: 'density' }} grid />
        {NS.map((n, i) => {
          const post = posteriorAt(n);
          return <Curve key={n} points={GRID.map((m) => [m, normalPdf(m, { mu: post.mu, sigma2: post.sigma2 })] as const)} color={tokens.series[i]!} width={1.75} />;
        })}
        <Legend entries={NS.map((n, i) => ({ label: `N=${n}`, color: tokens.series[i]!, mark: 'line' }))} placement="top-left" />
      </Plot>
      <p className="widget-readout">
        {`Prior N(0, ${SIGMA2}), data drawn from N(${TRUE_MU}, ${SIGMA2}). Equation 2.142 adds precision every point, `}
        {'so each curve is strictly narrower than the last, converging on the true mean.'}
      </p>
    </div>
  );
}
