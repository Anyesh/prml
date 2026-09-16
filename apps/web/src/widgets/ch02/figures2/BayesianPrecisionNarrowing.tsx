import { gammaPdf, linspace, normalSample, pcg32 } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const KNOWN_MEAN = 0;
const TRUE_LAMBDA = 2;
const PRIOR = { shape: 1, rate: 1 };
const NS = [0, 5, 20, 100];

const rng = pcg32(20260916, 19);
const DATA = Array.from({ length: 100 }, () => normalSample(rng, { mu: KNOWN_MEAN, sigma2: 1 / TRUE_LAMBDA }));

function posteriorAt(n: number) {
  const sumSq = DATA.slice(0, n).reduce((s, x) => s + (x - KNOWN_MEAN) ** 2, 0);
  return { shape: PRIOR.shape + n / 2, rate: PRIOR.rate + sumSq / 2 };
}

const GRID = linspace(0.01, 6, 200);

export default function BayesianPrecisionNarrowing() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[0, 6]} yDomain={[0, 1.6]} label="Posterior over the precision after 0, 5, 20, and 100 observations">
        <Axes x={{ label: 'λ' }} y={{ label: 'density' }} grid />
        <Rule x={TRUE_LAMBDA} color={tokens.color.inkFaint} />
        {NS.map((n, i) => {
          const post = posteriorAt(n);
          return <Curve key={n} points={GRID.map((l) => [l, gammaPdf(l, post)] as const)} color={tokens.series[i]!} width={1.75} />;
        })}
        <Legend entries={NS.map((n, i) => ({ label: `N=${n}`, color: tokens.series[i]!, mark: 'line' }))} placement="top-right" />
      </Plot>
      <p className="widget-readout">
        {`Gam(1,1) prior, data from a Gaussian of known mean and true precision 2. Equations 2.150-2.151 update `}
        {'shape and rate directly from the running sum of squares, tightening around the true precision.'}
      </p>
    </div>
  );
}
