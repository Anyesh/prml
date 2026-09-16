import { logGamma } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const D_VALUES = Array.from({ length: 40 }, (_, i) => i + 1);
const M_VALUES = [3, 9];

function logTotalParams(d: number, m: number): number {
  return logGamma(d + m + 1) - logGamma(d + 1) - logGamma(m + 1);
}

const SERIES = M_VALUES.map((m) => D_VALUES.map((d) => [d, logTotalParams(d, m) / Math.LN10] as const));
const AT_10 = Math.round(Math.exp(logTotalParams(10, 3)));
const AT_100 = Math.round(Math.exp(logTotalParams(100, 3)));

export default function PolynomialParameterGrowth() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[1, 40]} yDomain={[0, 14]} label="Total independent polynomial parameters against input dimension, log scale">
        <Axes x={{ label: 'D' }} y={{ label: 'log10 N(D,M)' }} grid />
        {SERIES.map((pts, i) => (
          <Curve key={M_VALUES[i]} points={pts} color={tokens.series[i % tokens.series.length]!} width={2} />
        ))}
        <Legend entries={M_VALUES.map((m, i) => ({ label: `degree M = ${m}`, color: tokens.series[i % tokens.series.length]!, mark: 'line' }))} placement="top-left" />
      </Plot>
      <p className="widget-readout">
        {`A cubic (M=3) polynomial needs ${AT_10.toLocaleString('en-US')} coefficients at D=10 and ${AT_100.toLocaleString('en-US')} at D=100, from N(D,M) = (D+M)!/(D!M!): no exponential in sight, and still unusable.`}
      </p>
    </div>
  );
}
