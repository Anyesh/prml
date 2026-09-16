import { normalSample, pcg32 } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const TRUE_MU = 2;
const TRUE_SIGMA2 = 3;
const N = 400;
const TRUE_EX2 = TRUE_MU * TRUE_MU + TRUE_SIGMA2;

const rng = pcg32(20260916, 3);
let sumX = 0;
let sumX2 = 0;
const traceX: (readonly [number, number])[] = [];
const traceX2: (readonly [number, number])[] = [];
for (let n = 1; n <= N; n++) {
  const x = normalSample(rng, { mu: TRUE_MU, sigma2: TRUE_SIGMA2 });
  sumX += x;
  sumX2 += x * x;
  traceX.push([n, sumX / n]);
  traceX2.push([n, sumX2 / n]);
}

export default function SufficientStatisticConvergence() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[1, N]} yDomain={[0, 8]} label="Running averages of x and x-squared against their true expectations">
        <Axes x={{ label: 'N' }} y={{ label: 'running average of u(x)' }} grid />
        <Rule y={TRUE_MU} color={tokens.color.inkFaint} />
        <Rule y={TRUE_EX2} color={tokens.color.inkFaint} />
        <Curve points={traceX} color={tokens.series[0]!} width={1.75} />
        <Curve points={traceX2} color={tokens.series[1]!} width={1.75} />
        <Legend
          entries={[
            { label: 'mean of x → µ', color: tokens.series[0]!, mark: 'line' },
            { label: 'mean of x² → µ²+σ²', color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="bottom-right"
        />
      </Plot>
      <p className="widget-readout">
        {`400 draws from N(${TRUE_MU}, ${TRUE_SIGMA2}). Equation 2.228 says η_ML depends only on `}
        {'these two running sums, and both settle on the true moments without ever revisiting a raw sample.'}
      </p>
    </div>
  );
}
