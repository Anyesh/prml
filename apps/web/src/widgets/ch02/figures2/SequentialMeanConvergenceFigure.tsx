import { normalSample, pcg32, sequentialMean } from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const TRUE_MU = 3;
const SIGMA2 = 9;
const N = 200;

const rng = pcg32(20260916, 13);
let running = 0;
const trace: (readonly [number, number])[] = [];
for (let n = 1; n <= N; n++) {
  const x = normalSample(rng, { mu: TRUE_MU, sigma2: SIGMA2 });
  running = sequentialMean(running, n, x);
  trace.push([n, running]);
}

export default function SequentialMeanConvergenceFigure() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[1, N]} yDomain={[-2, 8]} label="Sequential ML mean estimate against number of observations">
        <Axes x={{ label: 'N' }} y={{ label: 'µ_ML^(N)' }} grid />
        <Rule y={TRUE_MU} color={tokens.color.inkFaint} />
        <Curve points={trace} color={tokens.color.accent} width={1.75} />
      </Plot>
      <p className="widget-readout">
        {`Equation 2.126 applied one point at a time to 200 draws from N(${TRUE_MU}, ${SIGMA2}), never revisiting a `}
        {`past sample. Final estimate ${trace[trace.length - 1]![1].toFixed(3)}, within the noise this sample size allows.`}
      </p>
    </div>
  );
}
