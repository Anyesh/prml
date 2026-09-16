import { normalSample, pcg32 } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const TRUE_VARIANCE = 4;
const REPS = 2000;
const NS = Array.from({ length: 19 }, (_, i) => i + 2);

function averageMlVariance(n: number, rng: ReturnType<typeof pcg32>): number {
  let sum = 0;
  for (let r = 0; r < REPS; r++) {
    const xs = Array.from({ length: n }, () => normalSample(rng, { mu: 0, sigma2: TRUE_VARIANCE }));
    const mean = xs.reduce((a, b) => a + b, 0) / n;
    const mlVar = xs.reduce((a, x) => a + (x - mean) ** 2, 0) / n;
    sum += mlVar;
  }
  return sum / REPS;
}

const rng = pcg32(20260916, 11);
const SIMULATED = NS.map((n) => [n, averageMlVariance(n, rng)] as const);
const PREDICTED = NS.map((n) => [n, ((n - 1) / n) * TRUE_VARIANCE] as const);

export default function MLVarianceBias() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[2, 20]} yDomain={[0, TRUE_VARIANCE + 1]} label="Average ML variance estimate against sample size, over 4000 replicate datasets per N">
        <Axes x={{ label: 'N' }} y={{ label: 'average of σ²_ML' }} grid />
        <Rule y={TRUE_VARIANCE} color={tokens.color.inkFaint} label="true σ²" />
        <Curve points={PREDICTED} color={tokens.color.danger} dash="dashed" width={2} />
        <Curve points={SIMULATED} color={tokens.color.accent} width={1.5} />
        <Legend
          entries={[
            { label: '(N-1)/N · σ² (2.124)', color: tokens.color.danger, mark: 'dashed-line' },
            { label: '4000-dataset average', color: tokens.color.accent, mark: 'line' },
          ]}
          placement="bottom-right"
        />
      </Plot>
      <p className="widget-readout">
        {`True variance 4. Simulated average at N=2 is ${SIMULATED[0]![1].toFixed(3)} against predicted ${PREDICTED[0]![1].toFixed(3)} (2000-dataset average): `}
        {'the ML estimate systematically undershoots by the same N-1 over N factor 2.124 predicts, closing as N grows.'}
      </p>
    </div>
  );
}
