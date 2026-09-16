import { pcg32, standardNormal, variance } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const TRUE_SIGMA2 = 1;
const TRIALS = 20000;
const N_VALUES = [2, 5, 10, 20, 50];

function averageMlVariance(n: number, streamId: number): number {
  const rng = pcg32(20260701, streamId);
  let sum = 0;
  for (let t = 0; t < TRIALS; t++) {
    const xs = Array.from({ length: n }, () => Math.sqrt(TRUE_SIGMA2) * standardNormal(rng));
    sum += variance(xs);
  }
  return sum / TRIALS;
}

const EMPIRICAL = N_VALUES.map((n, i) => averageMlVariance(n, i + 1));
const THEORY = N_VALUES.map((n) => ((n - 1) / n) * TRUE_SIGMA2);

export default function MLVarianceBias() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 52]} yDomain={[0, 1.1]} label="Average maximum-likelihood variance estimate against sample size">
        <Axes x={{ label: 'N' }} y={{ label: 'E[sigma_ML^2]' }} grid />
        <Rule y={TRUE_SIGMA2} color={tokens.color.danger} label="true variance" />
        <Curve points={N_VALUES.map((n, i) => [n, THEORY[i]!] as const)} color={tokens.series[1]!} dash="dashed" width={1.5} />
        <ScatterField points={N_VALUES.map((n, i) => ({ x: n, y: EMPIRICAL[i]!, id: n }))} color={tokens.series[0]!} size={4.5} />
        <Legend
          entries={[
            { label: 'empirical mean of 20000 fits', color: tokens.series[0]!, mark: 'dot' },
            { label: '(N-1)/N, PRML 1.58', color: tokens.series[1]!, mark: 'dashed-line' },
          ]}
          placement="bottom-right"
        />
      </Plot>
      <p className="widget-readout">
        {`At N=${N_VALUES[0]}: mean estimate ${EMPIRICAL[0]!.toFixed(3)} against a true variance of 1 and a predicted ${THEORY[0]!.toFixed(3)}. The gap shrinks steadily and both curves approach 1 as N grows.`}
      </p>
    </div>
  );
}
