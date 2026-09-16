import { binomialPmf, logBeta, logBinomialCoefficient } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N = 10;
const OUTCOMES = Array.from({ length: N + 1 }, (_, k) => k);
const MARKED = [6, 8];

function fairEvidence(k: number): number {
  return binomialPmf(k, { n: N, mu: 0.5 });
}

/**
 * Beta-Binomial marginal with a flat Beta(1,1) prior over the bias: PRML's coin-flip
 * worked example generalised to every possible outcome rather than one fixed count.
 */
function uniformEvidence(k: number): number {
  return Math.exp(logBinomialCoefficient(N, k) + logBeta(1 + k, 1 + N - k) - logBeta(1, 1));
}

export default function BayesFactorAcrossOutcomes() {
  const tokens = useResolvedTokens();

  const fair = OUTCOMES.map(fairEvidence);
  const uniform = OUTCOMES.map(uniformEvidence);
  const peak = Math.max(...fair, ...uniform);

  return (
    <div>
      <Plot
        height={220}
        xDomain={[0, N]}
        yDomain={[0, peak * 1.1]}
        label="Evidence for the fair-coin and uniform-prior models across every outcome"
      >
        <Axes x={{ label: 'heads out of 10', ticks: OUTCOMES }} y={{ label: 'evidence' }} grid />
        <Curve points={OUTCOMES.map((k, i) => [k, fair[i]!] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={OUTCOMES.map((k, i) => [k, uniform[i]!] as const)} color={tokens.series[1]!} width={2} />
        <ScatterField
          points={OUTCOMES.map((k, i) => ({
            x: k,
            y: fair[i]!,
            id: `f${k}`,
            color: tokens.series[0]!,
            size: MARKED.includes(k) ? 5.5 : 3,
          }))}
          label={(p) => `fair coin, ${p.x} heads`}
        />
        <ScatterField
          points={OUTCOMES.map((k, i) => ({
            x: k,
            y: uniform[i]!,
            id: `u${k}`,
            color: tokens.series[1]!,
            size: MARKED.includes(k) ? 5.5 : 3,
          }))}
          label={(p) => `uniform prior, ${p.x} heads`}
        />
        <Legend
          entries={[
            { label: 'fair coin (μ=0.5)', color: tokens.series[0]!, mark: 'dot' },
            { label: 'uniform prior over μ', color: tokens.series[1]!, mark: 'dot' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {`At 6 heads the fair coin wins (${fairEvidence(6).toFixed(3)} vs ${uniformEvidence(6).toFixed(3)}); `}
        {`at 8 heads the uniform prior wins instead (${fairEvidence(8).toFixed(3)} vs ${uniformEvidence(8).toFixed(3)}). `}
        {'The curves cross between 2 and 3 heads, and again between 7 and 8: past those points the extra parameter starts paying for itself.'}
      </p>
    </div>
  );
}
