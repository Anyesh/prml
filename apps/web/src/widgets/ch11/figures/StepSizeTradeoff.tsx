import { autocorrelation, gaussianRandomWalkProposal, metropolisHastingsChain, mvnLogPdf, pcg32 } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const MEAN = [0, 0];
const COV = [
  [1, 0.9],
  [0.9, 1],
];
const N_STEPS = 800;
const MAX_LAG = 30;
const STEP_SIZES = [0.03, 0.06, 0.1, 0.2, 0.4, 0.7, 1, 1.5, 2.5, 4, 6];

function integratedAutocorrelationTime(stepSize: number): { acceptanceRate: number; tau: number } {
  const rng = pcg32(7, 61);
  const proposal = gaussianRandomWalkProposal(stepSize);
  const chain = metropolisHastingsChain(rng, [0, 0], N_STEPS, { ...proposal, targetLogPdf: (z) => mvnLogPdf(z, { mean: MEAN, cov: COV }) });
  const rho = autocorrelation(
    chain.states.map((s) => s[0]!),
    MAX_LAG,
  );
  const tau = 1 + 2 * rho.slice(1).reduce((s, r) => s + Math.max(r, 0), 0);
  return { acceptanceRate: chain.acceptanceRate, tau };
}

export default function StepSizeTradeoff() {
  const tokens = useResolvedTokens();
  const results = STEP_SIZES.map((s) => ({ stepSize: s, ...integratedAutocorrelationTime(s) }));

  return (
    <div className="widget-grid">
      <Plot width={420} height={260} xDomain={[STEP_SIZES[0]!, STEP_SIZES[STEP_SIZES.length - 1]!]} xScaleKind="log" yDomain={[0, 1]} label="Acceptance rate against proposal step size, log-scaled x axis">
        <Axes x={{ label: 'step size (log scale)' }} y={{ label: 'acceptance rate' }} grid />
        <Curve points={results.map((r) => [r.stepSize, r.acceptanceRate] as const)} color={tokens.series[0]!} width={2} />
      </Plot>
      <Plot width={420} height={200} xDomain={[STEP_SIZES[0]!, STEP_SIZES[STEP_SIZES.length - 1]!]} xScaleKind="log" yScaleKind="log" yDomain={[1, 200]} label="Integrated autocorrelation time against proposal step size">
        <Axes x={{ label: 'step size (log scale)' }} y={{ label: 'autocorrelation time (log scale)' }} grid />
        <Curve points={results.map((r) => [r.stepSize, Math.max(r.tau, 1)] as const)} color={tokens.series[1]!} width={2} />
      </Plot>
      <p className="widget-readout">
        The best step size sits where neither curve is extreme: acceptance still reasonable, autocorrelation time not
        yet exploding. Both tails are bad for different reasons.
      </p>
    </div>
  );
}
