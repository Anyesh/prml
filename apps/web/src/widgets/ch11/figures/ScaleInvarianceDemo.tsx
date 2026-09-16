import { gaussianRandomWalkProposal, metropolisHastingsChain, normalLogPdf, pcg32 } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const LOG_SCALE_OFFSET = 7.3;

export default function ScaleInvarianceDemo() {
  const tokens = useResolvedTokens();
  const proposal = gaussianRandomWalkProposal(0.6);

  const properlyNormalized = metropolisHastingsChain(pcg32(2026, 300), [0], 400, {
    ...proposal,
    targetLogPdf: (z) => normalLogPdf(z[0]!, { mu: 0, sigma2: 1 }),
  });
  const arbitraryScale = metropolisHastingsChain(pcg32(2026, 300), [0], 400, {
    ...proposal,
    targetLogPdf: (z) => normalLogPdf(z[0]!, { mu: 0, sigma2: 1 }) + LOG_SCALE_OFFSET,
  });

  const maxDifference = Math.max(...properlyNormalized.states.map((s, i) => Math.abs(s[0]! - arbitraryScale.states[i]![0]!)));

  return (
    <div className="widget-grid">
      <Plot width={420} height={200} xDomain={[0, 400]} yDomain={[-4, 4]} label="Two Metropolis-Hastings chains, one on a normalized target and one on it multiplied by e^7.3">
        <Axes x={{ label: 'step' }} y={{ label: 'z' }} grid />
        <Curve points={properlyNormalized.states.map((s, i) => [i, s[0]!] as const)} color={tokens.series[0]!} width={2.5} />
        <Curve points={arbitraryScale.states.map((s, i) => [i, s[0]!] as const)} color={tokens.series[1]!} width={1} dash="dotted" />
      </Plot>
      <p className="widget-readout">
        Multiplying the target by e^7.3 before sampling changes nothing about the chain: the largest difference
        between the two runs above, step for step, is {maxDifference === 0 ? 'exactly zero' : maxDifference.toExponential(1)}.
      </p>
    </div>
  );
}
