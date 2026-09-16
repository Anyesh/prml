import { gammaLogPdf, pcg32, samplingImportanceResampling, exponentialLogPdf, exponentialSample } from '@prml/math';
import { Annotation, Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SHAPE = 3;
const L = 24;

export default function SirResampling() {
  const tokens = useResolvedTokens();
  const rng = pcg32(2024, 3);
  const result = samplingImportanceResampling(rng, {
    sampleProposal: (r) => exponentialSample(r, 1),
    proposalLogPdf: (z) => exponentialLogPdf(z, 1),
    targetLogPdf: (z) => gammaLogPdf(z, { shape: SHAPE, rate: 1 }),
    l: L,
  });
  const maxWeight = Math.max(...result.weights);

  return (
    <div className="widget-grid">
      <Plot width={420} height={220} xDomain={[0, 14]} yDomain={[-0.5, 1.5]} label="Proposal draws sized by importance weight, above the resampled set drawn from those weights">
        <Axes x={{ label: 'z' }} y={false} grid />
        <ScatterField
          points={result.proposalSamples.map((z, i) => ({ x: z, y: 1, color: tokens.series[0]!, size: 2 + 16 * (result.weights[i]! / maxWeight) }))}
        />
        <ScatterField points={result.resampled.map((z) => ({ x: z, y: 0, color: tokens.series[1]!, size: 5 }))} />
        <Annotation x={13.5} y={1} text="proposal, sized by weight" anchor="end" dy={-10} size="xs" />
        <Annotation x={13.5} y={0} text="resampled set" anchor="end" dy={-10} size="xs" />
      </Plot>
      <p className="widget-readout">
        Proposal Exp(1), target Gam(z|3,1): the resampled row is the second-stage draw from the top row's weights,
        so points the proposal rarely favoured all but vanish (PRML 11.1.5).
      </p>
    </div>
  );
}
