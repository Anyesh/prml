import { useMemo, useState } from 'react';
import { pcaFitCov, pcaProject, pcaReconstruct, pcaReconstructionError } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import { signalDemoData } from '../data.js';

import '../../widgets.css';

const { data: DATA } = signalDemoData();
const DIM = DATA[0]!.length;
const POSITIONS = Array.from({ length: DIM }, (_, i) => i);
const PCA = pcaFitCov(DATA);
const SAMPLE_INDEX = 0;
const SAMPLE = DATA[SAMPLE_INDEX]!;

export default function ReconstructionByComponents() {
  const tokens = useResolvedTokens();
  const [m, setM] = useState(1);

  const reconstructed = useMemo(() => {
    const scores = pcaProject([SAMPLE], PCA.mean, PCA.components, m);
    return pcaReconstruct(scores, PCA.mean, PCA.components)[0]!;
  }, [m]);
  const error = useMemo(() => pcaReconstructionError(DATA, PCA.mean, PCA.components, m), [m]);

  return (
    <div>
      <Plot height={220} xDomain={[0, DIM - 1]} yDomain={[-2.2, 2.2]} label="One sample curve, original against its reconstruction from M components">
        <Axes x={{ label: 'coordinate' }} y={{ label: 'value' }} grid zeroLine />
        <Curve points={POSITIONS.map((i) => [i, SAMPLE[i]!] as const)} color={tokens.color.inkMuted} width={1.5} />
        <Curve points={POSITIONS.map((i) => [i, reconstructed[i]!] as const)} color={tokens.color.accent} width={2} dash="dashed" />
        <Legend
          entries={[
            { label: 'original', color: tokens.color.inkMuted, mark: 'line' },
            { label: `reconstruction, M = ${m}`, color: tokens.color.accent, mark: 'dashed-line' },
          ]}
        />
      </Plot>
      <Slider label="Components kept (M)" value={m} onChange={setM} min={1} max={DIM} step={1} />
      <p className="widget-readout">Mean squared reconstruction error across all samples: {error.toFixed(4)}.</p>
    </div>
  );
}
