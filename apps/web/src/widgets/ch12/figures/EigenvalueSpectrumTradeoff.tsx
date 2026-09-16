import { useMemo, useState } from 'react';
import { discardedEigenvalueSum, pcaFitCov } from '@prml/math';
import { Axes, Bars, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import { spectrumDemoData } from '../data.js';

import '../../widgets.css';

const PCA = pcaFitCov(spectrumDemoData());
const DIM = PCA.eigenvalues.length;
const TOTAL = PCA.eigenvalues.reduce((a, b) => a + b, 0);

export default function EigenvalueSpectrumTradeoff() {
  const tokens = useResolvedTokens();
  const [m, setM] = useState(2);

  const error = useMemo(() => discardedEigenvalueSum(PCA.eigenvalues, m), [m]);

  return (
    <div>
      <Plot height={220} xDomain={[-0.7, DIM - 0.3]} yDomain={[0, PCA.eigenvalues[0]! * 1.1]} label="Eigenvalue spectrum, largest first">
        <Axes x={{ label: 'component index', ticks: PCA.eigenvalues.map((_, i) => i) }} y={{ label: 'eigenvalue' }} grid />
        <Bars
          bars={PCA.eigenvalues.map((v, i) => ({ at: i, value: v, color: i < m ? tokens.color.accent : tokens.color.inkFaint }))}
          thickness={0.6}
        />
        <Rule x={m - 0.5} color={tokens.color.borderStrong} dash />
      </Plot>
      <Slider label="Components kept (M)" value={m} onChange={setM} min={0} max={DIM} step={1} />
      <p className="widget-readout">
        Keeping {m} of {DIM} axes leaves a reconstruction error of {error.toFixed(3)}, exactly the sum of the {DIM - m} faded bars
        ({((100 * error) / TOTAL).toFixed(1)}% of the total variance).
      </p>
    </div>
  );
}
