import { pcaFitCov, whiten } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { baseCloud2D } from '../data.js';

import '../../widgets.css';

const RAW = baseCloud2D();
const CORRELATED = RAW.map(([x, y]) => [2.4 * x!, 0.5 * x! + 0.8 * y!]);
const PCA = pcaFitCov(CORRELATED);
const WHITENED = whiten(CORRELATED, PCA.mean, PCA.components, PCA.eigenvalues);

export default function WhiteningEffect() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-5, 5]} yDomain={[-5, 5]} equalAspect label="Correlated data before whitening">
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
        <ScatterField points={CORRELATED.map((p) => ({ x: p[0]!, y: p[1]! }))} color={tokens.color.inkMuted} size={3} />
      </Plot>
      <Plot height={220} xDomain={[-5, 5]} yDomain={[-5, 5]} equalAspect label="The same data after whitening">
        <Axes x={{ label: 'y1' }} y={{ label: 'y2' }} grid zeroLine />
        <ScatterField points={WHITENED.map((p) => ({ x: p[0]!, y: p[1]! }))} color={tokens.color.accent} size={3} />
      </Plot>
    </div>
  );
}
