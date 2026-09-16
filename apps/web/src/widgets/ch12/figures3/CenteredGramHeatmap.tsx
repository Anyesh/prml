import { centerGramMatrix, gramMatrix, rbfKernelGamma } from '@prml/math';
import { divergingScale, Heatmap, Plot } from '@prml/viz';
import { ringsData } from '../data.js';

import '../../widgets.css';

const DATA = ringsData();
const K = gramMatrix(rbfKernelGamma(0.25), DATA);
const K_CENTERED = centerGramMatrix(K);
const N = DATA.length;
const MAX_ABS = Math.max(...K_CENTERED.flat().map((v) => Math.abs(v)));

export default function CenteredGramHeatmap() {
  const interpolator = divergingScale([-MAX_ABS, MAX_ABS]);

  return (
    <Plot height={260} width={260} xDomain={[0, N]} yDomain={[0, N]} equalAspect label="The centred Gram matrix: entry (i, j) is the centred kernel value between points i and j">
      <Heatmap data={{ xs: Array.from({ length: N }, (_, i) => i), ys: Array.from({ length: N }, (_, i) => i), values: K_CENTERED }} interpolator={interpolator} />
    </Plot>
  );
}
