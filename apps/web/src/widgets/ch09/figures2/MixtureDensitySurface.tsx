import { evalGrid, gmmPdf, linspace, type GmmParams } from '@prml/math';
import { Axes, ContourField, Heatmap, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

/** The same two-component mixture pinned in tools/golden/gen_gmm.py, so this surface and the prose numbers agree exactly. */
const PARAMS: GmmParams = {
  components: [
    { weight: 0.6, mean: [0, 0], cov: [[1, 0.3], [0.3, 1]] },
    { weight: 0.4, mean: [3, 3], cov: [[1.5, -0.2], [-0.2, 0.5]] },
  ],
};

const DOMAIN: readonly [number, number] = [-4, 6];
const GRID_AXIS = linspace(DOMAIN[0], DOMAIN[1], 64);
const DENSITY = evalGrid(GRID_AXIS, GRID_AXIS, (x, y) => gmmPdf([x, y], PARAMS));

let peak = 0;
for (const row of DENSITY.values) for (const v of row) if (v > peak) peak = v;
const FILL = sequentialScale([0, peak]);

export default function MixtureDensitySurface() {
  const tokens = useResolvedTokens();
  return (
    <Plot width={460} height={460} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Mixture density as a coloured field, with the two component means marked">
      <Heatmap data={DENSITY} interpolator={FILL} />
      <ContourField data={DENSITY} levelCount={6} color={tokens.color.plotBg} />
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
      <ScatterField
        points={PARAMS.components.map((c) => ({ x: c.mean[0]!, y: c.mean[1]!, color: tokens.color.plotBg, shape: 'cross', size: 8 }))}
      />
    </Plot>
  );
}
