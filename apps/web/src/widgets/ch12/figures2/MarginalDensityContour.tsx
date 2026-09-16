import { evalGrid, linspace, ppcaMLE, ppcaMarginalLogPdf } from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, VectorField, useResolvedTokens } from '@prml/viz';
import { ppcaDemoData } from '../data.js';

import '../../widgets.css';

const DATA = ppcaDemoData();
const PARAMS = ppcaMLE(DATA, 1);
const DOMAIN: readonly [number, number] = [-4, 4];
const GRID_AXIS = linspace(DOMAIN[0], DOMAIN[1], 60);
const GRID = evalGrid(GRID_AXIS, GRID_AXIS, (x, y) => ppcaMarginalLogPdf([x, y], PARAMS));
const LATENT_NORM = Math.hypot(PARAMS.w[0]![0]!, PARAMS.w[1]![0]!) || 1;

export default function MarginalDensityContour() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Log-density contours of the fitted marginal, with the latent axis drawn through the mean">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
      <ContourField data={GRID} levelCount={7} color={tokens.color.borderStrong} />
      <ScatterField points={DATA.map((p) => ({ x: p[0]!, y: p[1]! }))} color={tokens.color.inkMuted} size={2.5} opacity={0.6} />
      <VectorField
        origins={[[PARAMS.mean[0]!, PARAMS.mean[1]!]]}
        field={() => [(2.5 * PARAMS.w[0]![0]!) / LATENT_NORM, (2.5 * PARAMS.w[1]![0]!) / LATENT_NORM]}
        color={tokens.color.accent}
        maxLength={110}
      />
    </Plot>
  );
}
