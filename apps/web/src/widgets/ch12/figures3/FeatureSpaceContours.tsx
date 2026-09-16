import { evalGrid, kernelPcaFit, kernelPcaProject, rbfKernelGamma, linspace } from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { ringsData } from '../data.js';

import '../../widgets.css';

const DATA = ringsData();
const MODEL = kernelPcaFit(DATA, rbfKernelGamma(0.25), 1);
const DOMAIN: readonly [number, number] = [-4, 4];
const GRID_AXIS = linspace(DOMAIN[0], DOMAIN[1], 50);
const GRID = evalGrid(GRID_AXIS, GRID_AXIS, (x, y) => kernelPcaProject(MODEL, [x, y])[0]!);

export default function FeatureSpaceContours() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Level sets of the first kernel component, drawn back in the original input space">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
      <ContourField data={GRID} levelCount={9} color={tokens.color.borderStrong} />
      <ScatterField points={DATA.map((p) => ({ x: p[0]!, y: p[1]! }))} color={tokens.color.inkMuted} size={3} />
    </Plot>
  );
}
