import { evalGrid, linspace } from '@prml/math';
import { Axes, ContourField, Plot, useResolvedTokens } from '@prml/viz';
import { mdnComponentDensity } from '../mdnToyProblem';
import '../../widgets.css';

const T_GRID = linspace(0.05, 0.95, 60);
const X_GRID = linspace(0, 1, 60);
const DENSITY = evalGrid(T_GRID, X_GRID, (t, x) => mdnComponentDensity(t, x));

export default function ConditionalDensityContour() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={260} xDomain={[0, 1]} yDomain={[0, 1]} equalAspect label="The mixture's predictive density p(x | t) over the input-target plane">
      <ContourField data={DENSITY} levelCount={10} color={tokens.color.accent} />
      <Axes x={{ label: 't' }} y={{ label: 'x' }} grid />
    </Plot>
  );
}
