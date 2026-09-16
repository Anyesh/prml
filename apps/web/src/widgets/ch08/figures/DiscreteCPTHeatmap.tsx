import { Axes, Heatmap, Plot, sequentialScale } from '@prml/viz';
import '../../widgets.css';

const VALUES = [
  [0.7, 0.1],
  [0.2, 0.3],
  [0.1, 0.6],
];

export default function DiscreteCPTHeatmap() {
  const colorAt = sequentialScale([0, 1]);
  return (
    <Plot width={220} height={220} xDomain={[0, 1]} yDomain={[0, 2]} equalAspect label="A conditional table: three child states against two parent states, each column summing to 1">
      <Heatmap data={{ xs: [0, 1], ys: [0, 1, 2], values: VALUES }} interpolator={colorAt} />
      <Axes x={{ label: 'parent state' }} y={{ label: 'child state' }} />
    </Plot>
  );
}
