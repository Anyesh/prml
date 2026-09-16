import { linspace, logisticLocalBound, sigmoid } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DOMAIN: readonly [number, number] = [-6, 6];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 120);
const XI_VALUES = [1.5, 3.5];

export default function TangentGapVisualization() {
  const tokens = useResolvedTokens();
  const gaps = XI_VALUES.map((xi) => GRID.map((x) => [x, sigmoid(x) - logisticLocalBound(x, xi)] as const));

  return (
    <Plot height={200} xDomain={DOMAIN} yDomain={[0, 0.15]} label="How far the bound sits below the true sigmoid, for two choices of xi">
      <Axes x={{ label: 'x' }} y={{ label: 'sigma(x) - bound(x)' }} grid />
      {gaps.map((pts, i) => (
        <Curve key={i} points={pts} color={tokens.series[i % tokens.series.length]!} width={2} />
      ))}
      <Legend entries={XI_VALUES.map((xi, i) => ({ label: `xi = ${xi}`, color: tokens.series[i % tokens.series.length]!, mark: 'line' as const }))} />
    </Plot>
  );
}
