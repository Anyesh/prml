import { linspace } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DOMAIN: readonly [number, number] = [-1, 3];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 80);
const XI_VALUES = [0.5, 1, 2];

function f(x: number): number {
  return Math.exp(-x);
}

function tangentAt(xi: number, x: number): number {
  return f(xi) - f(xi) * (x - xi);
}

export default function ConvexDualityTangent() {
  const tokens = useResolvedTokens();
  const curve = GRID.map((x) => [x, f(x)] as const);
  const tangents = XI_VALUES.map((xi) => GRID.map((x) => [x, tangentAt(xi, x)] as const));

  return (
    <Plot height={220} xDomain={DOMAIN} yDomain={[-0.5, 3]} label="exp(-x) and its tangent lines at three points, each a valid lower bound that is exact only where it touches">
      <Axes x={{ label: 'x' }} y={{ label: 'f(x)' }} grid zeroLine />
      <Curve points={curve} color={tokens.color.ink} width={2} />
      {tangents.map((pts, i) => (
        <Curve key={i} points={pts} color={tokens.series[i % tokens.series.length]!} width={1.5} dash="dashed" />
      ))}
    </Plot>
  );
}
