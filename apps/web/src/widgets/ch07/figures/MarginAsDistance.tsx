import { Annotation, Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const W = [1, 0.6];
const B = -0.4;
const POINT: readonly [number, number] = [2.6, 3.1];

function boundaryLine(): readonly [number, number][] {
  return [-5, 5].map((x) => [x, -(B + W[0]! * x) / W[1]!] as const);
}

function footOfPerpendicular(x: number, y: number): readonly [number, number] {
  const wNormSq = W[0]! * W[0]! + W[1]! * W[1]!;
  const signedDistance = (W[0]! * x + W[1]! * y + B) / wNormSq;
  return [x - signedDistance * W[0]!, y - signedDistance * W[1]!];
}

export default function MarginAsDistance() {
  const tokens = useResolvedTokens();
  const [fx, fy] = footOfPerpendicular(POINT[0], POINT[1]);
  const distance = Math.hypot(POINT[0] - fx, POINT[1] - fy);

  return (
    <Plot height={240} xDomain={[-5, 5]} yDomain={[-4, 5]} equalAspect label="Perpendicular distance from a point to the decision surface">
      <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
      <Curve points={boundaryLine()} color={tokens.color.ink} width={2} />
      <Curve points={[[POINT[0], POINT[1]], [fx, fy]]} color={tokens.color.accent} width={1.5} dash="dashed" />
      <ScatterField points={[{ x: POINT[0], y: POINT[1], id: 'x', color: tokens.color.accent, size: 4.5 }]} />
      <ScatterField points={[{ x: fx, y: fy, id: 'foot', color: tokens.color.inkMuted, shape: 'cross', size: 4 }]} />
      <Annotation x={0.5 * (POINT[0] + fx)} y={0.5 * (POINT[1] + fy)} text={`|y(x)|/‖w‖ ≈ ${distance.toFixed(2)}`} color={tokens.color.accent} dy={-10} plate />
      <Annotation x={-3.5} y={boundarySample(-3.5)} text="y(x) = 0" color={tokens.color.inkMuted} dy={-10} plate />
    </Plot>
  );
}

function boundarySample(x: number): number {
  return -(B + W[0]! * x) / W[1]!;
}
