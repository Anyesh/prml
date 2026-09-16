import { linspace } from '@prml/math';
import { Annotation, Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRID = linspace(-3, 3, 121);
const EPSILON = 0.3;
const H = 1e-4;

function bump(x: number): number {
  return Math.exp(-x * x);
}

function tangent(x: number): number {
  return -(bump(x + H) - bump(x - H)) / (2 * H);
}

const ORIGINAL = GRID.map((x) => [x, bump(x)] as const);
const TRUE_SHIFT = GRID.map((x) => [x, bump(x - EPSILON)] as const);
const TANGENT_APPROX = GRID.map((x) => [x, bump(x) + EPSILON * tangent(x)] as const);
const MAX_GAP = Math.max(...GRID.map((x) => Math.abs(bump(x - EPSILON) - (bump(x) + EPSILON * tangent(x)))));

export default function TangentVectorConcept() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={220} xDomain={[-3, 3]} yDomain={[-0.2, 1.1]} label="A shifted curve against its tangent-vector linear approximation">
      <Axes x={{ label: 'x' }} y={{ label: 's(x, ξ)' }} grid />
      <Curve points={ORIGINAL} color={tokens.color.inkFaint} width={1.5} />
      <Curve points={TRUE_SHIFT} color={tokens.color.accent} width={2} />
      <Curve points={TANGENT_APPROX} color={tokens.color.danger} width={1.5} dash="dashed" />
      <Annotation x={-2.9} y={1.0} anchor="start" size="xs" color={tokens.color.inkMuted} text={`max gap at ξ = ${EPSILON}: ${MAX_GAP.toExponential(2)}`} plate />
    </Plot>
  );
}
