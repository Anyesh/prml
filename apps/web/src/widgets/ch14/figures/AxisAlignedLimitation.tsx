import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const STEPS = [
  [-2, -1.6], [-1.6, -1.2], [-1.2, -0.8], [-0.8, -0.4], [-0.4, 0], [0, 0.4], [0.4, 0.8], [0.8, 1.2], [1.2, 1.6], [1.6, 2],
] as const;

export default function AxisAlignedLimitation() {
  const tokens = useResolvedTokens();
  const staircase = STEPS.flatMap(([a, b]) => [[a, a] as const, [a, b] as const, [b, b] as const]);

  return (
    <Plot height={200} xDomain={[-2, 2]} yDomain={[-2, 2]} equalAspect label="A diagonal boundary approximated by many axis-aligned steps, against the single oblique split it actually needs">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
      <Curve points={[[-2, -2], [2, 2]]} color={tokens.color.borderStrong} width={2} dash="dashed" />
      <Curve points={staircase} color={tokens.series[0]!} width={1.5} />
    </Plot>
  );
}
