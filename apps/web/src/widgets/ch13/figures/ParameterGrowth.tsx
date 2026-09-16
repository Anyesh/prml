import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';

const K = 4;
const MAX_ORDER = 6;

export default function ParameterGrowth() {
  const tokens = useResolvedTokens();
  const orders = Array.from({ length: MAX_ORDER }, (_, i) => i + 1);
  const tablePoints = orders.map((m) => [m, Math.pow(K, m - 1) * (K - 1)] as const);
  const linearPoints = orders.map((m) => [m, m * K * (K - 1)] as const);

  return (
    <Plot width={360} height={220} xDomain={[1, MAX_ORDER]} yDomain={[0, tablePoints[tablePoints.length - 1]![1] * 1.05]} label="Parameter count against Markov order">
      <Axes x={{ label: 'order M' }} y={{ label: 'free parameters' }} grid />
      <Curve points={tablePoints} color={tokens.color.accent} width={2} />
      <Curve points={linearPoints} color={tokens.color.inkMuted} width={1.5} dash="dashed" />
    </Plot>
  );
}
