import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const K = 2;
const N_VALUES = Array.from({ length: 15 }, (_, i) => i + 2);

export default function ComplexityCurve() {
  const tokens = useResolvedTokens();
  const bruteForce: [number, number][] = N_VALUES.map((n) => [n, K ** n]);
  const sumProductCost: [number, number][] = N_VALUES.map((n) => [n, n * K ** 2]);

  return (
    <Plot
      width={340}
      height={220}
      xDomain={[2, 16]}
      yDomain={[1, K ** 16]}
      yScaleKind="log"
      label="States enumerated by brute force against messages sent by sum-product, both for binary variables in a chain"
    >
      <Axes x={{ label: 'chain length N' }} y={{ label: 'work (log scale)' }} grid />
      <Curve points={bruteForce} color={tokens.color.danger} width={2} />
      <Curve points={sumProductCost} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
