import { betaPdf, linspace, type BetaParams } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const X = linspace(0.005, 0.995, 200);
const SHAPES: readonly BetaParams[] = [
  { a: 0.1, b: 0.1 },
  { a: 1, b: 1 },
  { a: 2, b: 3 },
  { a: 8, b: 4 },
];

export default function BetaShapes() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[0, 1]} yDomain={[0, 3]} label="Beta density for four hyperparameter pairs">
        <Axes x={{ label: 'µ' }} y={{ label: 'density' }} grid />
        {SHAPES.map((p, i) => (
          <Curve key={i} points={X.map((x) => [x, Math.min(betaPdf(x, p), 3)] as const)} color={tokens.series[i]!} width={1.75} />
        ))}
        <Legend
          entries={SHAPES.map((p, i) => ({ label: `a=${p.a}, b=${p.b}`, color: tokens.series[i]!, mark: 'line' }))}
          placement="top-right"
        />
      </Plot>
    </div>
  );
}
