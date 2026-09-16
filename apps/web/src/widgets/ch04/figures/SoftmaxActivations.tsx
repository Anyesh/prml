import { linspace, softmax } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DOMAIN: readonly [number, number] = [-5, 5];
const GRID = linspace(-5, 5, 200);

const ACTIVATIONS = [
  (x: number) => 1.2 * x,
  (x: number) => -0.9 * x + 1,
  (x: number) => 0.3 * x - 2.5,
];

export default function SoftmaxActivations() {
  const tokens = useResolvedTokens();
  const activationCurves = ACTIVATIONS.map((a) => GRID.map((x) => [x, a(x)] as const));
  const posteriorCurves = [0, 1, 2].map((k) => GRID.map((x) => [x, softmax(ACTIVATIONS.map((a) => a(x)))[k]!] as const));

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={DOMAIN} yDomain={[-8, 8]} label="Three linear activations aₖ(x)">
        <Axes x={{ label: 'x' }} y={{ label: 'aₖ' }} grid zeroLine />
        {activationCurves.map((c, k) => (
          <Curve key={k} points={c} color={tokens.series[k]!} width={2} />
        ))}
      </Plot>
      <Plot height={200} xDomain={DOMAIN} yDomain={[-0.05, 1.05]} label="softmax(a₁, a₂, a₃), the normalised posterior">
        <Axes x={{ label: 'x' }} y={{ label: 'p(Cₖ|x)' }} grid />
        {posteriorCurves.map((c, k) => (
          <Curve key={k} points={c} color={tokens.series[k]!} width={2} />
        ))}
        <Legend
          entries={[0, 1, 2].map((k) => ({ label: `class ${k + 1}`, color: tokens.series[k]!, mark: 'line' as const }))}
          placement="top-right"
        />
      </Plot>
    </div>
  );
}
