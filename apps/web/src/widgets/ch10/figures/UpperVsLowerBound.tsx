import { linspace, logisticLocalBound, sigmoid } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DOMAIN: readonly [number, number] = [-6, 6];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 120);
const LAMBDA = 0.45;

/** PRML 10.135-10.137: the exponential upper bound, exact where sigma'(x) = lambda. */
function exponentialUpperBound(x: number, lambda: number): number {
  const g = -lambda * Math.log(lambda) - (1 - lambda) * Math.log(1 - lambda);
  return Math.exp(lambda * x - g);
}

export default function UpperVsLowerBound() {
  const tokens = useResolvedTokens();
  const sigmoidPoints = GRID.map((x) => [x, sigmoid(x)] as const);
  const upperPoints = GRID.map((x) => [x, Math.min(exponentialUpperBound(x, LAMBDA), 1.4)] as const);
  const lowerPoints = GRID.map((x) => [x, logisticLocalBound(x, 2.5)] as const);

  return (
    <Plot height={220} xDomain={DOMAIN} yDomain={[0, 1.4]} label="Exponential upper bound versus the Gaussian lower bound on the same sigmoid">
      <Axes x={{ label: 'x' }} y={{ label: 'sigma(x)' }} grid />
      <Curve points={sigmoidPoints} color={tokens.color.ink} width={2} />
      <Curve points={upperPoints} color={tokens.color.danger} width={1.5} dash="dashed" />
      <Curve points={lowerPoints} color={tokens.series[0]!} width={1.5} dash="dashed" />
      <Legend
        entries={[
          { label: 'sigma(x)', color: tokens.color.ink, mark: 'line' },
          { label: 'exponential upper bound', color: tokens.color.danger, mark: 'dashed-line' },
          { label: 'Gaussian lower bound', color: tokens.series[0]!, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
