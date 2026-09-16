import { linspace } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const W_GRID = linspace(1, 200, 100);
const BACKPROP_CONST = 4;
const FINITE_DIFF_CONST = 4;

export default function EfficiencyComparison() {
  const tokens = useResolvedTokens();
  const backpropCurve = W_GRID.map((w) => [w, BACKPROP_CONST * w] as const);
  const finiteDiffCurve = W_GRID.map((w) => [w, FINITE_DIFF_CONST * w * w] as const);

  return (
    <Plot height={240} xDomain={[0, 200]} yDomain={[0, 4000]} label="Cost of one gradient evaluation against the number of weights">
      <Axes x={{ label: 'W (number of weights)' }} y={{ label: 'multiply-adds (arbitrary units)' }} grid />
      <Curve points={backpropCurve} color={tokens.color.accent} width={2.5} />
      <Curve points={finiteDiffCurve} color={tokens.color.danger} width={2.5} />
    </Plot>
  );
}
