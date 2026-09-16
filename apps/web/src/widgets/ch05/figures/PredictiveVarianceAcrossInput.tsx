import { linspace, mdnPredictiveVariance } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { mdnParamsAt } from '../mdnToyProblem';
import '../../widgets.css';

const GRID = linspace(0.05, 0.95, 60);

export default function PredictiveVarianceAcrossInput() {
  const tokens = useResolvedTokens();
  const points = GRID.map((t) => [t, mdnPredictiveVariance(mdnParamsAt(t))] as const);

  return (
    <Plot height={200} xDomain={[0, 1]} yDomain={[0, Math.max(...points.map(([, v]) => v)) * 1.1]} label="Total predictive variance of the mixture across the input range">
      <Axes x={{ label: 't' }} y={{ label: 'Var[x | t]' }} grid />
      <Curve points={points} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
