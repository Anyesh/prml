import { linspace } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { predictiveMeanAndVariance, REG_DATASET } from './bayesianToyNetwork';
import '../widgets.css';

const GRID = linspace(-3, 3, 121);

export default function LaplacePredictiveErrorBars() {
  const tokens = useResolvedTokens();
  const stats = GRID.map((x) => ({ x, ...predictiveMeanAndVariance(x) }));
  const meanCurve = stats.map((s) => [s.x, s.mean] as const);
  const upper = stats.map((s) => [s.x, s.mean + 2 * Math.sqrt(s.variance)] as const);
  const lower = stats.map((s) => [s.x, s.mean - 2 * Math.sqrt(s.variance)] as const);

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={[-3, 3]} yDomain={[-3, 3]} label="Predictive mean with a two-standard-deviation band, from the Laplace-approximated weight posterior">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid zeroLine />
        <Curve points={upper} color={tokens.color.inkFaint} width={1} />
        <Curve points={lower} color={tokens.color.inkFaint} width={1} />
        <Curve points={meanCurve} color={tokens.color.accent} width={2.5} />
        <ScatterField points={REG_DATASET.inputs.map((x, i) => ({ x: x[0]!, y: REG_DATASET.targets[i]![0]!, id: i, color: tokens.color.ink, size: 4 }))} />
      </Plot>
      <p className="widget-readout">
        Training data sits inside [-0.8, 0.7]. The band stays narrow there and widens fast
        past either edge, where no data constrains which of many weight settings the network
        should trust.
      </p>
    </div>
  );
}
