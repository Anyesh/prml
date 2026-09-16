import { linspace, mean } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const D_GRID = linspace(-2, 2, 200);
const Q_VALUES = [0.3, 1, 2, 10];

const T_VALUES = [1, 2, 2, 3, 3, 10];
const Y_GRID = linspace(0, 6, 6001);

function lossAt(y: number, q: number): number {
  return T_VALUES.reduce((s, t) => s + Math.abs(y - t) ** q, 0) / T_VALUES.length;
}

function minimiser(q: number): number {
  let best = Y_GRID[0]!;
  let bestLoss = Infinity;
  for (const y of Y_GRID) {
    const loss = lossAt(y, q);
    if (loss < bestLoss) {
      bestLoss = loss;
      best = y;
    }
  }
  return best;
}

const MEAN_T = mean(T_VALUES);
const MEDIAN_MINIMISER = minimiser(1);
const MEAN_MINIMISER = minimiser(2);

export default function MinkowskiLossShapes() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[-2, 2]} yDomain={[0, 2]} label="Minkowski loss |y - t|^q for several q">
        <Axes x={{ label: 'y - t' }} y={{ label: '|y-t|^q' }} grid />
        {Q_VALUES.map((q, i) => (
          <Curve key={q} points={D_GRID.map((d) => [d, Math.min(Math.abs(d) ** q, 2)] as const)} color={tokens.series[i % tokens.series.length]!} width={2} />
        ))}
        <Legend entries={Q_VALUES.map((q, i) => ({ label: `q = ${q}`, color: tokens.series[i % tokens.series.length]!, mark: 'line' }))} placement="top-right" />
      </Plot>
      <p className="widget-readout">
        {`Targets {${T_VALUES.join(', ')}}: q=2 minimises at the mean, ${MEAN_MINIMISER.toFixed(2)} (computed ${MEAN_T.toFixed(2)}). q=1 minimises at the median, ${MEDIAN_MINIMISER.toFixed(2)}. The single outlier at 10 pulls the mean well above every other point; the median ignores it entirely.`}
      </p>
    </div>
  );
}
