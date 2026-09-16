import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SCENARIOS = [
  { label: 'small fit', parameters: 4, points: 6 },
  { label: 'richer fit', parameters: 10, points: 40 },
  { label: 'large dataset', parameters: 10, points: 5000 },
];

export default function MemoryFootprint() {
  const tokens = useResolvedTokens();
  const maxValue = Math.max(...SCENARIOS.flatMap((s) => [s.parameters, s.points]));

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-0.6, SCENARIOS.length * 2 - 0.4]} yDomain={[0, maxValue * 1.1]} label="Numbers a parametric model keeps versus numbers a memory-based model keeps">
        <Axes
          x={{ label: 'scenario', ticks: SCENARIOS.flatMap((_, i) => [i * 2, i * 2 + 1]) }}
          y={{ label: 'count kept after training', format: (v) => (v >= 1000 ? `${v / 1000}k` : String(Math.round(v))) }}
          grid
        />
        <Bars
          bars={SCENARIOS.flatMap((s, i) => [
            { at: i * 2, value: s.parameters, color: tokens.series[0]! },
            { at: i * 2 + 1, value: s.points, color: tokens.series[1]! },
          ])}
          thickness={0.7}
        />
      </Plot>
      <p className="widget-readout">
        A parametric fit keeps a handful of weights no matter how much data trained it. A
        model that keeps every training point instead grows with the dataset, which is the
        whole trade this section is about.
      </p>
    </div>
  );
}
