import { evalGrid, linspace, normalSample, pcg32 } from '@prml/math';
import { Axes, Heatmap, Plot, ScatterField, interpolateStops, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

interface Labelled {
  readonly x: number;
  readonly y: number;
  readonly label: 0 | 1;
}

const CENTRES: readonly [number, number][] = [
  [-1, -1],
  [1, 1],
];

const POINTS: readonly Labelled[] = (() => {
  const rng = pcg32(20260916, 9);
  const out: Labelled[] = [];
  for (const label of [0, 1] as const) {
    const [cx, cy] = CENTRES[label]!;
    for (let i = 0; i < 25; i++) {
      out.push({ x: normalSample(rng, { mu: cx, sigma2: 0.4 }), y: normalSample(rng, { mu: cy, sigma2: 0.4 }), label });
    }
  }
  return out;
})();

const GRID = linspace(-3, 3, 60);

function majorityClass(x: number, y: number, k: number): number {
  const distances = POINTS.map((p) => ({ d: (p.x - x) ** 2 + (p.y - y) ** 2, label: p.label }));
  distances.sort((a, b) => a.d - b.d);
  let ones = 0;
  for (let i = 0; i < k; i++) ones += distances[i]!.label;
  return ones / k > 0.5 ? 1 : 0;
}

function Panel({ k, tokens }: { k: number; tokens: ReturnType<typeof useResolvedTokens> }) {
  const field = evalGrid(GRID, GRID, (x, y) => majorityClass(x, y, k));
  const fill = interpolateStops([tokens.series[0]!, tokens.series[1]!]);
  return (
    <div>
      <Plot height={200} xDomain={[-3, 3]} yDomain={[-3, 3]} equalAspect label={`K-nearest-neighbour decision regions, K=${k}`}>
        <Heatmap data={field} interpolator={fill} opacity={0.35} />
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} />
        <ScatterField
          points={POINTS.map((p) => ({ x: p.x, y: p.y, color: tokens.series[p.label]!, size: 3 }))}
        />
      </Plot>
      <p className="widget-readout">{`K = ${k}`}</p>
    </div>
  );
}

export default function KnnClassification() {
  const tokens = useResolvedTokens();
  return (
    <div className="widget-grid">
      <Panel k={1} tokens={tokens} />
      <Panel k={15} tokens={tokens} />
    </div>
  );
}
