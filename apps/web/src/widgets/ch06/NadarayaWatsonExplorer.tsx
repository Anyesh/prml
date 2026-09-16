import { useMemo, useState } from 'react';
import {
  linspace,
  nadarayaWatsonPredict,
  nadarayaWatsonWeights,
  pcg32,
  rbfKernel,
  standardNormal,
} from '@prml/math';
import { Axes, Bars, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const N = 14;
const GRID = linspace(0, 1, 140);

function buildDataset() {
  const rng = pcg32(20260615);
  const xs = Array.from({ length: N }, () => rng.next()).sort((a, b) => a - b);
  const ts = xs.map((x) => Math.sin(2 * Math.PI * x) + 0.2 * standardNormal(rng));
  return { xs, ts };
}

const DATASET = buildDataset();
const TRAIN_VECS = DATASET.xs.map((x) => [x]);

export default function NadarayaWatsonExplorer() {
  const [bandwidth, setBandwidth] = useState(0.08);
  const [queryX, setQueryX] = useState(0.5);
  const tokens = useResolvedTokens();

  const kernel = useMemo(() => rbfKernel(bandwidth), [bandwidth]);

  const curve = useMemo(
    () => GRID.map((x) => [x, nadarayaWatsonPredict(kernel, TRAIN_VECS, DATASET.ts, [x])] as const),
    [kernel],
  );

  const weights = useMemo(() => nadarayaWatsonWeights(kernel, TRAIN_VECS, [queryX]), [kernel, queryX]);
  const weightSum = weights.reduce((s, w) => s + w, 0);

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[0, 1]} yDomain={[-1.6, 1.6]} label="Nadaraya-Watson regression curve over the sinusoidal dataset">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={curve} color={tokens.color.accent} width={2.5} />
        <ScatterField
          points={DATASET.xs.map((x, i) => ({ x, y: DATASET.ts[i]!, id: i }))}
          color={tokens.color.ink}
          size={4}
        />
        <ScatterField
          points={[{ x: queryX, y: nadarayaWatsonPredict(kernel, TRAIN_VECS, DATASET.ts, [queryX]), id: 'query' }]}
          color={tokens.color.danger}
          size={6}
          onMove={(_, x) => setQueryX(Math.min(1, Math.max(0, x)))}
        />
      </Plot>

      <Plot height={280} xDomain={[-0.5, N - 0.5]} yDomain={[0, Math.max(...weights) * 1.15]} label="Weight each training point receives at the query point">
        <Axes x={{ label: 'training point index' }} y={{ label: 'k(x, xn)' }} grid />
        <Bars bars={weights.map((w, i) => ({ at: i, value: w, color: tokens.series[0]! }))} thickness={0.7} />
      </Plot>

      <Panel columns={2} dense>
        <Slider label="Bandwidth" value={bandwidth} onChange={setBandwidth} min={0.01} max={0.4} step={0.005} scale="log" />
        <Slider label="Query point x" value={queryX} onChange={setQueryX} min={0} max={1} step={0.01} />
        <p className="widget-readout">
          {`The ${N} weights at x = ${queryX.toFixed(2)} sum to ${weightSum.toFixed(4)}, as (6.46) guarantees for any bandwidth. Shrink it and the weight concentrates on whichever point sits closest; grow it and every point contributes almost equally.`}
        </p>
      </Panel>
    </div>
  );
}
