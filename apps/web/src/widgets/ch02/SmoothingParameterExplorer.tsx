import { useMemo, useState } from 'react';
import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Select, Slider } from '@prml/ui';
import { DATA, DOMAIN_HI, DOMAIN_LO, truePdf } from './nonparametricData';
import '../widgets.css';

export const title = 'Three smoothers, one dataset';
export const caption = 'Pick a method and drag its one smoothing parameter. All three chase the same bias-variance trade-off from a different knob.';
export const figure = '2.26';

type Method = 'histogram' | 'kernel' | 'knn';

const GRID = linspace(DOMAIN_LO, DOMAIN_HI, 160);

function histogramEstimate(delta: number): (readonly [number, number])[] {
  const binCount = Math.max(1, Math.round((DOMAIN_HI - DOMAIN_LO) / delta));
  const counts = new Array(binCount).fill(0);
  for (const x of DATA) {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((x - DOMAIN_LO) / delta)));
    counts[idx]++;
  }
  const points: (readonly [number, number])[] = [];
  for (let i = 0; i < binCount; i++) {
    const left = DOMAIN_LO + i * delta;
    const density = counts[i] / (DATA.length * delta);
    points.push([left, density], [left + delta, density]);
  }
  return points;
}

function kernelEstimate(h: number): (readonly [number, number])[] {
  return GRID.map((x) => {
    let sum = 0;
    for (const xn of DATA) sum += normalPdf(x, { mu: xn, sigma2: h * h });
    return [x, sum / DATA.length] as const;
  });
}

function knnEstimate(k: number): (readonly [number, number])[] {
  return GRID.map((x) => {
    const distances = DATA.map((xn) => Math.abs(x - xn)).sort((a, b) => a - b);
    const radius = distances[k - 1]!;
    const volume = 2 * Math.max(radius, 1e-6);
    return [x, k / (DATA.length * volume)] as const;
  });
}

const METHOD_LABEL: Record<Method, string> = {
  histogram: 'Bin width Δ',
  kernel: 'Bandwidth h',
  knn: 'K',
};

export default function SmoothingParameterExplorer() {
  const [method, setMethod] = useState<Method>('kernel');
  const [delta, setDelta] = useState(0.4);
  const [h, setH] = useState(0.3);
  const [k, setK] = useState(5);
  const tokens = useResolvedTokens();

  const estimate = useMemo(() => {
    if (method === 'histogram') return histogramEstimate(delta);
    if (method === 'kernel') return kernelEstimate(h);
    return knnEstimate(k);
  }, [method, delta, h, k]);

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={[DOMAIN_LO, DOMAIN_HI]} yDomain={[0, 0.6]} label="True density, data, and the chosen smoother's estimate">
        <Axes x={{ label: 'x' }} y={{ label: 'density' }} grid />
        <Curve points={GRID.map((x) => [x, truePdf(x)] as const)} color={tokens.color.inkFaint} dash="dashed" width={1.5} />
        <Curve points={estimate} color={tokens.color.accent} width={2} />
        <ScatterField points={DATA.map((x) => ({ x, y: 0.02, size: 2, color: tokens.color.ink }))} />
        <Legend
          entries={[
            { label: 'true density', color: tokens.color.inkFaint, mark: 'dashed-line' },
            { label: 'estimate', color: tokens.color.accent, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>

      <Panel columns={2} dense>
        <Select
          label="Method"
          value={method}
          onChange={(v) => setMethod(v as Method)}
          options={[
            { value: 'histogram', label: 'Histogram' },
            { value: 'kernel', label: 'Gaussian kernel' },
            { value: 'knn', label: 'K-nearest-neighbour' },
          ]}
        />
        {method === 'histogram' ? (
          <Slider label={METHOD_LABEL.histogram} value={delta} onChange={setDelta} min={0.05} max={1.5} />
        ) : method === 'kernel' ? (
          <Slider label={METHOD_LABEL.kernel} value={h} onChange={setH} min={0.03} max={1.2} />
        ) : (
          <Slider label={METHOD_LABEL.knn} value={k} onChange={setK} min={1} max={30} step={1} />
        )}
        <p className="widget-readout">
          {'Same 50 points every time. Small parameter: noisy and spiky. Large: smooth but blind to the second mode.'}
        </p>
      </Panel>
    </div>
  );
}
