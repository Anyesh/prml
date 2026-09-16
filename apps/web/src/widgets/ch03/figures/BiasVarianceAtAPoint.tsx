import { useMemo, useState } from 'react';
import {
  designMatrix,
  gaussianBasis,
  linspace,
  matvec,
  mean,
  pcg32,
  regularisedWeights,
  standardNormal,
  uniformArray,
  variance,
} from '@prml/math';
import { Annotation, Axes, Band, Curve, FunctionCurve, Legend, Plot, ScatterCloud, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../../widgets.css';

const SEED = 20260321;
const JITTER_SEED = 20260322;
const N_DATASETS = 24;
const N_PER_DATASET = 25;
const NOISE_STD = 0.3;
const CENTRE_COUNT = 20;
const BASIS_SCALE = 0.12;
const LAMBDA = Math.exp(-1);
const JITTER_WIDTH = 0.03;
const OFFSET = 0.05;

interface Dataset {
  readonly xs: number[];
  readonly ts: number[];
}

const TEST_GRID = linspace(0, 1, 60);
const TRUE_CURVE = TEST_GRID.map((x) => Math.sin(2 * Math.PI * x));
const PHI = gaussianBasis(linspace(0, 1, CENTRE_COUNT), BASIS_SCALE, { bias: true });
const TEST_DESIGN = designMatrix(TEST_GRID, PHI);
const JITTER = uniformArray(pcg32(JITTER_SEED), N_DATASETS).map((u) => (u - 0.5) * JITTER_WIDTH);

function buildDatasets(): Dataset[] {
  const base = pcg32(SEED);
  const datasets: Dataset[] = [];
  for (let i = 0; i < N_DATASETS; i++) {
    const rng = base.fork(i);
    const xs: number[] = [];
    const ts: number[] = [];
    for (let n = 0; n < N_PER_DATASET; n++) {
      const x = rng.next();
      xs.push(x);
      ts.push(Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));
    }
    datasets.push({ xs, ts });
  }
  return datasets;
}

const DATASETS = buildDatasets();

/** `predictions[l][i]` is dataset `l`'s prediction at `TEST_GRID[i]`, computed once so the slider only indexes into it. */
const PREDICTIONS: number[][] = DATASETS.map((d) => {
  const design = designMatrix(d.xs, PHI);
  const weights = regularisedWeights(design, d.ts, LAMBDA);
  return matvec(TEST_DESIGN, weights);
});

const Y_DOMAIN: readonly [number, number] = (() => {
  let lo = Infinity;
  let hi = -Infinity;
  for (const curve of PREDICTIONS) {
    for (const y of curve) {
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
  }
  const pad = (hi - lo) * 0.1;
  return [lo - pad, hi + pad];
})();

export default function BiasVarianceAtAPoint() {
  const [x, setX] = useState(0.5);
  const tokens = useResolvedTokens();

  const { idx, xAt, atX, truthAtX, meanAtX, stdAtX, biasAtX } = useMemo(() => {
    const i = Math.round(x * (TEST_GRID.length - 1));
    const draws = PREDICTIONS.map((curve) => curve[i]!);
    const m = mean(draws);
    const std = Math.sqrt(variance(draws));
    const truth = TRUE_CURVE[i]!;
    return { idx: i, xAt: TEST_GRID[i]!, atX: draws, truthAtX: truth, meanAtX: m, stdAtX: std, biasAtX: m - truth };
  }, [x]);

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[0, 1]} yDomain={Y_DOMAIN} label="Ensemble of predictions at one test input, with bias and variance marked">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <FunctionCurve f={(v) => Math.sin(2 * Math.PI * v)} domain={[0, 1]} color={tokens.color.danger} dash="dashed" width={1.5} opacity={0.6} />
        <Band
          points={[
            [xAt - OFFSET - 0.01, meanAtX - stdAtX, meanAtX + stdAtX],
            [xAt - OFFSET + 0.01, meanAtX - stdAtX, meanAtX + stdAtX],
          ]}
          color={tokens.series[1]!}
          opacity={0.35}
        />
        <Curve
          points={[
            [xAt + OFFSET, truthAtX],
            [xAt + OFFSET, meanAtX],
          ]}
          color={tokens.series[0]!}
          dash="dashed"
          width={2}
        />
        <ScatterCloud points={atX.map((y, l) => [xAt + JITTER[l]!, y] as const)} color={tokens.color.inkMuted} size={2.5} opacity={0.8} />
        <ScatterField points={[{ x: xAt, y: meanAtX, id: 'mean' }]} color={tokens.series[0]!} shape="square" size={6} label={() => 'ensemble mean'} />
        <ScatterField points={[{ x: xAt, y: truthAtX, id: 'truth' }]} color={tokens.color.danger} shape="cross" size={7} label={() => 'truth h(x)'} />
        <Annotation
          x={xAt + OFFSET}
          y={(truthAtX + meanAtX) / 2}
          text={`bias ${biasAtX >= 0 ? '+' : ''}${biasAtX.toFixed(3)}`}
          color={tokens.series[0]!}
          anchor="start"
          dx={8}
          plate
        />
        <Annotation
          x={xAt - OFFSET}
          y={meanAtX + stdAtX}
          text={`var ${(stdAtX * stdAtX).toFixed(4)}`}
          color={tokens.series[1]!}
          anchor="end"
          dx={-8}
          dy={-6}
          plate
        />
        <Legend
          entries={[
            { label: 'individual predictions', color: tokens.color.inkMuted, mark: 'dot' },
            { label: 'ensemble mean', color: tokens.series[0]!, mark: 'dot' },
            { label: 'truth h(x)', color: tokens.color.danger, mark: 'dot' },
            { label: 'variance band (±1 std)', color: tokens.series[1]!, mark: 'swatch' },
          ]}
          placement="top-right"
        />
      </Plot>
      <Panel columns={1} dense>
        <Slider label="Test input x" value={x} onChange={setX} min={0} max={1} hint="Moves the vertical strip along the domain: bias and variance both change with position." />
        <p className="widget-readout">
          {`At x = ${xAt.toFixed(2)} (grid point ${idx}): truth = ${truthAtX.toFixed(3)}, ensemble mean = ${meanAtX.toFixed(3)}. Bias and variance are labelled beside the distances they measure in the figure above.`}
        </p>
      </Panel>
    </div>
  );
}
