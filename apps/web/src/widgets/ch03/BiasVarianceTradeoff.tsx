import { useMemo, useState } from 'react';
import {
  biasVarianceDecomposition,
  designMatrix,
  ensembleMean,
  gaussianBasis,
  linspace,
  matvec,
  pcg32,
  regularisedWeights,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

interface Dataset {
  readonly xs: number[];
  readonly ts: number[];
}

interface LambdaPoint {
  readonly lnLambda: number;
  readonly bias2: number;
  readonly variance: number;
  readonly total: number;
}

const ENSEMBLE_SEED = 20260302;
const N_DATASETS = 20;
const N_PER_DATASET = 25;
const NOISE_STD = 0.3;
const CENTRE_COUNT = 24;
const BASIS_SCALE = 0.1;
const LN_LAMBDA_MIN = -6;
const LN_LAMBDA_MAX = 6;
const LAMBDA_CURVE_POINTS = 41;

const TEST_GRID = linspace(0, 1, 100);
const TRUE_CURVE = TEST_GRID.map((x) => Math.sin(2 * Math.PI * x));
const PHI = gaussianBasis(linspace(0, 1, CENTRE_COUNT), BASIS_SCALE, { bias: true });
const TEST_DESIGN = designMatrix(TEST_GRID, PHI);
const LN_LAMBDA_GRID = linspace(LN_LAMBDA_MIN, LN_LAMBDA_MAX, LAMBDA_CURVE_POINTS);

/** Each dataset draws from its own fork so every panel sees the same 20 resamples regardless of render order. */
function buildDatasets(): Dataset[] {
  const base = pcg32(ENSEMBLE_SEED);
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

function predictAtTest(dataset: Dataset, lambda: number): number[] {
  const design = designMatrix(dataset.xs, PHI);
  const weights = regularisedWeights(design, dataset.ts, lambda);
  return matvec(TEST_DESIGN, weights);
}

function buildLambdaCurve(): LambdaPoint[] {
  return LN_LAMBDA_GRID.map((lnLambda) => {
    const lambda = Math.exp(lnLambda);
    const predictions = DATASETS.map((d) => predictAtTest(d, lambda));
    const { bias2, variance, total } = biasVarianceDecomposition(predictions, TRUE_CURVE);
    return { lnLambda, bias2, variance, total };
  });
}

// Computed once at module load: sweeping 41 lambdas across 20 datasets on every slider tick
// would stall the drag, so only the single currently-selected lambda is refit on change.
const LAMBDA_CURVE = buildLambdaCurve();
const LAMBDA_CURVE_MAX = LAMBDA_CURVE.reduce((m, p) => Math.max(m, p.bias2, p.variance, p.total), 0);

export default function BiasVarianceTradeoff() {
  const [lambda, setLambda] = useState(Math.exp(-1));
  const tokens = useResolvedTokens();

  const lnLambda = Math.log(lambda);

  const ensemble = useMemo(() => {
    const predictions = DATASETS.map((d) => predictAtTest(d, lambda));
    const mean = ensembleMean(predictions);
    const decomposition = biasVarianceDecomposition(predictions, TRUE_CURVE);
    return { predictions, mean, decomposition };
  }, [lambda]);

  const leftYDomain = useMemo<readonly [number, number]>(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const curve of ensemble.predictions) {
      for (const y of curve) {
        if (y < lo) lo = y;
        if (y > hi) hi = y;
      }
    }
    const mid = (lo + hi) / 2;
    const halfRange = Math.max((hi - lo) / 2, 0.75) * 1.15;
    return [mid - halfRange, mid + halfRange];
  }, [ensemble.predictions]);

  const regime =
    lnLambda < -2
      ? "a low-λ regime, where each fit chases its own dataset's noise: the curves scatter widely (high variance) even though they average out close to the sine (low bias)"
      : lnLambda > 2
        ? 'a high-λ regime, where regularisation flattens every fit toward the same shape: the curves barely scatter (low variance) but that shared shape misses the sine (high bias)'
        : 'the regime near the total-error minimum, where bias² and variance are both moderate and their sum sits close to its smallest value';

  return (
    <div className="widget-grid">
      <Plot
        height={300}
        xDomain={[0, 1]}
        yDomain={leftYDomain}
        label="Ensemble of fits at the current lambda, one per resampled dataset"
      >
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        {ensemble.predictions.map((curve, i) => (
          <Curve
            key={i}
            points={TEST_GRID.map((x, j) => [x, curve[j]!] as const)}
            color={tokens.color.inkFaint}
            opacity={0.5}
            width={1}
          />
        ))}
        <Curve points={TEST_GRID.map((x, j) => [x, ensemble.mean[j]!] as const)} color={tokens.color.accent} width={2.5} />
        <Curve points={TEST_GRID.map((x, j) => [x, TRUE_CURVE[j]!] as const)} color={tokens.color.danger} dash="dashed" width={1.75} />
        <Legend
          entries={[
            { label: 'individual fits', color: tokens.color.inkFaint, mark: 'line' },
            { label: 'average fit', color: tokens.color.accent, mark: 'line' },
            { label: 'sin(2πx)', color: tokens.color.danger, mark: 'dashed-line' },
          ]}
          placement="top-right"
        />
      </Plot>

      <Plot
        height={300}
        xDomain={[LN_LAMBDA_MIN, LN_LAMBDA_MAX]}
        yDomain={[0, LAMBDA_CURVE_MAX * 1.05]}
        label="Bias squared, variance, and their sum against ln lambda"
      >
        <Axes x={{ label: 'ln λ' }} y={{ label: 'error' }} grid />
        <Curve points={LAMBDA_CURVE.map((p) => [p.lnLambda, p.bias2] as const)} color={tokens.series[0]!} width={1.75} />
        <Curve points={LAMBDA_CURVE.map((p) => [p.lnLambda, p.variance] as const)} color={tokens.series[1]!} width={1.75} />
        <Curve points={LAMBDA_CURVE.map((p) => [p.lnLambda, p.total] as const)} color={tokens.series[2]!} width={2} />
        <ScatterField
          points={[
            { x: lnLambda, y: ensemble.decomposition.bias2, color: tokens.series[0]!, id: 'bias2' },
            { x: lnLambda, y: ensemble.decomposition.variance, color: tokens.series[1]!, id: 'variance' },
            { x: lnLambda, y: ensemble.decomposition.total, color: tokens.series[2]!, id: 'total' },
          ]}
          size={4}
        />
        <Legend
          entries={[
            { label: 'bias²', color: tokens.series[0]!, mark: 'line' },
            { label: 'variance', color: tokens.series[1]!, mark: 'line' },
            { label: 'bias² + variance', color: tokens.series[2]!, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>

      <Panel columns={2} dense>
        <Slider
          label="Regularisation λ"
          value={lambda}
          onChange={setLambda}
          min={Math.exp(LN_LAMBDA_MIN)}
          max={Math.exp(LN_LAMBDA_MAX)}
          scale="log"
          format={(v) => `ln λ = ${Math.log(v).toFixed(2)}`}
          hint="Slides on a log scale because λ acts multiplicatively: equal steps here are equal ratios, not equal amounts."
        />
        <p className="widget-readout">
          {`bias² = ${ensemble.decomposition.bias2.toFixed(4)}, variance = ${ensemble.decomposition.variance.toFixed(4)}, total = ${ensemble.decomposition.total.toFixed(4)}. This is ${regime}.`}
        </p>
      </Panel>
    </div>
  );
}
