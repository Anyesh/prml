import {
  designMatrix,
  gaussianBasis,
  linspace,
  matvec,
  pcg32,
  regularisedWeights,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260318;
const N_DATASETS = 20;
const N_PER_DATASET = 25;
const NOISE_STD = 0.3;
const CENTRE_COUNT = 24;
const BASIS_SCALE = 0.1;
const HEAVY_LAMBDA = 50;
const LIGHT_LAMBDA = 1e-6;

interface Dataset {
  readonly xs: number[];
  readonly ts: number[];
}

const TEST_GRID = linspace(0, 1, 100);
const TRUE_CURVE = TEST_GRID.map((x) => Math.sin(2 * Math.PI * x));
const PHI = gaussianBasis(linspace(0, 1, CENTRE_COUNT), BASIS_SCALE, { bias: true });
const TEST_DESIGN = designMatrix(TEST_GRID, PHI);

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

function fitsAt(lambda: number): number[][] {
  return DATASETS.map((d) => {
    const design = designMatrix(d.xs, PHI);
    const weights = regularisedWeights(design, d.ts, lambda);
    return matvec(TEST_DESIGN, weights);
  });
}

function panelRange(fits: readonly (readonly number[])[]): readonly [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const curve of fits) {
    for (const y of curve) {
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
  }
  const pad = Math.max((hi - lo) * 0.1, 0.2);
  return [lo - pad, hi + pad];
}

const HEAVY_FITS = fitsAt(HEAVY_LAMBDA);
const LIGHT_FITS = fitsAt(LIGHT_LAMBDA);
const HEAVY_RANGE = panelRange(HEAVY_FITS);
const LIGHT_RANGE = panelRange(LIGHT_FITS);

function CloudPanel({
  fits,
  yDomain,
  label,
}: {
  fits: readonly (readonly number[])[];
  yDomain: readonly [number, number];
  label: string;
}) {
  const tokens = useResolvedTokens();
  return (
    <Plot height={220} xDomain={[0, 1]} yDomain={yDomain} label={label}>
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      {fits.map((curve, i) => (
        <Curve
          key={i}
          points={TEST_GRID.map((x, j) => [x, curve[j]!] as const)}
          color={tokens.color.inkFaint}
          opacity={0.5}
          width={1}
        />
      ))}
      <Curve points={TEST_GRID.map((x, j) => [x, TRUE_CURVE[j]!] as const)} color={tokens.color.danger} dash="dashed" width={2} />
    </Plot>
  );
}

export default function FitCloud() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))', gap: 'var(--prml-space-4)' }}>
      <CloudPanel fits={HEAVY_FITS} yDomain={HEAVY_RANGE} label="Heavily regularised: tight fits that miss the sine" />
      <CloudPanel fits={LIGHT_FITS} yDomain={LIGHT_RANGE} label="Barely regularised: scattered fits, right on average" />
    </div>
  );
}
