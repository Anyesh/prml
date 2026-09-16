import { useMemo, useState } from 'react';
import {
  designMatrix,
  dot,
  gaussianBasis,
  linspace,
  maximumLikelihoodWeights,
  meanSquaredError,
  pcg32,
  polynomialBasis,
  sigmoidalBasis,
  standardNormal,
  type BasisFunction,
} from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Select, Slider } from '@prml/ui';
import '../widgets.css';

type BasisFamily = 'polynomial' | 'gaussian' | 'sigmoidal';

const FAMILY_OPTIONS = [
  { value: 'polynomial', label: 'Polynomial' },
  { value: 'gaussian', label: 'Gaussian' },
  { value: 'sigmoidal', label: 'Sigmoidal' },
] as const;

const DATASET_SEED = 20260301;
const N_POINTS = 25;
const NOISE_STD = 0.25;
const BASIS_DOMAIN: readonly [number, number] = [-1, 1];
const BASIS_GRID = linspace(-1, 1, 161);
const FIT_GRID = linspace(0, 1, 161);

interface Dataset {
  readonly xs: number[];
  readonly ts: number[];
}

/**
 * xs live on [0, 1], the classic PRML sinusoidal curve-fitting domain, so the right-panel
 * fit is meaningful; the left panel then samples the same phi across the wider [-1, 1]
 * window PRML figure 3.1 uses, purely to show each function's shape.
 */
function buildDataset(): Dataset {
  const rng = pcg32(DATASET_SEED);
  const xs: number[] = [];
  const ts: number[] = [];
  for (let i = 0; i < N_POINTS; i++) {
    const x = rng.next();
    xs.push(x);
    ts.push(Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));
  }
  return { xs, ts };
}

const DATASET = buildDataset();

/** `count` includes phi_0 = 1, so a family-specific bump count of `count - 1` matches across families. */
function buildBasis(family: BasisFamily, count: number, scale: number): BasisFunction {
  if (family === 'polynomial') return polynomialBasis(count - 1, { bias: true });
  const centres = linspace(0, 1, count - 1);
  return family === 'gaussian'
    ? gaussianBasis(centres, scale, { bias: true })
    : sigmoidalBasis(centres, scale, { bias: true });
}

export default function BasisFunctionExplorer() {
  const [family, setFamily] = useState<BasisFamily>('polynomial');
  const [count, setCount] = useState(5);
  const [scale, setScale] = useState(0.1);
  const tokens = useResolvedTokens();

  const phi = useMemo(() => buildBasis(family, count, scale), [family, count, scale]);

  const basisCurves = useMemo(() => {
    const columns: number[][] = Array.from({ length: count }, () => []);
    for (const x of BASIS_GRID) {
      const row = phi(x);
      for (let j = 0; j < count; j++) columns[j]!.push(row[j] ?? 0);
    }
    return columns.map((ys) => BASIS_GRID.map((x, i) => [x, ys[i]!] as const));
  }, [phi, count]);

  const fit = useMemo(() => {
    const design = designMatrix(DATASET.xs, phi);
    const weights = maximumLikelihoodWeights(design, DATASET.ts);
    const mse = meanSquaredError(design, DATASET.ts, weights);
    const curve = FIT_GRID.map((x) => [x, dot(phi(x), weights)] as const);
    return { mse, curve };
  }, [phi]);

  const dataPoints = DATASET.xs.map((x, i) => ({ x, y: DATASET.ts[i]!, id: i }));

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={BASIS_DOMAIN} yDomain={[-1.1, 1.1]} label="Individual basis functions">
        <Axes x={{ label: 'x' }} y={{ label: 'φⱼ(x)' }} grid zeroLine />
        {basisCurves.map((points, j) => (
          <Curve key={j} points={points} color={tokens.series[j % tokens.series.length]!} width={1.5} />
        ))}
      </Plot>

      <Plot height={280} xDomain={[0, 1]} yDomain={[-1.6, 1.6]} label="Least-squares fit of a noisy sinusoid">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={fit.curve} color={tokens.color.accent} width={2} />
        <ScatterField points={dataPoints} color={tokens.color.ink} size={3.5} />
      </Plot>

      <Panel columns={2} dense>
        <Select label="Basis family" value={family} onChange={(v) => setFamily(v as BasisFamily)} options={FAMILY_OPTIONS} />
        <Slider
          label="Number of basis functions M"
          value={count}
          onChange={setCount}
          min={2}
          max={12}
          step={1}
          hint="Includes the constant φ₀ = 1, so M − 1 functions vary with the family below."
        />
        {family !== 'polynomial' && (
          <Slider
            label="Scale s"
            value={scale}
            onChange={setScale}
            min={0.02}
            max={0.5}
            scale="log"
            hint="How far each bump reaches before it fades to zero."
          />
        )}
        <p className="widget-readout">
          {`Mean squared error: ${fit.mse.toFixed(4)}. ${
            family === 'polynomial'
              ? 'Polynomials are global: every coefficient shapes the curve everywhere, so nudging one data point can bend the fit far from where it sits.'
              : `${family === 'gaussian' ? 'Gaussian' : 'Sigmoidal'} basis functions are local: each one only responds near its own centre, so nudging one data point mostly bends the fit nearby.`
          }`}
        </p>
      </Panel>
    </div>
  );
}
