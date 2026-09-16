import {
  designMatrix,
  linspace,
  maximumLikelihoodWeights,
  mean,
  pcg32,
  polynomialBasis,
  standardNormal,
} from '@prml/math';
import { Axes, FunctionCurve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260319;
const N_DATASETS = 3;
const N_PER_DATASET = 6;
const NOISE_STD = 0.25;
const GRID: readonly [number, number] = [0, 1];

interface Dataset {
  readonly xs: number[];
  readonly ts: number[];
}

function buildDatasets(): Dataset[] {
  const base = pcg32(SEED);
  const datasets: Dataset[] = [];
  for (let i = 0; i < N_DATASETS; i++) {
    const rng = base.fork(i);
    const xs = linspace(0, 1, N_PER_DATASET).map((x, j) => (j === 0 || j === N_PER_DATASET - 1 ? x : x + 0.06 * (rng.next() - 0.5)));
    const ts = xs.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));
    datasets.push({ xs, ts });
  }
  return datasets;
}

const DATASETS = buildDatasets();
const INTERPOLANT_BASIS = polynomialBasis(N_PER_DATASET - 1);

function constantFit(dataset: Dataset): (x: number) => number {
  const c = mean(dataset.ts);
  return () => c;
}

function interpolantFit(dataset: Dataset): (x: number) => number {
  const design = designMatrix(dataset.xs, INTERPOLANT_BASIS);
  const weights = maximumLikelihoodWeights(design, dataset.ts);
  const phi = INTERPOLANT_BASIS;
  return (x: number) => phi(x).reduce((sum, v, i) => sum + v * weights[i]!, 0);
}

export default function ConstantVersusInterpolant() {
  const tokens = useResolvedTokens();
  const constantColor = tokens.series[0]!;
  const interpolantColor = tokens.series[1]!;
  const truthColor = tokens.color.danger;

  return (
    <Plot
      height={240}
      xDomain={GRID}
      yDomain={[-2.2, 2.2]}
      label="Three datasets each fit with a constant and with a degree-5 interpolant"
    >
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <FunctionCurve f={(x) => Math.sin(2 * Math.PI * x)} domain={GRID} color={truthColor} dash="dashed" width={1.75} />
      {DATASETS.map((d, i) => (
        <FunctionCurve key={`const-${i}`} f={constantFit(d)} domain={GRID} color={constantColor} width={1.5} opacity={0.85} />
      ))}
      {DATASETS.map((d, i) => (
        <FunctionCurve key={`interp-${i}`} f={interpolantFit(d)} domain={GRID} color={interpolantColor} width={1.5} opacity={0.85} />
      ))}
      {DATASETS.map((d, i) => (
        <ScatterField
          key={`points-${i}`}
          points={d.xs.map((x, j) => ({ x, y: d.ts[j]!, id: `${i}-${j}` }))}
          color={tokens.color.inkMuted}
          size={3}
          label={() => 'sample point'}
        />
      ))}
      <Legend
        entries={[
          { label: 'constant fit (3 datasets)', color: constantColor, mark: 'line' },
          { label: 'degree-5 interpolant (3 datasets)', color: interpolantColor, mark: 'line' },
          { label: 'sin(2πx)', color: truthColor, mark: 'dashed-line' },
        ]}
        placement="top-right"
      />
    </Plot>
  );
}
