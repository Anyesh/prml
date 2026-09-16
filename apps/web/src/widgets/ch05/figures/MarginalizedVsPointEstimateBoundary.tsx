import {
  backpropGradientBatch,
  evalGrid,
  exactHessian,
  eye,
  flattenWeights,
  forwardPass,
  initializeWeights,
  inverse,
  kappa,
  matAdd,
  outputWeightGradient,
  pcg32,
  sigmoid,
  standardNormal,
  unflattenWeights,
  type Dataset,
  type NetworkSpec,
} from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [2, 4, 1], hiddenActivation: 'tanh', outputActivation: 'logistic' };
const N_PER_CLASS = 15;

function buildDataset(): Dataset {
  const rng = pcg32(20260640);
  const class0 = Array.from({ length: N_PER_CLASS }, () => [-1 + standardNormal(rng) * 0.6, -1 + standardNormal(rng) * 0.6]);
  const class1 = Array.from({ length: N_PER_CLASS }, () => [1 + standardNormal(rng) * 0.6, 1 + standardNormal(rng) * 0.6]);
  return { inputs: [...class0, ...class1], targets: [...class0.map(() => [0]), ...class1.map(() => [1])] };
}

const DATASET = buildDataset();
const ALPHA = 0.2;

function train(): number[] {
  let flat = flattenWeights(initializeWeights(SPEC, pcg32(20260641), 1.0));
  for (let step = 0; step < 800; step++) {
    const weights = unflattenWeights(SPEC, flat);
    const grad = flattenWeights(backpropGradientBatch(SPEC, weights, DATASET, 'crossEntropyBinary'));
    flat = flat.map((w, i) => w - (0.5 / DATASET.inputs.length) * grad[i]! - ALPHA * w * (0.5 / DATASET.inputs.length));
  }
  return flat;
}

const W_MAP = unflattenWeights(SPEC, train());
const H = exactHessian(SPEC, W_MAP, DATASET, 'crossEntropyBinary');
const A_INVERSE = inverse(matAdd(H, eye(H.length, ALPHA)));

function activationAt(x: number, y: number): number {
  const augmented = [1, ...forwardPass(SPEC, W_MAP, [x, y]).activations[1]!];
  const outRow = W_MAP[W_MAP.length - 1]![0]!;
  let a = 0;
  for (let i = 0; i < augmented.length; i++) a += outRow[i]! * augmented[i]!;
  return a;
}

function marginalizedProbAt(x: number, y: number): number {
  const a = activationAt(x, y);
  const g = outputWeightGradient(SPEC, W_MAP, [x, y]);
  let variance = 0;
  for (let i = 0; i < g.length; i++) for (let j = 0; j < g.length; j++) variance += g[i]! * A_INVERSE[i]![j]! * g[j]!;
  return sigmoid(kappa(variance) * a);
}

const GRID = Array.from({ length: 41 }, (_, i) => -3 + (6 * i) / 40);
const POINT_ESTIMATE = evalGrid(GRID, GRID, (x, y) => forwardPass(SPEC, W_MAP, [x, y]).output[0]!);
const MARGINALIZED = evalGrid(GRID, GRID, marginalizedProbAt);

export default function MarginalizedVsPointEstimateBoundary() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={280} xDomain={[-3, 3]} yDomain={[-3, 3]} equalAspect label="Point-estimate confidence contours against the marginalized ones, both sharing the same p = 0.5 boundary">
      <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid zeroLine />
      <ContourField data={POINT_ESTIMATE} levels={[0.1, 0.5, 0.9]} color={tokens.color.danger} lineWidth={1.5} />
      <ContourField data={MARGINALIZED} levels={[0.1, 0.5, 0.9]} color={tokens.color.accent} lineWidth={1.5} />
      <ScatterField points={DATASET.inputs.map((p, i) => ({ x: p[0]!, y: p[1]!, id: i, color: DATASET.targets[i]![0] === 1 ? tokens.series[1] : tokens.series[0] }))} size={3.5} />
    </Plot>
  );
}
