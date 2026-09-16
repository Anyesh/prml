import {
  backpropGradientBatch,
  exactHessian,
  eye,
  flattenWeights,
  forwardPass,
  initializeWeights,
  inverse,
  matAdd,
  matScale,
  networkError,
  outputWeightGradient,
  pcg32,
  standardNormal,
  unflattenWeights,
  type Dataset,
  type NetworkSpec,
} from '@prml/math';

export const REG_SPEC: NetworkSpec = { layerSizes: [1, 6, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };

const XS = [-0.8, -0.5, -0.2, 0.1, 0.4, 0.7];

function buildDataset(): Dataset {
  const rng = pcg32(20260630);
  const inputs = XS.map((x) => [x]);
  const targets = XS.map((x) => [Math.sin(2 * x) + 0.05 * standardNormal(rng)]);
  return { inputs, targets };
}

export const REG_DATASET = buildDataset();

const LEARNING_RATE = 0.3;
const STEPS = 1500;
export const ALPHA = 0.05;
export const BETA = 25;

function train(): number[] {
  let flat = flattenWeights(initializeWeights(REG_SPEC, pcg32(20260631), 1.0));
  for (let step = 0; step < STEPS; step++) {
    const weights = unflattenWeights(REG_SPEC, flat);
    const grad = flattenWeights(backpropGradientBatch(REG_SPEC, weights, REG_DATASET, 'sumSquared'));
    flat = flat.map((w, i) => w - (LEARNING_RATE / XS.length) * grad[i]!);
  }
  return flat;
}

export const W_MAP = unflattenWeights(REG_SPEC, train());
const H = exactHessian(REG_SPEC, W_MAP, REG_DATASET, 'sumSquared');
const A = matAdd(matScale(H, BETA), eye(H.length, ALPHA));
export const A_INVERSE = inverse(A);

export function predictiveMeanAndVariance(x: number): { mean: number; variance: number } {
  const mean = forwardPass(REG_SPEC, W_MAP, [x]).output[0]!;
  const g = outputWeightGradient(REG_SPEC, W_MAP, [x]);
  let quad = 0;
  for (let i = 0; i < g.length; i++) {
    for (let j = 0; j < g.length; j++) {
      quad += g[i]! * A_INVERSE[i]![j]! * g[j]!;
    }
  }
  return { mean, variance: 1 / BETA + quad };
}

export function trainedError(): number {
  return networkError(REG_SPEC, W_MAP, REG_DATASET, 'sumSquared');
}
