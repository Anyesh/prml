import { backpropGradientBatch, flattenWeights, initializeWeights, pcg32, unflattenWeights, type Dataset, type NetworkSpec } from '@prml/math';

export const HESSIAN_SPEC: NetworkSpec = { layerSizes: [2, 3, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };

const XS = [
  [0.6, -0.4],
  [-0.5, 0.7],
  [0.2, 0.9],
  [-0.8, -0.3],
];
const TS = [[0.5], [-0.3], [0.6], [-0.7]];

export const HESSIAN_DATASET: Dataset = { inputs: XS, targets: TS };

const LEARNING_RATE = 0.3;

export function trainHessianToy(steps: number): number[] {
  let flat = flattenWeights(initializeWeights(HESSIAN_SPEC, pcg32(20260550), 1.0));
  for (let step = 0; step < steps; step++) {
    const weights = unflattenWeights(HESSIAN_SPEC, flat);
    const grad = flattenWeights(backpropGradientBatch(HESSIAN_SPEC, weights, HESSIAN_DATASET, 'sumSquared'));
    flat = flat.map((w, i) => w - (LEARNING_RATE / XS.length) * grad[i]!);
  }
  return flat;
}
