import { evalGrid, exactHessian, networkError, type Dataset, type Grid2D, type NetworkSpec, type NetworkWeights } from '@prml/math';

export const TOY_SPEC: NetworkSpec = { layerSizes: [1, 1, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };

const XS = [-1.5, -0.5, 0.5, 1.5];
const TS = [-0.85, -0.35, 0.35, 0.85];

export const TOY_DATASET: Dataset = { inputs: XS.map((x) => [x]), targets: TS.map((t) => [t]) };

export const TOY_RANGE = 5;

export function toyWeightsAt(w1: number, w2: number): NetworkWeights {
  return [
    [[0, w1]],
    [[0, w2]],
  ];
}

export function toyErrorAt(w1: number, w2: number): number {
  return networkError(TOY_SPEC, toyWeightsAt(w1, w2), TOY_DATASET, 'sumSquared');
}

/**
 * Central difference rather than `backpropGradientBatch` here, because this toy surface
 * fixes both biases at zero and only descends `(w1, w2)`; reusing the general gradient
 * would still require picking the two relevant entries back out, and the finite-difference
 * form makes the restriction to two free coordinates explicit at the call site.
 */
export function toyGradientAt(w1: number, w2: number): readonly [number, number] {
  const h = 1e-4;
  const dw1 = (toyErrorAt(w1 + h, w2) - toyErrorAt(w1 - h, w2)) / (2 * h);
  const dw2 = (toyErrorAt(w1, w2 + h) - toyErrorAt(w1, w2 - h)) / (2 * h);
  return [dw1, dw2];
}

export function toyDescend(start: readonly [number, number], learningRate: number, maxSteps: number): (readonly [number, number])[] {
  const path: (readonly [number, number])[] = [start];
  let [w1, w2] = start;
  for (let step = 0; step < maxSteps; step++) {
    const [g1, g2] = toyGradientAt(w1, w2);
    w1 -= learningRate * g1;
    w2 -= learningRate * g2;
    path.push([w1, w2]);
    if (!Number.isFinite(w1) || !Number.isFinite(w2) || Math.abs(w1) > 3 * TOY_RANGE || Math.abs(w2) > 3 * TOY_RANGE) break;
  }
  return path;
}

export function toySurface(resolution: number): Grid2D {
  const grid = Array.from({ length: resolution }, (_, i) => -TOY_RANGE + (2 * TOY_RANGE * i) / (resolution - 1));
  return evalGrid(grid, grid, toyErrorAt);
}

export function toyExactHessianAt(w1: number, w2: number): number[][] {
  return exactHessian(TOY_SPEC, toyWeightsAt(w1, w2), TOY_DATASET, 'sumSquared') as number[][];
}
