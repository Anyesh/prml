import { dot } from '../linalg/index.js';
import type { Mat, Vec } from '../types.js';

/** `wᵀφ(x)`, the argument to the perceptron's step activation, PRML 4.52. */
export function perceptronActivation(weights: Vec, features: Vec): number {
  return dot(weights, features);
}

/** The step activation of PRML 4.53, on the `{-1, +1}` coding the perceptron uses. */
export function perceptronPredict(weights: Vec, features: Vec): 1 | -1 {
  return perceptronActivation(weights, features) >= 0 ? 1 : -1;
}

export interface PerceptronStepResult {
  readonly weights: number[];
  readonly updated: boolean;
}

/**
 * PRML 4.54-4.55: a pattern is correctly classified when `wᵀφ(xn) tn > 0`; anything else,
 * including the boundary case of exact zero, counts as misclassified, since the perceptron
 * criterion assigns it nonzero error. The update is skipped entirely when the pattern is
 * already correct, which is what makes the total number of updates finite on separable data.
 */
export function perceptronStep(
  weights: Vec,
  features: Vec,
  target: 1 | -1,
  learningRate = 1,
): PerceptronStepResult {
  const misclassified = perceptronActivation(weights, features) * target <= 0;
  if (!misclassified) return { weights: [...weights], updated: false };
  const next = weights.map((w, i) => w + learningRate * features[i]! * target);
  return { weights: next, updated: true };
}

export interface PerceptronTrainResult {
  readonly weights: number[];
  readonly epochs: number;
  readonly converged: boolean;
  readonly history: readonly { readonly epoch: number; readonly misclassified: number }[];
}

export interface PerceptronTrainOptions {
  readonly initialWeights: Vec;
  readonly maxEpochs?: number;
  readonly learningRate?: number;
}

/**
 * Cycles through the rows of `design` in the given order once per epoch, which is PRML's
 * own description of the algorithm ("we cycle through the training patterns in turn").
 * Order therefore matters and is the caller's to fix; a stochastic implementation that
 * reshuffled between epochs would make the trajectory a different, unreproducible one.
 */
export function perceptronTrain(
  design: Mat,
  targets: readonly (1 | -1)[],
  options: PerceptronTrainOptions,
): PerceptronTrainResult {
  const maxEpochs = options.maxEpochs ?? 1000;
  const learningRate = options.learningRate ?? 1;
  let weights = [...options.initialWeights];
  const history: { epoch: number; misclassified: number }[] = [];

  for (let epoch = 0; epoch < maxEpochs; epoch++) {
    let misclassified = 0;
    for (let n = 0; n < design.length; n++) {
      const step = perceptronStep(weights, design[n]!, targets[n]!, learningRate);
      if (step.updated) misclassified++;
      weights = step.weights;
    }
    history.push({ epoch, misclassified });
    if (misclassified === 0) return { weights, epochs: epoch + 1, converged: true, history };
  }
  return { weights, epochs: maxEpochs, converged: false, history };
}
