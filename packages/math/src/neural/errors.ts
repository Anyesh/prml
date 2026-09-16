import type { Vec } from '../types.js';
import { forwardPass } from './network.js';
import type { Dataset, ErrorKind, NetworkSpec, NetworkWeights } from './types.js';

/** PRML 5.11, one pattern's contribution (no leading sum over `n`). */
export function sumSquaredErrorSingle(predicted: Vec, target: Vec): number {
  let sum = 0;
  for (let k = 0; k < predicted.length; k++) {
    const d = predicted[k]! - target[k]!;
    sum += d * d;
  }
  return 0.5 * sum;
}

function softplus(a: number): number {
  return Math.max(a, 0) + Math.log1p(Math.exp(-Math.abs(a)));
}

/**
 * PRML 5.21/5.23, evaluated from the pre-sigmoid activations via the softplus identity
 * `E = softplus(a) - t*a`, exactly as `classification/irls.ts` does for the same
 * reason: it stays finite where `-ln(sigmoid(a))` would evaluate `ln(0)` once training
 * pushes an activation to saturation.
 */
export function crossEntropyBinarySingleFromActivations(preActivations: Vec, target: Vec): number {
  let sum = 0;
  for (let k = 0; k < preActivations.length; k++) {
    sum += softplus(preActivations[k]!) - target[k]! * preActivations[k]!;
  }
  return sum;
}

/** PRML 5.24, from the softmax probabilities directly; `target` is 1-of-K. */
export function crossEntropySoftmaxSingle(predicted: Vec, target: Vec): number {
  let sum = 0;
  for (let k = 0; k < predicted.length; k++) {
    if (target[k]! !== 0) sum -= target[k]! * Math.log(Math.max(predicted[k]!, Number.EPSILON));
  }
  return sum;
}

/**
 * Dispatches on `kind`, reading `preActivations` for the cross-entropy-binary case
 * (see `crossEntropyBinarySingleFromActivations`) and `predicted` for the other two,
 * so the softplus stabilisation is only paid for the case that needs it.
 */
export function errorSingle(kind: ErrorKind, predicted: Vec, preActivations: Vec, target: Vec): number {
  if (kind === 'sumSquared') return sumSquaredErrorSingle(predicted, target);
  if (kind === 'crossEntropyBinary') return crossEntropyBinarySingleFromActivations(preActivations, target);
  return crossEntropySoftmaxSingle(predicted, target);
}

/** PRML 5.44/5.57: total error is a sum over patterns of the per-pattern error above. */
export function networkError(spec: NetworkSpec, weights: NetworkWeights, dataset: Dataset, kind: ErrorKind): number {
  let total = 0;
  for (let n = 0; n < dataset.inputs.length; n++) {
    const trace = forwardPass(spec, weights, dataset.inputs[n]!);
    total += errorSingle(kind, trace.output, trace.preActivations[trace.preActivations.length - 1]!, dataset.targets[n]!);
  }
  return total;
}

export function networkErrorSingle(spec: NetworkSpec, weights: NetworkWeights, input: Vec, target: Vec, kind: ErrorKind): number {
  const trace = forwardPass(spec, weights, input);
  return errorSingle(kind, trace.output, trace.preActivations[trace.preActivations.length - 1]!, target);
}

/** Canonical activation each error kind must be paired with, PRML's remark after 5.18. */
export const CANONICAL_OUTPUT: Record<ErrorKind, 'linear' | 'logistic' | 'softmax'> = {
  sumSquared: 'linear',
  crossEntropyBinary: 'logistic',
  crossEntropySoftmax: 'softmax',
};

export function assertCanonicalPairing(spec: NetworkSpec, kind: ErrorKind): void {
  const expected = CANONICAL_OUTPUT[kind];
  if (spec.outputActivation !== expected) {
    throw new Error(
      `${kind} is canonically paired with a "${expected}" output activation (PRML 5.18), got "${spec.outputActivation}"`,
    );
  }
}
