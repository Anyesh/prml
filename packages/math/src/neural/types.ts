import type { Mat, Vec } from '../types.js';

/** Hidden units share one activation; PRML 5.1-5.3 allows any differentiable choice. */
export type HiddenActivation = 'tanh' | 'logistic';

/**
 * `'softmax'` acts on the whole output vector at once (PRML 5.25), unlike the other
 * three which act elementwise, so it is kept in the same union rather than split out:
 * every call site already switches on this value to decide how to combine outputs.
 */
export type OutputActivation = 'linear' | 'logistic' | 'softmax';

/**
 * A feed-forward network's topology, PRML 5.1-5.9: `layerSizes[0]` is the input
 * dimension, `layerSizes[layerSizes.length - 1]` the output dimension, and everything
 * between a hidden layer's unit count. Two entries (no hidden layer) is a valid, if
 * degenerate, spec; the exemplar in this chapter has exactly one hidden layer, matching
 * every worked example and figure in PRML chapter 5.
 */
export interface NetworkSpec {
  readonly layerSizes: readonly number[];
  readonly hiddenActivation: HiddenActivation;
  readonly outputActivation: OutputActivation;
}

/**
 * One matrix per layer transition. `weights[l]` has shape
 * `(layerSizes[l + 1], layerSizes[l] + 1)`: column 0 is the bias `w_{j0}` (PRML absorbs
 * this into the sum by prepending `x_0 = 1`, eq 5.8), columns `1..layerSizes[l]` are
 * `w_{ji}`.
 */
export type NetworkWeights = readonly Mat[];

/**
 * Every intermediate quantity from one forward pass, kept because backpropagation
 * (PRML 5.48-5.56) needs both `z` (to multiply into the weight gradient, eq 5.53) and
 * `a` (to evaluate `h'(a_j)` in the recursion, eq 5.56) at every layer, not just the
 * final output.
 */
export interface ForwardTrace {
  /** `activations[0]` is the input `x`; `activations[l]` for `l > 0` is `z` at layer `l`. */
  readonly activations: readonly Vec[];
  /** `preActivations[0]` is unused (empty); `preActivations[l]` is `a` at layer `l`. */
  readonly preActivations: readonly Vec[];
  readonly output: Vec;
}

export const OUTPUT_ERROR_KINDS = ['sumSquared', 'crossEntropyBinary', 'crossEntropySoftmax'] as const;
export type ErrorKind = (typeof OUTPUT_ERROR_KINDS)[number];

export interface Dataset {
  readonly inputs: Mat;
  readonly targets: Mat;
}
