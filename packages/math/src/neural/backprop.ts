import type { Vec } from '../types.js';
import { hiddenDerivative } from './activations.js';
import { augment, forwardPass } from './network.js';
import { assertCanonicalPairing } from './errors.js';
import type { Dataset, ErrorKind, NetworkSpec, NetworkWeights } from './types.js';

/**
 * Backpropagated deltas at every layer for one pattern, PRML 5.51-5.56. Only the
 * output delta (5.54, `y - t`) is used directly by `backpropGradientSingle`; the
 * hidden deltas and the trace are returned too because the exact-Hessian R-operator
 * in `hessian.ts` needs to replay this same recursion with an extra perturbation term
 * at every line.
 */
export interface BackpropTrace {
  readonly activations: readonly Vec[];
  readonly preActivations: readonly Vec[];
  /** `deltas[l]` is `δ` at layer `l` (1-indexed as in the trace); `deltas[0]` is unused. */
  readonly deltas: readonly Vec[];
}

/**
 * PRML's canonical-link result (the remark following 5.18): for sum-of-squares with a
 * linear output, cross-entropy with a logistic output, and multiclass cross-entropy
 * with a softmax output, the output delta is `y - t` in every case, so a single
 * formula covers 5.54, and no case needs the output activation's own derivative.
 */
/**
 * PRML 5.56 alone, propagating an arbitrary terminal (output-layer) delta back through
 * the hidden layers. Factored out because `hessian.ts`'s outer-product approximation
 * needs the same recursion for `b_n = ∇a_n`, whose terminal delta is `1` rather than
 * `y - t`; duplicating the loop would leave two copies of eq 5.56 to keep in sync.
 */
export function propagateDeltas(spec: NetworkSpec, weights: NetworkWeights, activations: readonly Vec[], terminalDelta: Vec): Vec[] {
  const numLayers = weights.length;
  const deltas: Vec[] = new Array(numLayers + 1).fill([]);
  deltas[numLayers] = terminalDelta;

  for (let l = numLayers - 1; l >= 1; l--) {
    const z = activations[l]!;
    const nextWeights = weights[l]!;
    const nextDelta = deltas[l + 1]!;
    deltas[l] = z.map((zj, j) => {
      let backSum = 0;
      for (let k = 0; k < nextWeights.length; k++) backSum += nextWeights[k]![j + 1]! * nextDelta[k]!;
      return hiddenDerivative(spec.hiddenActivation, zj) * backSum;
    });
  }

  return deltas;
}

/**
 * PRML's canonical-link result (the remark following 5.18): for sum-of-squares with a
 * linear output, cross-entropy with a logistic output, and multiclass cross-entropy
 * with a softmax output, the output delta is `y - t` in every case, so a single
 * formula covers 5.54, and no case needs the output activation's own derivative.
 */
export function backpropTrace(spec: NetworkSpec, weights: NetworkWeights, input: Vec, target: Vec, kind: ErrorKind): BackpropTrace {
  assertCanonicalPairing(spec, kind);
  const { activations, preActivations } = forwardPass(spec, weights, input);
  const numLayers = weights.length;
  const output = activations[numLayers]!;
  const terminalDelta = output.map((y, k) => y - target[k]!);
  const deltas = propagateDeltas(spec, weights, activations, terminalDelta);
  return { activations, preActivations, deltas };
}

/** PRML 5.53: `∂E_n/∂w_ji = δ_j z_i`, assembled per layer from a completed backprop trace. */
export function gradientFromTrace(weights: NetworkWeights, trace: Pick<BackpropTrace, 'activations' | 'deltas'>): NetworkWeights {
  const numLayers = weights.length;
  const grad: number[][][] = [];
  for (let l = 0; l < numLayers; l++) {
    const zPrev = augment(trace.activations[l]!);
    const delta = trace.deltas[l + 1]!;
    const rows = delta.map((d) => zPrev.map((z) => d * z));
    grad.push(rows);
  }
  return grad;
}

/** One pattern's gradient, PRML 5.53 assembled across every layer via `gradientFromTrace`. */
export function backpropGradientSingle(spec: NetworkSpec, weights: NetworkWeights, input: Vec, target: Vec, kind: ErrorKind): NetworkWeights {
  const trace = backpropTrace(spec, weights, input, target, kind);
  return gradientFromTrace(weights, trace);
}

function addInto(accumulator: number[][][], addend: NetworkWeights): void {
  for (let l = 0; l < accumulator.length; l++) {
    for (let j = 0; j < accumulator[l]!.length; j++) {
      for (let i = 0; i < accumulator[l]![j]!.length; i++) {
        accumulator[l]![j]![i] = accumulator[l]![j]![i]! + addend[l]![j]![i]!;
      }
    }
  }
}

function zerosLike(weights: NetworkWeights): number[][][] {
  return weights.map((layer) => layer.map((row) => row.map(() => 0)));
}

/** PRML 5.57: `∂E/∂w_ji = Σ_n ∂E_n/∂w_ji`, the batch total over the whole dataset. */
export function backpropGradientBatch(spec: NetworkSpec, weights: NetworkWeights, dataset: Dataset, kind: ErrorKind): NetworkWeights {
  const total = zerosLike(weights);
  for (let n = 0; n < dataset.inputs.length; n++) {
    addInto(total, backpropGradientSingle(spec, weights, dataset.inputs[n]!, dataset.targets[n]!, kind));
  }
  return total;
}
