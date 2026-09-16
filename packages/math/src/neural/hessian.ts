import { outer, matZeros, matAdd, matScale, symmetrise } from '../linalg/index.js';
import type { Mat, Vec } from '../types.js';
import { hiddenDerivative, hiddenSecondDerivative } from './activations.js';
import { augment, countWeights, flattenWeights, forwardPass, unflattenWeights } from './network.js';
import { gradientFromTrace, propagateDeltas } from './backprop.js';
import type { Dataset, ErrorKind, HiddenActivation, NetworkSpec, NetworkWeights } from './types.js';

function requireSingleOutput(spec: NetworkSpec): void {
  if (spec.layerSizes[spec.layerSizes.length - 1] !== 1) {
    throw new Error('outer-product and exact Hessians in this module assume a single output unit (PRML 5.82-5.84)');
  }
}

/** `b_n = ∇_w a_n(x_n)`, PRML 5.84's per-pattern vector, via `propagateDeltas` with a terminal delta of 1. */
export function outputWeightGradient(spec: NetworkSpec, weights: NetworkWeights, input: Vec): number[] {
  requireSingleOutput(spec);
  const { activations } = forwardPass(spec, weights, input);
  const deltas = propagateDeltas(spec, weights, activations, [1]);
  const grad = gradientFromTrace(weights, { activations, deltas });
  return flattenWeights(grad);
}

/**
 * PRML 5.84 (sum-of-squares) and 5.85 (cross-entropy): the Levenberg-Marquardt
 * approximation drops the `Σ_n(y_n - t_n)∇∇y_n` term of the exact Hessian (5.83),
 * which vanishes at a perfect fit and is guaranteed positive semi-definite even away
 * from one, unlike the exact Hessian used for `exactHessian` below.
 */
export function outerProductHessian(spec: NetworkSpec, weights: NetworkWeights, dataset: Dataset, kind: ErrorKind): Mat {
  requireSingleOutput(spec);
  const w = countWeights(spec);
  let h: Mat = matZeros(w, w);
  for (let n = 0; n < dataset.inputs.length; n++) {
    const input = dataset.inputs[n]!;
    const b = outputWeightGradient(spec, weights, input);
    const y = forwardPass(spec, weights, input).output[0]!;
    const scale = kind === 'sumSquared' ? 1 : y * (1 - y);
    h = matAdd(h, matScale(outer(b, b), scale));
  }
  return symmetrise(h);
}

function unitDerivative(kind: HiddenActivation | 'linear', z: number): number {
  return kind === 'linear' ? 1 : hiddenDerivative(kind, z);
}

function unitSecondDerivative(kind: HiddenActivation | 'linear', z: number): number {
  return kind === 'linear' ? 0 : hiddenSecondDerivative(kind, z);
}

/**
 * Pearlmutter's R{.} technique (PRML 5.96-5.111): differentiating every line of the
 * forward and backward pass in the direction `v` gives `Hv` exactly, in one extra
 * forward-backward pass, with no finite differences anywhere. Restricted to a
 * non-softmax output (`linear` or `logistic`) because both are elementwise, so
 * `R{z_L} = h'(a_L) ⊙ R{a_L}` holds at the output layer exactly as it does at a hidden
 * one; softmax mixes units and would need the full softmax Jacobian's own
 * directional derivative, which this chapter's networks do not require.
 */
export function hessianVectorProduct(
  spec: NetworkSpec,
  weights: NetworkWeights,
  input: Vec,
  target: Vec,
  kind: ErrorKind,
  v: NetworkWeights,
): NetworkWeights {
  if (spec.outputActivation === 'softmax') {
    throw new Error('exact Hessian via the R-operator needs an elementwise output activation, not softmax');
  }
  const { activations } = forwardPass(spec, weights, input);
  const numLayers = weights.length;

  const rz: Vec[] = [new Array(spec.layerSizes[0]!).fill(0)];
  const ra: Vec[] = [[]];
  for (let l = 0; l < numLayers; l++) {
    const augZ = augment(activations[l]!);
    const augRz = [0, ...rz[l]!];
    const raNext = matvecAdd(v[l]!, augZ, weights[l]!, augRz);
    const isOutput = l === numLayers - 1;
    const kindAt: HiddenActivation | 'linear' = isOutput ? (spec.outputActivation as 'linear' | 'logistic') : spec.hiddenActivation;
    const rzNext = activations[l + 1]!.map((z, j) => unitDerivative(kindAt, z) * raNext[j]!);
    ra.push(raNext);
    rz.push(rzNext);
  }

  const deltas = propagateDeltas(spec, weights, activations, activations[numLayers]!.map((y, k) => y - target[k]!));
  const rDeltas: Vec[] = new Array(numLayers + 1).fill([]);
  rDeltas[numLayers] = rz[numLayers]!;

  for (let l = numLayers - 1; l >= 1; l--) {
    const z = activations[l]!;
    const nextWeights = weights[l]!;
    const nextV = v[l]!;
    const nextDelta = deltas[l + 1]!;
    const nextRDelta = rDeltas[l + 1]!;
    rDeltas[l] = z.map((zj, j) => {
      let backSum = 0;
      let backSumR = 0;
      for (let k = 0; k < nextWeights.length; k++) {
        backSum += nextWeights[k]![j + 1]! * nextDelta[k]!;
        backSumR += nextV[k]![j + 1]! * nextDelta[k]! + nextWeights[k]![j + 1]! * nextRDelta[k]!;
      }
      return unitSecondDerivative(spec.hiddenActivation, zj) * ra[l]![j]! * backSum + unitDerivative(spec.hiddenActivation, zj) * backSumR;
    });
  }

  const hv: number[][][] = [];
  for (let l = 0; l < numLayers; l++) {
    const augZ = augment(activations[l]!);
    const augRz = [0, ...rz[l]!];
    const delta = deltas[l + 1]!;
    const rDelta = rDeltas[l + 1]!;
    const rows = delta.map((d, k) => augZ.map((z, i) => rDelta[k]! * z + d * augRz[i]!));
    hv.push(rows);
  }
  return hv;
}

function matvecAdd(a: Mat, x: Vec, b: Mat, y: Vec): number[] {
  const out = new Array<number>(a.length);
  for (let k = 0; k < a.length; k++) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += a[k]![i]! * x[i]!;
    for (let i = 0; i < y.length; i++) sum += b[k]![i]! * y[i]!;
    out[k] = sum;
  }
  return out;
}

/**
 * Assembles the full exact Hessian one column at a time, `He_i` for each basis
 * vector `e_i`, via `hessianVectorProduct`. Quadratic in the weight count rather than
 * `hessianVectorProduct`'s linear cost, which is exactly the trade the book makes
 * explicit in 5.4.6: use the vector product directly wherever only `Hv` is needed, and
 * pay the full `W` passes only when the whole matrix must be materialised, as the
 * eigen-decomposition figures in this chapter do.
 */
export function exactHessian(spec: NetworkSpec, weights: NetworkWeights, dataset: Dataset, kind: ErrorKind): Mat {
  const w = countWeights(spec);
  const h: number[][] = matZeros(w, w);
  for (let i = 0; i < w; i++) {
    const basis = new Array<number>(w).fill(0);
    basis[i] = 1;
    const v = unflattenWeights(spec, basis);
    let column = new Array<number>(w).fill(0);
    for (let n = 0; n < dataset.inputs.length; n++) {
      const hv = flattenWeights(hessianVectorProduct(spec, weights, dataset.inputs[n]!, dataset.targets[n]!, kind, v));
      column = column.map((c, j) => c + hv[j]!);
    }
    for (let j = 0; j < w; j++) h[j]![i] = column[j]!;
  }
  return symmetrise(h);
}
