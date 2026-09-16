import { cholesky, dot, eye, matAdd, solveCholesky } from '../linalg/index.js';
import type { Mat, Vec } from '../types.js';
import { kernelVector } from './combine.js';
import type { KernelFunction } from './functions.js';

/**
 * PRML 6.8: dual ridge-regression coefficients `a = (K + lambda I)^-1 t`. `gram` is the
 * caller's own `gramMatrix(kernel, trainX)`, kept as a separate argument rather than
 * recomputed here so a widget that already built it for display does not pay for it twice.
 */
export function dualRidgeCoefficients(gram: Mat, targets: Vec, lambda: number): number[] {
  const regularised = matAdd(gram, eye(gram.length, lambda));
  const l = cholesky(regularised);
  return solveCholesky(l, targets);
}

/** PRML 6.9: the dual prediction `y(x) = k(x)^T a`. */
export function dualPredict(kernel: KernelFunction, trainX: readonly Vec[], coefficients: Vec, x: Vec): number {
  return dot(kernelVector(kernel, trainX, x), coefficients);
}

/** PRML 6.46: the Nadaraya-Watson weights, the kernel row over training points normalised to sum to one. */
export function nadarayaWatsonWeights(kernel: KernelFunction, trainX: readonly Vec[], x: Vec): number[] {
  const raw = kernelVector(kernel, trainX, x);
  const total = raw.reduce((s, v) => s + v, 0);
  return raw.map((v) => v / total);
}

/** PRML 6.45: the Nadaraya-Watson prediction, the weights dotted with the training targets. */
export function nadarayaWatsonPredict(kernel: KernelFunction, trainX: readonly Vec[], trainT: Vec, x: Vec): number {
  return dot(nadarayaWatsonWeights(kernel, trainX, x), trainT);
}
