import { dot } from '../linalg/index.js';
import type { Mat, Vec } from '../types.js';

/**
 * A kernel evaluation, owned here rather than imported from `../kernels/`: that directory
 * is chapter 6's, being written concurrently, and this chapter cannot depend on its
 * signatures landing before its own or staying stable while both are in flight. Every name
 * here is prefixed `sparse` for exactly that reason: chapter 6's `kernels/` module already
 * exports `linearKernel`, `polynomialKernel`, `rbfKernel` and `gramMatrix` with different
 * argument conventions (curried `KernelFunction` factories, a length-scale rather than a
 * `gamma`, an offset default of `0` rather than the book's `1` in 7.42), and the root
 * barrel re-exports both directories, so identical names would collide. Consolidating the
 * two into one kernel module, reconciling those convention differences, is a merge-time job.
 */
export type Kernel = (a: Vec, b: Vec) => number;

/** `k(x, z) = xᵀz`, the feature map that makes an SVM identical to a linear discriminant. */
export function sparseLinearKernel(a: Vec, b: Vec): number {
  return dot(a, b);
}

export interface PolynomialKernelOptions {
  readonly degree: number;
  /** The book's `1` in `(1 + xᵀz)^M`, PRML 7.42. Defaults to that. */
  readonly offset?: number;
}

/** PRML 7.42: `(offset + xᵀz)^degree`, an inner product in a feature space of degree-`M` monomials. */
export function sparsePolynomialKernel(options: PolynomialKernelOptions): Kernel {
  const offset = options.offset ?? 1;
  const degree = options.degree;
  return (a, b) => Math.pow(offset + dot(a, b), degree);
}

/**
 * PRML 6.23 with `gamma = 1/(2*sigma^2)`: `exp(-gamma ||x - x'||^2)`. Named for the
 * parameter actually exposed rather than "Gaussian", since callers tune `gamma` directly.
 */
export function sparseRbfKernel(gamma: number): Kernel {
  return (a, b) => {
    let squared = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i]! - b[i]!;
      squared += d * d;
    }
    return Math.exp(-gamma * squared);
  };
}

/** The symmetric Gram matrix `K[n][m] = k(x_n, x_m)`, computed once per fit. */
export function sparseGramMatrix(kernel: Kernel, points: Mat): number[][] {
  const n = points.length;
  const k: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = kernel(points[i]!, points[j]!);
      k[i]![j] = v;
      k[j]![i] = v;
    }
  }
  return k;
}

/** `k(x, x_n)` for every training point, used to score a new input without rebuilding the Gram matrix. */
export function sparseGramVector(kernel: Kernel, points: Mat, x: Vec): number[] {
  return points.map((p) => kernel(p, x));
}
