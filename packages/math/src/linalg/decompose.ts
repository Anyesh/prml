import type { Mat, Vec } from '../types.js';
import { NotImplemented } from '../types.js';

/**
 * Lower-triangular `L` with `A = L Lᵀ`. Throws when `a` is not positive definite rather
 * than returning NaNs, because a silently NaN-poisoned covariance surfaces three widgets
 * downstream as a blank canvas with no error.
 */
export function cholesky(a: Mat): number[][] {
  void a;
  throw new NotImplemented('cholesky');
}

/**
 * Adds `eps` to the diagonal. GP covariance matrices over near-duplicate inputs are
 * numerically singular, and this is the standard fix before factorising.
 */
export function jitter(a: Mat, eps = 1e-8): number[][] {
  void a;
  void eps;
  throw new NotImplemented('jitter');
}

/** Solves `A x = b` by LU. For symmetric positive-definite `A`, prefer `solveCholesky`. */
export function solve(a: Mat, b: Vec): number[] {
  void a;
  void b;
  throw new NotImplemented('solve');
}

export function solveMat(a: Mat, b: Mat): number[][] {
  void a;
  void b;
  throw new NotImplemented('solveMat');
}

/** Solves `A x = b` given `L` from `cholesky(A)`, by forward then back substitution. */
export function solveCholesky(l: Mat, b: Vec): number[] {
  void l;
  void b;
  throw new NotImplemented('solveCholesky');
}

/**
 * Explicit inverse. Callers that only need `A⁻¹ b` must use `solve` instead: the inverse
 * is present for the handful of places PRML displays one (the equivalent kernel, the
 * posterior covariance a widget draws as an ellipse), not as a way to solve systems.
 */
export function inverse(a: Mat): number[][] {
  void a;
  throw new NotImplemented('inverse');
}

export function det(a: Mat): number {
  void a;
  throw new NotImplemented('det');
}

/**
 * `log|A|` for symmetric positive-definite `A`, via `2 Σ log Lᵢᵢ`. Required because the
 * determinant of a 50×50 GP kernel matrix underflows to zero long before its log does.
 */
export function logDet(a: Mat): number {
  void a;
  throw new NotImplemented('logDet');
}

export interface EigenSym {
  /** Eigenvalues in descending order. */
  readonly values: number[];
  /** `vectors[i]` is the eigenvector for `values[i]`, unit length, sign-fixed so its largest-magnitude entry is positive. */
  readonly vectors: number[][];
}

/**
 * Symmetric eigendecomposition. Vectors are returned as rows rather than columns, which
 * is the opposite of numpy's `eigh`, because every consumer here iterates principal
 * directions. Golden fixtures must transpose before comparing.
 *
 * The sign convention is not cosmetic: eigenvectors are determined only up to sign, so
 * without fixing one, PCA axes flip arbitrarily between frames as the user drags data.
 */
export function eigSym(a: Mat): EigenSym {
  void a;
  throw new NotImplemented('eigSym');
}

export interface Svd {
  readonly u: number[][];
  /** Singular values in descending order. */
  readonly s: number[];
  /** `v`, not `vᵀ`. */
  readonly v: number[][];
}

export function svd(a: Mat): Svd {
  void a;
  throw new NotImplemented('svd');
}

/** Moore-Penrose pseudo-inverse via SVD, with singular values below `rcond * s[0]` dropped. */
export function pinv(a: Mat, rcond = 1e-12): number[][] {
  void a;
  void rcond;
  throw new NotImplemented('pinv');
}
