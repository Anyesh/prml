import {
  CholeskyDecomposition,
  EigenvalueDecomposition,
  LuDecomposition,
  Matrix,
  SingularValueDecomposition,
} from 'ml-matrix';
import type { Mat, Vec } from '../types.js';
import { matmul, matScale, transpose } from './core.js';

/**
 * Lower-triangular `L` with `A = L Lᵀ`. Throws when `a` is not positive definite rather
 * than returning NaNs, because a silently NaN-poisoned covariance surfaces three widgets
 * downstream as a blank canvas with no error.
 */
export function cholesky(a: Mat): number[][] {
  const cho = new CholeskyDecomposition(new Matrix(a.map((row) => [...row])));
  if (!cho.isPositiveDefinite()) {
    throw new Error('cholesky: matrix is not positive definite');
  }
  return cho.lowerTriangularMatrix.to2DArray();
}

/**
 * Adds `eps` to the diagonal. GP covariance matrices over near-duplicate inputs are
 * numerically singular, and this is the standard fix before factorising.
 */
export function jitter(a: Mat, eps = 1e-8): number[][] {
  return a.map((row, i) => row.map((v, j) => (i === j ? v + eps : v)));
}

/** Solves `A x = b` by LU. For symmetric positive-definite `A`, prefer `solveCholesky`. */
export function solve(a: Mat, b: Vec): number[] {
  const lu = new LuDecomposition(new Matrix(a.map((row) => [...row])));
  const x = lu.solve(Matrix.columnVector([...b]));
  return x.to1DArray();
}

export function solveMat(a: Mat, b: Mat): number[][] {
  const lu = new LuDecomposition(new Matrix(a.map((row) => [...row])));
  const x = lu.solve(new Matrix(b.map((row) => [...row])));
  return x.to2DArray();
}

/** Solves `A x = b` given `L` from `cholesky(A)`, by forward then back substitution. */
export function solveCholesky(l: Mat, b: Vec): number[] {
  const n = l.length;
  const y = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i]!;
    for (let k = 0; k < i; k++) sum -= l[i]![k]! * y[k]!;
    y[i] = sum / l[i]![i]!;
  }
  const x = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i]!;
    for (let k = i + 1; k < n; k++) sum -= l[k]![i]! * x[k]!;
    x[i] = sum / l[i]![i]!;
  }
  return x;
}

/**
 * Explicit inverse. Callers that only need `A⁻¹ b` must use `solve` instead: the inverse
 * is present for the handful of places PRML displays one (the equivalent kernel, the
 * posterior covariance a widget draws as an ellipse), not as a way to solve systems.
 */
export function inverse(a: Mat): number[][] {
  return solveMat(a, identity(a.length));
}

export function det(a: Mat): number {
  return new LuDecomposition(new Matrix(a.map((row) => [...row]))).determinant;
}

/**
 * `log|A|` for symmetric positive-definite `A`, via `2 Σ log Lᵢᵢ`. Required because the
 * determinant of a 50×50 GP kernel matrix underflows to zero long before its log does.
 */
export function logDet(a: Mat): number {
  const l = cholesky(a);
  let sum = 0;
  for (let i = 0; i < l.length; i++) sum += Math.log(l[i]![i]!);
  return 2 * sum;
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
  const evd = new EigenvalueDecomposition(new Matrix(a.map((row) => [...row])), {
    assumeSymmetric: true,
  });
  const n = a.length;
  const ascending = evd.realEigenvalues;
  const vecCols = evd.eigenvectorMatrix;

  const order = ascending.map((_, i) => i).sort((i, j) => ascending[j]! - ascending[i]!);
  const values = order.map((i) => ascending[i]!);
  const vectors = order.map((col) => {
    const row = new Array<number>(n);
    for (let r = 0; r < n; r++) row[r] = vecCols.get(r, col);
    let maxIdx = 0;
    for (let i = 1; i < n; i++) {
      if (Math.abs(row[i]!) > Math.abs(row[maxIdx]!)) maxIdx = i;
    }
    if (row[maxIdx]! < 0) {
      for (let i = 0; i < n; i++) row[i] = -row[i]!;
    }
    return row;
  });
  return { values, vectors };
}

export interface Svd {
  readonly u: number[][];
  /** Singular values in descending order. */
  readonly s: number[];
  /** `v`, not `vᵀ`. */
  readonly v: number[][];
}

export function svd(a: Mat): Svd {
  const decomp = new SingularValueDecomposition(new Matrix(a.map((row) => [...row])));
  return {
    u: decomp.leftSingularVectors.to2DArray(),
    s: decomp.diagonal,
    v: decomp.rightSingularVectors.to2DArray(),
  };
}

/** Moore-Penrose pseudo-inverse via SVD, with singular values below `rcond * s[0]` dropped. */
export function pinv(a: Mat, rcond = 1e-12): number[][] {
  const { u, s, v } = svd(a);
  const threshold = rcond * (s[0] ?? 0);
  const sInv = s.map((sv) => (sv > threshold ? 1 / sv : 0));
  const uScaled = transpose(u).map((col, i) => col.map((v) => v * sInv[i]!));
  return matmul(v, uScaled);
}

function identity(n: number): number[][] {
  const out: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) out[i]![i] = 1;
  return out;
}
