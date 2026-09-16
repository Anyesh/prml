import type { Mat, Vec } from '../types.js';

export function zeros(n: number): number[] {
  return new Array(n).fill(0);
}

export function matZeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

export function eye(n: number, scale = 1): number[][] {
  const out = matZeros(n, n);
  for (let i = 0; i < n; i++) out[i]![i] = scale;
  return out;
}

export function diag(values: Vec): number[][] {
  const n = values.length;
  const out = matZeros(n, n);
  for (let i = 0; i < n; i++) out[i]![i] = values[i]!;
  return out;
}

/** The diagonal of a square matrix, the inverse of `diag`. */
export function diagOf(a: Mat): number[] {
  return a.map((row, i) => row[i]!);
}

export function transpose(a: Mat): number[][] {
  const rows = a.length;
  const cols = a[0]?.length ?? 0;
  const out = matZeros(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j]![i] = a[i]![j]!;
    }
  }
  return out;
}

export function matmul(a: Mat, b: Mat): number[][] {
  const rows = a.length;
  const inner = b.length;
  const cols = b[0]?.length ?? 0;
  const out = matZeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const aik = a[i]![k]!;
      if (aik === 0) continue;
      const bRow = b[k]!;
      const outRow = out[i]!;
      for (let j = 0; j < cols; j++) {
        outRow[j]! += aik * bRow[j]!;
      }
    }
  }
  return out;
}

export function matvec(a: Mat, x: Vec): number[] {
  return a.map((row) => dot(row, x));
}

/** `x^T A y`. Present separately because forming `A y` then dotting is the hot path in every log-density. */
export function quadForm(x: Vec, a: Mat, y: Vec): number {
  let sum = 0;
  for (let i = 0; i < x.length; i++) {
    sum += x[i]! * dot(a[i]!, y);
  }
  return sum;
}

export function matAdd(a: Mat, b: Mat): number[][] {
  return a.map((row, i) => row.map((v, j) => v + b[i]![j]!));
}

export function matSub(a: Mat, b: Mat): number[][] {
  return a.map((row, i) => row.map((v, j) => v - b[i]![j]!));
}

export function matScale(a: Mat, s: number): number[][] {
  return a.map((row) => row.map((v) => v * s));
}

export function vecAdd(a: Vec, b: Vec): number[] {
  return a.map((v, i) => v + b[i]!);
}

export function vecSub(a: Vec, b: Vec): number[] {
  return a.map((v, i) => v - b[i]!);
}

export function vecScale(a: Vec, s: number): number[] {
  return a.map((v) => v * s);
}

export function dot(a: Vec, b: Vec): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
}

export function outer(a: Vec, b: Vec): number[][] {
  return a.map((av) => b.map((bv) => av * bv));
}

export function norm(a: Vec): number {
  return Math.sqrt(dot(a, a));
}

export function trace(a: Mat): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]![i]!;
  return sum;
}

/** Rows of `a` indexed by `rows`, columns by `cols`. Backs Gaussian marginals and conditionals. */
export function submatrix(a: Mat, rows: readonly number[], cols: readonly number[]): number[][] {
  return rows.map((i) => cols.map((j) => a[i]![j]!));
}

export function subvector(x: Vec, idx: readonly number[]): number[] {
  return idx.map((i) => x[i]!);
}

/** Symmetric to within `tol` in absolute value. Guards the SPD entry points. */
export function isSymmetric(a: Mat, tol = 1e-10): boolean {
  const n = a.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(a[i]![j]! - a[j]![i]!) > tol) return false;
    }
  }
  return true;
}

/**
 * `(A + A^T) / 2`. Repeated rank-one updates in sequential Bayesian learning accumulate
 * asymmetry at the 1e-16 level, which is enough to make Cholesky reject a matrix that
 * is mathematically SPD, so covariance updates pass through this before factorising.
 */
export function symmetrise(a: Mat): number[][] {
  const n = a.length;
  const out = matZeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i]![j] = (a[i]![j]! + a[j]![i]!) / 2;
    }
  }
  return out;
}
