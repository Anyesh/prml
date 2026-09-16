import type { Mat, Vec } from '../types.js';
import { NotImplemented } from '../types.js';

export function zeros(n: number): number[] {
  void n;
  throw new NotImplemented('zeros');
}

export function matZeros(rows: number, cols: number): number[][] {
  void rows;
  void cols;
  throw new NotImplemented('matZeros');
}

export function eye(n: number, scale = 1): number[][] {
  void n;
  void scale;
  throw new NotImplemented('eye');
}

export function diag(values: Vec): number[][] {
  void values;
  throw new NotImplemented('diag');
}

/** The diagonal of a square matrix, the inverse of `diag`. */
export function diagOf(a: Mat): number[] {
  void a;
  throw new NotImplemented('diagOf');
}

export function transpose(a: Mat): number[][] {
  void a;
  throw new NotImplemented('transpose');
}

export function matmul(a: Mat, b: Mat): number[][] {
  void a;
  void b;
  throw new NotImplemented('matmul');
}

export function matvec(a: Mat, x: Vec): number[] {
  void a;
  void x;
  throw new NotImplemented('matvec');
}

/** `x^T A y`. Present separately because forming `A y` then dotting is the hot path in every log-density. */
export function quadForm(x: Vec, a: Mat, y: Vec): number {
  void x;
  void a;
  void y;
  throw new NotImplemented('quadForm');
}

export function matAdd(a: Mat, b: Mat): number[][] {
  void a;
  void b;
  throw new NotImplemented('matAdd');
}

export function matSub(a: Mat, b: Mat): number[][] {
  void a;
  void b;
  throw new NotImplemented('matSub');
}

export function matScale(a: Mat, s: number): number[][] {
  void a;
  void s;
  throw new NotImplemented('matScale');
}

export function vecAdd(a: Vec, b: Vec): number[] {
  void a;
  void b;
  throw new NotImplemented('vecAdd');
}

export function vecSub(a: Vec, b: Vec): number[] {
  void a;
  void b;
  throw new NotImplemented('vecSub');
}

export function vecScale(a: Vec, s: number): number[] {
  void a;
  void s;
  throw new NotImplemented('vecScale');
}

export function dot(a: Vec, b: Vec): number {
  void a;
  void b;
  throw new NotImplemented('dot');
}

export function outer(a: Vec, b: Vec): number[][] {
  void a;
  void b;
  throw new NotImplemented('outer');
}

export function norm(a: Vec): number {
  void a;
  throw new NotImplemented('norm');
}

export function trace(a: Mat): number {
  void a;
  throw new NotImplemented('trace');
}

/** Rows of `a` indexed by `rows`, columns by `cols`. Backs Gaussian marginals and conditionals. */
export function submatrix(a: Mat, rows: readonly number[], cols: readonly number[]): number[][] {
  void a;
  void rows;
  void cols;
  throw new NotImplemented('submatrix');
}

export function subvector(x: Vec, idx: readonly number[]): number[] {
  void x;
  void idx;
  throw new NotImplemented('subvector');
}

/** Symmetric to within `tol` in absolute value. Guards the SPD entry points. */
export function isSymmetric(a: Mat, tol = 1e-10): boolean {
  void a;
  void tol;
  throw new NotImplemented('isSymmetric');
}

/**
 * `(A + A^T) / 2`. Repeated rank-one updates in sequential Bayesian learning accumulate
 * asymmetry at the 1e-16 level, which is enough to make Cholesky reject a matrix that
 * is mathematically SPD, so covariance updates pass through this before factorising.
 */
export function symmetrise(a: Mat): number[][] {
  void a;
  throw new NotImplemented('symmetrise');
}
