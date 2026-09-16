import type { Family, Mat, Rng, Vec } from '../types.js';
import { NotImplemented } from '../types.js';

export interface StudentTParams {
  readonly mu: number;
  /** Squared scale, not variance: the variance is `nu / (nu - 2)` times this and is undefined for `nu <= 2`. */
  readonly scale2: number;
  readonly nu: number;
}

export function studentTLogPdf(x: number, p: StudentTParams): number {
  void x;
  void p;
  throw new NotImplemented('studentTLogPdf');
}

export function studentTPdf(x: number, p: StudentTParams): number {
  void x;
  void p;
  throw new NotImplemented('studentTPdf');
}

export function studentTCdf(x: number, p: StudentTParams): number {
  void x;
  void p;
  throw new NotImplemented('studentTCdf');
}

export function studentTSample(rng: Rng, p: StudentTParams): number {
  void rng;
  void p;
  throw new NotImplemented('studentTSample');
}

export interface MultivariateTParams {
  readonly mean: Vec;
  readonly scale: Mat;
  readonly nu: number;
}

/** PRML 2.162. The predictive density of a Gaussian with unknown precision, so it is not optional. */
export function multivariateTLogPdf(x: Vec, p: MultivariateTParams): number {
  void x;
  void p;
  throw new NotImplemented('multivariateTLogPdf');
}

export const StudentT: Family<number, StudentTParams> = {
  name: 'student-t',
  logPdf: studentTLogPdf,
  pdf: studentTPdf,
  sample: studentTSample,
  mean: (p) => p.mu,
};
