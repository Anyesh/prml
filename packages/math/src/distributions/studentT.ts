import type { Family, Mat, Rng, Vec } from '../types.js';
import { logGamma, betainc } from '../special.js';
import { standardNormal } from '../rng.js';
import { solve, logDet } from '../linalg/decompose.js';
import { dot, vecSub } from '../linalg/core.js';
import { gammaSample } from './gamma.js';

export interface StudentTParams {
  readonly mu: number;
  /** Squared scale, not variance: the variance is `nu / (nu - 2)` times this and is undefined for `nu <= 2`. */
  readonly scale2: number;
  readonly nu: number;
}

export function studentTLogPdf(x: number, p: StudentTParams): number {
  const d = x - p.mu;
  return (
    logGamma((p.nu + 1) / 2) -
    logGamma(p.nu / 2) -
    0.5 * Math.log(p.nu * Math.PI * p.scale2) -
    ((p.nu + 1) / 2) * Math.log(1 + (d * d) / (p.nu * p.scale2))
  );
}

export function studentTPdf(x: number, p: StudentTParams): number {
  return Math.exp(studentTLogPdf(x, p));
}

export function studentTCdf(x: number, p: StudentTParams): number {
  const t = (x - p.mu) / Math.sqrt(p.scale2);
  const ib = betainc(p.nu / (p.nu + t * t), p.nu / 2, 0.5);
  return t >= 0 ? 1 - 0.5 * ib : 0.5 * ib;
}

export function studentTSample(rng: Rng, p: StudentTParams): number {
  const z = standardNormal(rng);
  const chi2 = 2 * gammaSample(rng, { shape: p.nu / 2, rate: 0.5 });
  return p.mu + Math.sqrt(p.scale2) * (z / Math.sqrt(chi2 / p.nu));
}

export interface MultivariateTParams {
  readonly mean: Vec;
  readonly scale: Mat;
  readonly nu: number;
}

/** PRML 2.162. The predictive density of a Gaussian with unknown precision, so it is not optional. */
export function multivariateTLogPdf(x: Vec, p: MultivariateTParams): number {
  const d = x.length;
  const diff = vecSub(x, p.mean);
  const delta = dot(diff, solve(p.scale, diff));
  return (
    logGamma((p.nu + d) / 2) -
    logGamma(p.nu / 2) -
    (d / 2) * Math.log(p.nu * Math.PI) -
    0.5 * logDet(p.scale) -
    ((p.nu + d) / 2) * Math.log(1 + delta / p.nu)
  );
}

export const StudentT: Family<number, StudentTParams> = {
  name: 'student-t',
  logPdf: studentTLogPdf,
  pdf: studentTPdf,
  sample: studentTSample,
  mean: (p) => p.mu,
};
