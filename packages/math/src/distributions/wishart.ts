import { standardNormal } from '../rng.js';
import { digamma, logMultivariateGamma } from '../special.js';
import type { Family, Mat, Rng } from '../types.js';
import { matmul, matZeros, trace, transpose } from '../linalg/core.js';
import { cholesky, logDet, solveMat } from '../linalg/decompose.js';
import { gammaSample } from './gamma.js';

export interface WishartParams {
  /** Scale matrix `W`, symmetric positive definite. */
  readonly scale: Mat;
  /** Degrees of freedom; must exceed `d - 1` for the density to be proper. */
  readonly nu: number;
}

export function wishartLogPdf(x: Mat, p: WishartParams): number {
  const d = p.scale.length;
  const logDetW = logDet(p.scale);
  const logDetX = logDet(x);
  const traceTerm = trace(solveMat(p.scale, x));
  const logNormaliser =
    -(p.nu / 2) * logDetW - ((p.nu * d) / 2) * Math.log(2) - logMultivariateGamma(p.nu / 2, d);
  return logNormaliser + ((p.nu - d - 1) / 2) * logDetX - 0.5 * traceTerm;
}

export function wishartPdf(x: Mat, p: WishartParams): number {
  return Math.exp(wishartLogPdf(x, p));
}

function bartlettFactor(rng: Rng, nu: number, d: number): number[][] {
  const a = matZeros(d, d);
  for (let i = 0; i < d; i++) {
    // Row i (0-indexed) holds the chi-squared draw with (nu - i) degrees of freedom,
    // per the Bartlett decomposition; chi-squared(k) is Gamma(shape=k/2, rate=1/2).
    a[i]![i] = Math.sqrt(gammaSample(rng, { shape: (nu - i) / 2, rate: 0.5 }));
    for (let j = 0; j < i; j++) {
      a[i]![j] = standardNormal(rng);
    }
  }
  return a;
}

/** Bartlett decomposition, which costs one Cholesky rather than `nu` outer products. */
export function wishartSample(rng: Rng, p: WishartParams): number[][] {
  const d = p.scale.length;
  const l = cholesky(p.scale);
  const a = bartlettFactor(rng, p.nu, d);
  const la = matmul(l, a);
  return matmul(la, transpose(la));
}

export function wishartMean(p: WishartParams): number[][] {
  return p.scale.map((row) => row.map((v) => v * p.nu));
}

/** `E[ln|Λ|]` under the Wishart (PRML 10.65), the term the variational GMM lower bound needs. */
export function wishartExpectedLogDet(p: WishartParams): number {
  const d = p.scale.length;
  let sum = 0;
  for (let i = 1; i <= d; i++) sum += digamma((p.nu + 1 - i) / 2);
  return sum + d * Math.log(2) + logDet(p.scale);
}

export const Wishart: Family<Mat, WishartParams> = {
  name: 'wishart',
  logPdf: wishartLogPdf,
  pdf: wishartPdf,
  sample: wishartSample,
  mean: wishartMean,
};
