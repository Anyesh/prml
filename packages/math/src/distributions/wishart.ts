import type { Family, Mat, Rng } from '../types.js';
import { NotImplemented } from '../types.js';

export interface WishartParams {
  /** Scale matrix `W`, symmetric positive definite. */
  readonly scale: Mat;
  /** Degrees of freedom; must exceed `d - 1` for the density to be proper. */
  readonly nu: number;
}

export function wishartLogPdf(x: Mat, p: WishartParams): number {
  void x;
  void p;
  throw new NotImplemented('wishartLogPdf');
}

export function wishartPdf(x: Mat, p: WishartParams): number {
  void x;
  void p;
  throw new NotImplemented('wishartPdf');
}

/** Bartlett decomposition, which costs one Cholesky rather than `nu` outer products. */
export function wishartSample(rng: Rng, p: WishartParams): number[][] {
  void rng;
  void p;
  throw new NotImplemented('wishartSample');
}

export function wishartMean(p: WishartParams): number[][] {
  void p;
  throw new NotImplemented('wishartMean');
}

/** `E[ln|Λ|]` under the Wishart (PRML 10.65), the term the variational GMM lower bound needs. */
export function wishartExpectedLogDet(p: WishartParams): number {
  void p;
  throw new NotImplemented('wishartExpectedLogDet');
}

export const Wishart: Family<Mat, WishartParams> = {
  name: 'wishart',
  logPdf: wishartLogPdf,
  pdf: wishartPdf,
  sample: wishartSample,
  mean: wishartMean,
};
