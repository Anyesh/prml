import type { Family, Rng } from '../types.js';
import { NotImplemented } from '../types.js';

export interface NormalParams {
  readonly mu: number;
  /** Variance, not standard deviation. PRML parameterises by precision or variance throughout; σ never appears alone. */
  readonly sigma2: number;
}

export function normalLogPdf(x: number, p: NormalParams): number {
  void x;
  void p;
  throw new NotImplemented('normalLogPdf');
}

export function normalPdf(x: number, p: NormalParams): number {
  void x;
  void p;
  throw new NotImplemented('normalPdf');
}

export function normalCdf(x: number, p: NormalParams): number {
  void x;
  void p;
  throw new NotImplemented('normalCdf');
}

/** Inverse CDF. Needed for credible-interval ribbons, which are drawn at fixed quantiles. */
export function normalQuantile(q: number, p: NormalParams): number {
  void q;
  void p;
  throw new NotImplemented('normalQuantile');
}

export function normalSample(rng: Rng, p: NormalParams): number {
  void rng;
  void p;
  throw new NotImplemented('normalSample');
}

export const Normal: Family<number, NormalParams> = {
  name: 'normal',
  logPdf: normalLogPdf,
  pdf: normalPdf,
  sample: normalSample,
  mean: (p) => p.mu,
};
