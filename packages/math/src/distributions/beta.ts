import type { Family, Rng } from '../types.js';
import { NotImplemented } from '../types.js';

export interface BetaParams {
  readonly a: number;
  readonly b: number;
}

export function betaLogPdf(x: number, p: BetaParams): number {
  void x;
  void p;
  throw new NotImplemented('betaLogPdf');
}

export function betaPdf(x: number, p: BetaParams): number {
  void x;
  void p;
  throw new NotImplemented('betaPdf');
}

export function betaCdf(x: number, p: BetaParams): number {
  void x;
  void p;
  throw new NotImplemented('betaCdf');
}

export function betaSample(rng: Rng, p: BetaParams): number {
  void rng;
  void p;
  throw new NotImplemented('betaSample');
}

export function betaMean(p: BetaParams): number {
  void p;
  throw new NotImplemented('betaMean');
}

export function betaVariance(p: BetaParams): number {
  void p;
  throw new NotImplemented('betaVariance');
}

/**
 * Conjugate update after `m` successes in `m + l` Bernoulli trials (PRML 2.18). The
 * whole point of the coin-flip widget is that this is addition, so it must not be
 * implemented as a re-fit.
 */
export function betaPosterior(prior: BetaParams, successes: number, failures: number): BetaParams {
  void prior;
  void successes;
  void failures;
  throw new NotImplemented('betaPosterior');
}

export const Beta: Family<number, BetaParams> = {
  name: 'beta',
  logPdf: betaLogPdf,
  pdf: betaPdf,
  sample: betaSample,
  mean: betaMean,
};
