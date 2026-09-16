import type { Family, Rng } from '../types.js';
import { logBeta, betainc } from '../special.js';
import { gammaSample } from './gamma.js';

export interface BetaParams {
  readonly a: number;
  readonly b: number;
}

export function betaLogPdf(x: number, p: BetaParams): number {
  if (x < 0 || x > 1) return -Infinity;
  // Guarded so (a - 1) * log(0) does not become 0 * -Infinity = NaN when a === 1.
  const left = p.a === 1 ? 0 : (p.a - 1) * Math.log(x);
  const right = p.b === 1 ? 0 : (p.b - 1) * Math.log(1 - x);
  return left + right - logBeta(p.a, p.b);
}

export function betaPdf(x: number, p: BetaParams): number {
  return Math.exp(betaLogPdf(x, p));
}

export function betaCdf(x: number, p: BetaParams): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return betainc(x, p.a, p.b);
}

export function betaSample(rng: Rng, p: BetaParams): number {
  const x = gammaSample(rng, { shape: p.a, rate: 1 });
  const y = gammaSample(rng, { shape: p.b, rate: 1 });
  return x / (x + y);
}

export function betaMean(p: BetaParams): number {
  return p.a / (p.a + p.b);
}

export function betaVariance(p: BetaParams): number {
  const s = p.a + p.b;
  return (p.a * p.b) / (s * s * (s + 1));
}

/**
 * Conjugate update after `m` successes in `m + l` Bernoulli trials (PRML 2.18). The
 * whole point of the coin-flip widget is that this is addition, so it must not be
 * implemented as a re-fit.
 */
export function betaPosterior(prior: BetaParams, successes: number, failures: number): BetaParams {
  return { a: prior.a + successes, b: prior.b + failures };
}

export const Beta: Family<number, BetaParams> = {
  name: 'beta',
  logPdf: betaLogPdf,
  pdf: betaPdf,
  sample: betaSample,
  mean: betaMean,
};
