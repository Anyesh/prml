import type { Rng, Vec } from '../types.js';
import { logGamma } from '../special.js';

export interface BernoulliParams {
  readonly mu: number;
}

export function bernoulliLogPmf(x: 0 | 1, p: BernoulliParams): number {
  return x === 1 ? Math.log(p.mu) : Math.log(1 - p.mu);
}

export function bernoulliSample(rng: Rng, p: BernoulliParams): 0 | 1 {
  return rng.next() < p.mu ? 1 : 0;
}

export interface BinomialParams {
  readonly n: number;
  readonly mu: number;
}

/** `log(n choose k)` via `logGamma`, so that n beyond 170 does not overflow the factorial. */
export function logBinomialCoefficient(n: number, k: number): number {
  return logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1);
}

export function binomialLogPmf(m: number, p: BinomialParams): number {
  if (m < 0 || m > p.n) return -Infinity;
  const logMu = m === 0 ? 0 : m * Math.log(p.mu);
  const logOneMinusMu = m === p.n ? 0 : (p.n - m) * Math.log(1 - p.mu);
  return logBinomialCoefficient(p.n, m) + logMu + logOneMinusMu;
}

export function binomialPmf(m: number, p: BinomialParams): number {
  return Math.exp(binomialLogPmf(m, p));
}

export interface MultinomialParams {
  readonly n: number;
  readonly probs: Vec;
}

export function multinomialLogPmf(counts: Vec, p: MultinomialParams): number {
  let logCoef = logGamma(p.n + 1);
  let sum = 0;
  for (let i = 0; i < counts.length; i++) {
    const ci = counts[i]!;
    logCoef -= logGamma(ci + 1);
    sum += ci === 0 ? 0 : ci * Math.log(p.probs[i]!);
  }
  return logCoef + sum;
}

export function multinomialSample(rng: Rng, p: MultinomialParams): number[] {
  const counts = new Array<number>(p.probs.length).fill(0);
  for (let t = 0; t < p.n; t++) {
    const u = rng.next();
    let cumulative = 0;
    let chosen = p.probs.length - 1;
    for (let i = 0; i < p.probs.length; i++) {
      cumulative += p.probs[i]!;
      if (u < cumulative) {
        chosen = i;
        break;
      }
    }
    counts[chosen]!++;
  }
  return counts;
}
