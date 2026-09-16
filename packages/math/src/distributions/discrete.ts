import type { Rng, Vec } from '../types.js';
import { NotImplemented } from '../types.js';

export interface BernoulliParams {
  readonly mu: number;
}

export function bernoulliLogPmf(x: 0 | 1, p: BernoulliParams): number {
  void x;
  void p;
  throw new NotImplemented('bernoulliLogPmf');
}

export function bernoulliSample(rng: Rng, p: BernoulliParams): 0 | 1 {
  void rng;
  void p;
  throw new NotImplemented('bernoulliSample');
}

export interface BinomialParams {
  readonly n: number;
  readonly mu: number;
}

export function binomialLogPmf(m: number, p: BinomialParams): number {
  void m;
  void p;
  throw new NotImplemented('binomialLogPmf');
}

export function binomialPmf(m: number, p: BinomialParams): number {
  void m;
  void p;
  throw new NotImplemented('binomialPmf');
}

export interface MultinomialParams {
  readonly n: number;
  readonly probs: Vec;
}

export function multinomialLogPmf(counts: Vec, p: MultinomialParams): number {
  void counts;
  void p;
  throw new NotImplemented('multinomialLogPmf');
}

export function multinomialSample(rng: Rng, p: MultinomialParams): number[] {
  void rng;
  void p;
  throw new NotImplemented('multinomialSample');
}

/** `log(n choose k)` via `logGamma`, so that n beyond 170 does not overflow the factorial. */
export function logBinomialCoefficient(n: number, k: number): number {
  void n;
  void k;
  throw new NotImplemented('logBinomialCoefficient');
}
