import type { Family, Rng, Vec } from '../types.js';
import { NotImplemented } from '../types.js';

export interface DirichletParams {
  readonly alpha: Vec;
}

/** Density with respect to Lebesgue measure on the simplex; `x` must sum to 1 and is not renormalised. */
export function dirichletLogPdf(x: Vec, p: DirichletParams): number {
  void x;
  void p;
  throw new NotImplemented('dirichletLogPdf');
}

export function dirichletPdf(x: Vec, p: DirichletParams): number {
  void x;
  void p;
  throw new NotImplemented('dirichletPdf');
}

export function dirichletSample(rng: Rng, p: DirichletParams): number[] {
  void rng;
  void p;
  throw new NotImplemented('dirichletSample');
}

export function dirichletMean(p: DirichletParams): number[] {
  void p;
  throw new NotImplemented('dirichletMean');
}

/** `E[ln x_k]` under the Dirichlet, `ψ(α_k) - ψ(Σα)`. The term variational Bayes needs (PRML 10.66). */
export function dirichletExpectedLog(p: DirichletParams): number[] {
  void p;
  throw new NotImplemented('dirichletExpectedLog');
}

export const Dirichlet: Family<Vec, DirichletParams> = {
  name: 'dirichlet',
  logPdf: dirichletLogPdf,
  pdf: dirichletPdf,
  sample: dirichletSample,
  mean: dirichletMean,
};
