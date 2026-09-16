import type { Family, Rng, Vec } from '../types.js';
import { logGamma, digamma } from '../special.js';
import { gammaSample } from './gamma.js';

export interface DirichletParams {
  readonly alpha: Vec;
}

function logNormaliser(alpha: Vec): number {
  let sumAlpha = 0;
  let sumLogGamma = 0;
  for (const a of alpha) {
    sumAlpha += a;
    sumLogGamma += logGamma(a);
  }
  return sumLogGamma - logGamma(sumAlpha);
}

/** Density with respect to Lebesgue measure on the simplex; `x` must sum to 1 and is not renormalised. */
export function dirichletLogPdf(x: Vec, p: DirichletParams): number {
  let sum = -logNormaliser(p.alpha);
  for (let k = 0; k < p.alpha.length; k++) {
    const a = p.alpha[k]!;
    const xk = x[k]!;
    // Guarded so (a - 1) * log(0) does not become 0 * -Infinity at a vertex when a === 1.
    sum += a === 1 ? 0 : (a - 1) * Math.log(xk);
  }
  return sum;
}

export function dirichletPdf(x: Vec, p: DirichletParams): number {
  return Math.exp(dirichletLogPdf(x, p));
}

export function dirichletSample(rng: Rng, p: DirichletParams): number[] {
  const draws = p.alpha.map((a) => gammaSample(rng, { shape: a, rate: 1 }));
  const total = draws.reduce((s, d) => s + d, 0);
  return draws.map((d) => d / total);
}

export function dirichletMean(p: DirichletParams): number[] {
  let total = 0;
  for (const a of p.alpha) total += a;
  return p.alpha.map((a) => a / total);
}

/** `E[ln x_k]` under the Dirichlet, `ψ(α_k) - ψ(Σα)`. The term variational Bayes needs (PRML 10.66). */
export function dirichletExpectedLog(p: DirichletParams): number[] {
  let total = 0;
  for (const a of p.alpha) total += a;
  const digammaTotal = digamma(total);
  return p.alpha.map((a) => digamma(a) - digammaTotal);
}

export const Dirichlet: Family<Vec, DirichletParams> = {
  name: 'dirichlet',
  logPdf: dirichletLogPdf,
  pdf: dirichletPdf,
  sample: dirichletSample,
  mean: dirichletMean,
};
