import type { Family, Rng } from '../types.js';
import { logGamma, gammaincLower } from '../special.js';
import { standardNormal } from '../rng.js';

/**
 * Shape-rate, matching PRML 2.146, where the gamma is a prior over a Gaussian precision.
 * scipy's `gamma` is shape-scale, so fixtures must pass `scale = 1 / rate`.
 */
export interface GammaParams {
  readonly shape: number;
  readonly rate: number;
}

export function gammaLogPdf(x: number, p: GammaParams): number {
  if (x <= 0) return -Infinity;
  // Guarded so (shape - 1) * log(x) does not become 0 * -Infinity at x -> 0 when shape === 1.
  const powerTerm = p.shape === 1 ? 0 : (p.shape - 1) * Math.log(x);
  return p.shape * Math.log(p.rate) - logGamma(p.shape) + powerTerm - p.rate * x;
}

export function gammaPdf(x: number, p: GammaParams): number {
  return Math.exp(gammaLogPdf(x, p));
}

export function gammaCdf(x: number, p: GammaParams): number {
  if (x <= 0) return 0;
  return gammaincLower(p.shape, p.rate * x);
}

/** Marsaglia-Tsang sample from Gamma(shape, rate = 1). */
function gammaSampleUnitRate(rng: Rng, shape: number): number {
  if (shape < 1) {
    // Boost trick: draw Gamma(shape + 1, 1) then thin by U^(1/shape) to avoid the
    // rejection loop below stalling for small shapes.
    const boosted = gammaSampleUnitRate(rng, shape + 1);
    const u = rng.next();
    return boosted * Math.pow(u, 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = standardNormal(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng.next();
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
      return d * v;
    }
  }
}

/** Marsaglia-Tsang, with the shape < 1 boost applied so small shapes do not loop forever. */
export function gammaSample(rng: Rng, p: GammaParams): number {
  return gammaSampleUnitRate(rng, p.shape) / p.rate;
}

export function gammaMean(p: GammaParams): number {
  return p.shape / p.rate;
}

export function gammaVariance(p: GammaParams): number {
  return p.shape / (p.rate * p.rate);
}

export const Gamma: Family<number, GammaParams> = {
  name: 'gamma',
  logPdf: gammaLogPdf,
  pdf: gammaPdf,
  sample: gammaSample,
  mean: gammaMean,
};
