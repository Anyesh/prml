import type { Family, Rng } from '../types.js';
import { NotImplemented } from '../types.js';

/**
 * Shape-rate, matching PRML 2.146, where the gamma is a prior over a Gaussian precision.
 * scipy's `gamma` is shape-scale, so fixtures must pass `scale = 1 / rate`.
 */
export interface GammaParams {
  readonly shape: number;
  readonly rate: number;
}

export function gammaLogPdf(x: number, p: GammaParams): number {
  void x;
  void p;
  throw new NotImplemented('gammaLogPdf');
}

export function gammaPdf(x: number, p: GammaParams): number {
  void x;
  void p;
  throw new NotImplemented('gammaPdf');
}

export function gammaCdf(x: number, p: GammaParams): number {
  void x;
  void p;
  throw new NotImplemented('gammaCdf');
}

/** Marsaglia-Tsang, with the shape < 1 boost applied so small shapes do not loop forever. */
export function gammaSample(rng: Rng, p: GammaParams): number {
  void rng;
  void p;
  throw new NotImplemented('gammaSample');
}

export function gammaMean(p: GammaParams): number {
  void p;
  throw new NotImplemented('gammaMean');
}

export function gammaVariance(p: GammaParams): number {
  void p;
  throw new NotImplemented('gammaVariance');
}

export const Gamma: Family<number, GammaParams> = {
  name: 'gamma',
  logPdf: gammaLogPdf,
  pdf: gammaPdf,
  sample: gammaSample,
  mean: gammaMean,
};
