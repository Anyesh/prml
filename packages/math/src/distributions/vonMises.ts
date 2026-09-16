import type { Family, Rng } from '../types.js';
import { NotImplemented } from '../types.js';

export interface VonMisesParams {
  /** Mean direction in radians. */
  readonly mu: number;
  /** Concentration. `kappa` of 0 is uniform on the circle; large `kappa` approaches a Gaussian of variance `1/kappa`. */
  readonly kappa: number;
}

export function vonMisesLogPdf(theta: number, p: VonMisesParams): number {
  void theta;
  void p;
  throw new NotImplemented('vonMisesLogPdf');
}

export function vonMisesPdf(theta: number, p: VonMisesParams): number {
  void theta;
  void p;
  throw new NotImplemented('vonMisesPdf');
}

/** Best-Fisher rejection sampling. Returns an angle wrapped to (-π, π]. */
export function vonMisesSample(rng: Rng, p: VonMisesParams): number {
  void rng;
  void p;
  throw new NotImplemented('vonMisesSample');
}

/**
 * Maximum likelihood fit to angles (PRML 2.169, 2.170). `kappa` is recovered from the
 * resultant length by inverting `A(κ) = I₁(κ)/I₀(κ)` numerically; no closed form exists.
 */
export function vonMisesFit(theta: readonly number[]): VonMisesParams {
  void theta;
  throw new NotImplemented('vonMisesFit');
}

export const VonMises: Family<number, VonMisesParams> = {
  name: 'von-mises',
  logPdf: vonMisesLogPdf,
  pdf: vonMisesPdf,
  sample: vonMisesSample,
  mean: (p) => p.mu,
};
