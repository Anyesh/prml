import type { Family, Rng } from '../types.js';
import { besselI } from '../special.js';

export interface VonMisesParams {
  /** Mean direction in radians. */
  readonly mu: number;
  /** Concentration. `kappa` of 0 is uniform on the circle; large `kappa` approaches a Gaussian of variance `1/kappa`. */
  readonly kappa: number;
}

export function vonMisesLogPdf(theta: number, p: VonMisesParams): number {
  return p.kappa * Math.cos(theta - p.mu) - Math.log(2 * Math.PI) - Math.log(besselI(0, p.kappa));
}

export function vonMisesPdf(theta: number, p: VonMisesParams): number {
  return Math.exp(vonMisesLogPdf(theta, p));
}

function wrapAngle(theta: number): number {
  const wrapped = ((theta + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return wrapped <= -Math.PI ? wrapped + 2 * Math.PI : wrapped;
}

/** Best-Fisher rejection sampling. Returns an angle wrapped to (-π, π]. */
export function vonMisesSample(rng: Rng, p: VonMisesParams): number {
  if (p.kappa < 1e-8) {
    // The Best-Fisher envelope divides by kappa, so near-zero concentration must fall
    // back to a direct uniform draw rather than let rho blow up.
    return wrapAngle(p.mu + (rng.next() * 2 - 1) * Math.PI);
  }
  const r = 1 + Math.sqrt(1 + 4 * p.kappa * p.kappa);
  const rho = (r - Math.sqrt(2 * r)) / (2 * p.kappa);
  const s = (1 + rho * rho) / (2 * rho);
  for (;;) {
    const u1 = rng.next();
    const z = Math.cos(Math.PI * u1);
    const f = (1 + s * z) / (s + z);
    const c = p.kappa * (s - f);
    const u2 = rng.next();
    if (c * (2 - c) - u2 > 0 || Math.log(c / u2) + 1 - c >= 0) {
      const u3 = rng.next();
      const sign = u3 > 0.5 ? 1 : -1;
      return wrapAngle(p.mu + sign * Math.acos(f));
    }
  }
}

const KAPPA_BISECTION_LO = 1e-8;
const KAPPA_BISECTION_HI = 1e6;
const KAPPA_ITMAX = 200;

function resultantLength(kappa: number): number {
  return besselI(1, kappa) / besselI(0, kappa);
}

/**
 * Maximum likelihood fit to angles (PRML 2.169, 2.170). `kappa` is recovered from the
 * resultant length by inverting `A(κ) = I₁(κ)/I₀(κ)` numerically; no closed form exists.
 * `A` is monotone increasing on (0, infinity), so bisection on the bracket [1e-8, 1e6]
 * must converge; it stops once the bracket is narrower than 1e-12 relative, comfortably
 * inside the 1e-9 golden tolerance.
 */
export function vonMisesFit(theta: readonly number[]): VonMisesParams {
  let cSum = 0;
  let sSum = 0;
  for (const t of theta) {
    cSum += Math.cos(t);
    sSum += Math.sin(t);
  }
  const n = theta.length;
  const c = cSum / n;
  const s = sSum / n;
  const r = Math.hypot(c, s);
  const mu = Math.atan2(s, c);

  if (r < 1e-12) return { mu, kappa: 0 };
  if (r > 1 - 1e-12) return { mu, kappa: KAPPA_BISECTION_HI };

  let lo = KAPPA_BISECTION_LO;
  let hi = KAPPA_BISECTION_HI;
  for (let i = 0; i < KAPPA_ITMAX; i++) {
    const mid = (lo + hi) / 2;
    if (resultantLength(mid) < r) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12 * hi) break;
  }
  return { mu, kappa: (lo + hi) / 2 };
}

export const VonMises: Family<number, VonMisesParams> = {
  name: 'von-mises',
  logPdf: vonMisesLogPdf,
  pdf: vonMisesPdf,
  sample: vonMisesSample,
  mean: (p) => p.mu,
};
