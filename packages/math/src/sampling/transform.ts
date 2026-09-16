import type { Rng } from '../types.js';

/**
 * PRML 11.1.1: the transformation method. Given a uniform `z` on (0, 1), transform it
 * through the inverse of the target's indefinite integral (11.6) to obtain a sample
 * with the target density. Only useful where that inverse CDF has a closed form, which
 * is why rejection and importance sampling exist for everything else.
 */
export function inverseTransformSample(rng: Rng, quantile: (u: number) => number): number {
  return quantile(rng.next());
}

/** PRML 11.6-11.7: `h(y) = 1 - exp(-lambda y)`, so `h^-1(z) = -ln(1 - z) / lambda`. */
export function exponentialQuantile(u: number, lambda: number): number {
  return -Math.log(1 - u) / lambda;
}

export function exponentialLogPdf(x: number, lambda: number): number {
  if (x < 0) return -Infinity;
  return Math.log(lambda) - lambda * x;
}

/** PRML 11.7: transforming a uniform variable through the exponential quantile (11.6). */
export function exponentialSample(rng: Rng, lambda: number): number {
  return inverseTransformSample(rng, (u) => exponentialQuantile(u, lambda));
}

export interface CauchyParams {
  /** Scale, PRML's `b` in (11.16). */
  readonly b: number;
  /** Location, PRML's `c`. */
  readonly c: number;
}

/** PRML 11.8, located and scaled: `p(y) = 1 / (pi b (1 + ((y - c) / b)^2))`. */
export function cauchyPdf(x: number, p: CauchyParams): number {
  const z = (x - p.c) / p.b;
  return 1 / (Math.PI * p.b * (1 + z * z));
}

/**
 * Inverse CDF of the Cauchy (exercise 11.7): the standard Cauchy CDF is
 * `1/2 + atan(y)/pi`, so its inverse is `tan(pi (u - 1/2))`, located and scaled by `c`
 * and `b`. This is the proposal sampler for the gamma-via-Cauchy rejection example in
 * `rejection.ts`.
 */
export function cauchyQuantile(u: number, p: CauchyParams): number {
  return p.b * Math.tan(Math.PI * (u - 0.5)) + p.c;
}

export function cauchySample(rng: Rng, p: CauchyParams): number {
  return inverseTransformSample(rng, (u) => cauchyQuantile(u, p));
}

export interface BoxMullerAttempt {
  readonly z1: number;
  readonly z2: number;
  readonly r2: number;
  readonly accepted: boolean;
}

export interface BoxMullerTrace {
  readonly attempts: readonly BoxMullerAttempt[];
  readonly y1: number;
  readonly y2: number;
}

/**
 * PRML 11.10-11.12, the Box-Muller/Marsaglia-polar method: uniform points in the unit
 * square are accepted only inside the unit circle (Figure 11.3), then mapped to a pair
 * of independent standard normals. This is the same algorithm `rng.ts`'s `standardNormal`
 * runs, but that function only returns one deviate per call and caches the other; this
 * one returns every attempt (rejected pairs included) because Figure 11.3's picture is
 * exactly the geometry of those attempts landing inside versus outside the disk.
 */
export function boxMullerTrace(rng: Rng): BoxMullerTrace {
  const attempts: BoxMullerAttempt[] = [];
  let z1 = 0;
  let z2 = 0;
  let r2 = 0;
  for (;;) {
    z1 = 2 * rng.next() - 1;
    z2 = 2 * rng.next() - 1;
    r2 = z1 * z1 + z2 * z2;
    const accepted = r2 > 0 && r2 <= 1;
    attempts.push({ z1, z2, r2, accepted });
    if (accepted) break;
  }
  const factor = Math.sqrt((-2 * Math.log(r2)) / r2);
  return { attempts, y1: z1 * factor, y2: z2 * factor };
}
