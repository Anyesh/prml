import type { Vec } from '../types.js';

/**
 * PRML 1.98, in nats. `p_i = 0` contributes 0 rather than `NaN` from `0 * -Infinity`,
 * matching the book's own convention (page 50) that `lim p ln p = 0` as `p -> 0`.
 */
export function discreteEntropy(p: Vec): number {
  let sum = 0;
  for (const pi of p) {
    if (pi > 0) sum -= pi * Math.log(pi);
  }
  return sum;
}

/**
 * PRML 1.110, the closed form reached by maximising differential entropy under a fixed
 * mean and variance (1.108-1.109). Unlike `discreteEntropy` this can go negative: a
 * Gaussian narrower than `1 / (2*pi*e)` in variance has negative differential entropy,
 * because differential entropy measures relative to a coordinate scale rather than
 * counting states.
 */
export function differentialEntropyGaussian(sigma2: number): number {
  return 0.5 * (1 + Math.log(2 * Math.PI * sigma2));
}
