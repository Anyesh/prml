import type { Mat } from '../types.js';
import type { NormalParams } from '../distributions/normal.js';
import { inverse } from '../linalg/decompose.js';

/** A jointly Gaussian `p(z1, z2)` given by its precision matrix, PRML 10.10. */
export interface BivariateGaussianPrecision {
  readonly mean: readonly [number, number];
  readonly precision: readonly [readonly [number, number], readonly [number, number]];
}

export interface FactorizedGaussian {
  readonly q1: NormalParams;
  readonly q2: NormalParams;
}

/**
 * One coordinate-ascent sweep of PRML 10.12-10.15, minimising `KL(q1 q2 || p)`. Each
 * factor's precision is fixed at the corresponding diagonal block of `p`'s precision
 * forever; only the means move, and they move towards each other's current mean.
 * `current` supplies the previous sweep's means, so the caller can seed away from the true
 * mean and watch coordinate ascent correct it (PRML never needs to, because the fixed
 * point is exact, but the approach path is what a step-through widget shows).
 */
export function factorizedGaussianForwardKlStep(
  p: BivariateGaussianPrecision,
  current: FactorizedGaussian,
): FactorizedGaussian {
  const [mu1, mu2] = p.mean;
  const [[lambda11, lambda12], [lambda21, lambda22]] = p.precision;

  const m1 = mu1 - (lambda12 / lambda11) * (current.q2.mu - mu2);
  const q1: NormalParams = { mu: m1, sigma2: 1 / lambda11 };

  const m2 = mu2 - (lambda21 / lambda22) * (m1 - mu1);
  const q2: NormalParams = { mu: m2, sigma2: 1 / lambda22 };

  return { q1, q2 };
}

/**
 * Runs `factorizedGaussianForwardKlStep` for `iterations` sweeps from `initial`, returning
 * every intermediate factorisation so a widget can step through the approach to the fixed
 * point. `initial` need not be close: the fixed point (`E[z1] = mu1`, `E[z2] = mu2`) is
 * unique and coordinate ascent reaches it regardless of where it starts, because 10.9 is a
 * genuine coordinate ascent on a bound that is quadratic (hence unimodal) in each factor.
 */
export function factorizedGaussianForwardKlFit(
  p: BivariateGaussianPrecision,
  initial: FactorizedGaussian,
  iterations: number,
): FactorizedGaussian[] {
  const trajectory: FactorizedGaussian[] = [initial];
  let current = initial;
  for (let i = 0; i < iterations; i++) {
    current = factorizedGaussianForwardKlStep(p, current);
    trajectory.push(current);
  }
  return trajectory;
}

/**
 * The exact minimiser of `KL(p || q1 q2)`, PRML 10.17: each factor equals the true
 * marginal of `p`. Closed form and requires no iteration, unlike the forward direction,
 * because the reverse KL's optimal factor is simply the marginal whatever the other
 * factor is set to.
 */
export function factorizedGaussianReverseKl(p: BivariateGaussianPrecision): FactorizedGaussian {
  const cov = inverse(p.precision as unknown as Mat);
  const [mu1, mu2] = p.mean;
  return {
    q1: { mu: mu1, sigma2: cov[0]![0]! },
    q2: { mu: mu2, sigma2: cov[1]![1]! },
  };
}

/** Diagonal of the true covariance, the variance each reverse-KL factor reports (PRML 10.17). */
export function trueMarginalVariances(p: BivariateGaussianPrecision): readonly [number, number] {
  const cov = inverse(p.precision as unknown as Mat);
  return [cov[0]![0]!, cov[1]![1]!];
}

/** `2 x 2` correlated precision matrix from a marginal-variance pair and a correlation coefficient, for the widget's slider. */
export function correlatedPrecision(mean: readonly [number, number], variances: readonly [number, number], rho: number): BivariateGaussianPrecision {
  const [v1, v2] = variances;
  const s1 = Math.sqrt(v1);
  const s2 = Math.sqrt(v2);
  const cov: Mat = [
    [v1, rho * s1 * s2],
    [rho * s1 * s2, v2],
  ];
  const precision = inverse(cov);
  return { mean, precision: [[precision[0]![0]!, precision[0]![1]!], [precision[1]![0]!, precision[1]![1]!]] };
}

/** Row-major 2x2 covariance recovered from a `BivariateGaussianPrecision`, for drawing `p`'s own contour. */
export function bivariateCovariance(p: BivariateGaussianPrecision): Mat {
  return inverse(p.precision as unknown as Mat);
}
