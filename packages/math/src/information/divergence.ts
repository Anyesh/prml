import type { Vec } from '../types.js';
import type { NormalParams } from '../distributions/normal.js';

/**
 * PRML 1.113 restricted to a discrete `p`, `q` over the same support: `sum p_i ln(p_i/q_i)`.
 * `p_i = 0` contributes 0, the same convention as `discreteEntropy`; `q_i = 0` with
 * `p_i > 0` is left to produce `Infinity` rather than guarded away, because that case is
 * the correct answer (a code built for `q` cannot represent an event `q` assigns zero
 * probability to).
 */
export function klDivergence(p: Vec, q: Vec): number {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    const pi = p[i]!;
    if (pi > 0) sum += pi * Math.log(pi / q[i]!);
  }
  return sum;
}

/**
 * Closed form for `KL(N(mu_p, sigma_p^2) || N(mu_q, sigma_q^2))`, PRML Exercise 1.30.
 * Parameterised by variance (`sigma2`), matching `NormalParams` throughout this package.
 */
export function gaussianKlDivergence(p: NormalParams, q: NormalParams): number {
  const meanTerm = (p.sigma2 + (p.mu - q.mu) ** 2) / (2 * q.sigma2);
  return 0.5 * Math.log(q.sigma2 / p.sigma2) + meanTerm - 0.5;
}
