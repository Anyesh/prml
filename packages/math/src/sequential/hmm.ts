import type { Mat, Vec } from '../types.js';
import { mvnPdf } from '../distributions/mvn.js';

/** Parameters of a homogeneous hidden Markov model with `K` discrete latent states (PRML 13.7-13.8). */
export interface HmmParams {
  /** `pi[k] = p(z1k = 1)`, PRML 13.8. */
  readonly pi: Vec;
  /** `A[j][k] = p(zn,k = 1 | zn-1,j = 1)`, PRML 13.7. Rows sum to one. */
  readonly A: Mat;
}

/** A Gaussian emission component, PRML's `phi_k` for `p(x | z_k = 1) = N(x | mu_k, Sigma_k)`. */
export interface HmmGaussianComponent {
  readonly mean: Vec;
  readonly cov: Mat;
}

export interface HmmGaussianParams extends HmmParams {
  readonly components: readonly HmmGaussianComponent[];
}

/**
 * `B[n][k] = p(x_n | z_n = k)` (PRML 13.9 specialised to Gaussian emissions), the one
 * quantity every recursion in `forwardBackward.ts` and `viterbi.ts` needs from the
 * observations. Kept in raw probability space, not log, because the book's own point
 * about scaling (13.2.4) is that the fix for underflow is re-normalising the *recursion*,
 * not taking logs of a sum of products.
 */
export function hmmGaussianEmissionMatrix(data: Mat, components: readonly HmmGaussianComponent[]): number[][] {
  return data.map((x) => components.map((c) => mvnPdf(x, { mean: c.mean, cov: c.cov })));
}
