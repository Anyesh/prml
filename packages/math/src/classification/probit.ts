import { dot, quadForm } from '../linalg/index.js';
import { normalCdf } from '../distributions/index.js';
import { sigmoid } from '../numeric.js';
import type { Mat, Vec } from '../types.js';

/** PRML 4.114: `Φ(a)`, the standard normal CDF used as a sigmoid-shaped activation function. */
export function probit(a: number): number {
  return normalCdf(a, { mu: 0, sigma2: 1 });
}

/** PRML 4.154: the rescaling that turns the probit approximation of `σ(a)` into a closed form. */
export function kappa(variance: number): number {
  return 1 / Math.sqrt(1 + (Math.PI * variance) / 8);
}

/**
 * PRML 4.153: closed-form approximation to `∫ σ(a) N(a | mean, variance) da`, a convolution
 * with no exact analytic form on its own. Approximating `σ` by a rescaled probit turns the
 * convolution of two Gaussians-of-a-kind back into a single probit, which is where `kappa`
 * comes from.
 */
export function logisticGaussianConvolution(mean: number, variance: number): number {
  return sigmoid(kappa(variance) * mean);
}

/**
 * PRML 4.149-4.150 then 4.155: the activation `a = wᵀφ` has mean and variance `μ_a`, `σ²_a`
 * under the Laplace posterior `q(w)`; feeding those through `logisticGaussianConvolution`
 * gives the marginalised predictive probability of C1, rather than the point estimate
 * `σ(wMAPᵀφ)` that ignores how uncertain `w` still is.
 */
export function laplaceLogisticPredictive(phi: Vec, posterior: { mean: Vec; covariance: Mat }): number {
  const mu = dot(phi, posterior.mean);
  const variance = quadForm(phi, posterior.covariance, phi);
  return logisticGaussianConvolution(mu, variance);
}
